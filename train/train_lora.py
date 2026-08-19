from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from config import DEFAULT_MAX_LENGTH, DEFAULT_SEED, PATHS
from model.hf_loader import load_model, load_processor
from model.lora_parity import discover_target_modules
from train.sft_data import CausalCollator, ChatJsonlDataset


def build_target_modules(model, requested: str | None) -> list[str] | str:
    if requested:
        return [item.strip() for item in requested.split(",") if item.strip()]
    try:
        return discover_target_modules(model)
    except RuntimeError:
        return "all-linear"


def is_main_process() -> bool:
    try:
        return int(os.environ.get("LOCAL_RANK", "0")) == 0
    except ValueError:
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Qwen3.5 with PEFT LoRA/QLoRA")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--train-data", type=Path, default=PATHS.processed / "sft" / "train.jsonl")
    parser.add_argument("--eval-data", type=Path, default=PATHS.processed / "sft" / "val.jsonl")
    parser.add_argument("--output-dir", type=Path, default=PATHS.checkpoints / "lora")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--eval-records", type=int, default=280, help="Eval 集读取条数（CMB val 为 280）")
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None, help="从 trainer checkpoint 目录恢复训练")
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--target-modules", default=None)
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-fsdp", action="store_true", help="Disable automatic FSDP on multi-GPU")
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

    class MemoryCallback(TrainerCallback):
        """在每个 eval/save 节点记录 CUDA 显存峰值（README M3 的显存峰值日志）。"""

        def on_evaluate(self, args, state, control, **kwargs):
            if torch.cuda.is_available():
                print(f"[memory] step={state.global_step} reserved={torch.cuda.max_memory_reserved() / 2**30:.2f}GiB "
                      f"allocated={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB", flush=True)

    set_seed(args.seed)
    processor = load_processor(args.model, trust_remote_code=args.trust_remote_code)
    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(args.model, qlora=args.qlora, trust_remote_code=args.trust_remote_code, text_only=True)
    if args.qlora:
        model = prepare_model_for_kbit_training(model)
    targets = build_target_modules(model, args.target_modules)
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.config.use_cache = False

    train_dataset = ChatJsonlDataset(args.train_data, tokenizer, args.max_length, args.max_records)
    eval_dataset = ChatJsonlDataset(args.eval_data, tokenizer, args.max_length, args.eval_records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if is_main_process():
        (args.output_dir / "run_config.json").write_text(json.dumps(vars(args), default=str, ensure_ascii=False, indent=2), encoding="utf-8")

    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    # 多卡默认走 FSDP；QLoRA 时 4bit 基座不参与分片，FSDP 只包装 LoRA adapter，
    # 这与 DDP + bnb 量化层不兼容（DDP 要求无量化层）相比是可运行的组合。
    use_fsdp = world_size > 1 and not args.no_fsdp
    fsdp_config = None
    if use_fsdp:
        layer_classes = sorted({
            module.__class__.__name__
            for name, module in model.named_modules()
            if ".layers." in name and module.__class__.__name__.endswith(("DecoderLayer", "Block"))
        })
        if not layer_classes:
            raise RuntimeError("FSDP requested but no decoder layer class was detected; pass --no-fsdp for a single-GPU smoke test")
        fsdp_config = {
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "fsdp_transformer_layer_cls_to_wrap": ",".join(layer_classes),
            "limit_all_gathers": True,
            # FSDP 原生激活检查点：避免 transformers 通用 checkpoint 在 backward 的冗余 AllGather。
            "activation_checkpointing": True,
        }
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        # transformers ≥5.x 移除了 warmup_ratio；max_steps 模式下按 3% 换算，epochs 模式不做 warmup。
        warmup_steps=max(1, int(args.max_steps * 0.03)) if (args.max_steps and args.max_steps > 0) else 0,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        gradient_checkpointing=not use_fsdp,
        bf16=use_bf16,
        fp16=bool(torch.cuda.is_available() and not use_bf16),
        remove_unused_columns=False,
        report_to=[],
        ddp_find_unused_parameters=False,
        optim="paged_adamw_8bit" if args.qlora else "adamw_torch",
        # transformers ≥5.15 已弃用 fsdp 字符串形式；fsdp=True 默认 full_shard auto_wrap。
        fsdp=use_fsdp,
        fsdp_config=fsdp_config,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CausalCollator(tokenizer),
        callbacks=[MemoryCallback()],
    )
    resume_from_checkpoint = str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    # 不调用 trainer.save_model(output_dir/"final")：FSDP2 下全模型 state_dict 是跨 rank
    # 集合通信，实测 rank 1-3 正常退出后 rank 0 在 NCCL 忙等（99.9% CPU、零磁盘 I/O），
    # final/ 永不落盘。Trainer 训练结束自动保存的 checkpoint-{step} 已含完整适配器产物
    # （adapter_model.safetensors + adapter_config.json + tokenizer + optimizer），
    # 下游 eval 直接指向该目录即可。
    if is_main_process():
        print(f"training done; load adapter from the latest checkpoint dir under {args.output_dir}")
    if torch.cuda.is_available():
        print(f"[memory] final reserved={torch.cuda.max_memory_reserved() / 2**30:.2f}GiB "
              f"allocated={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB", flush=True)


if __name__ == "__main__":
    main()
