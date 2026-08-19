from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from config import DEFAULT_MAX_LENGTH, DEFAULT_SEED, PATHS


def is_main_process() -> bool:
    try:
        return int(os.environ.get("LOCAL_RANK", "0")) == 0
    except ValueError:
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Guarded full-finetune smoke test (optional, not the mainline)")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--train-data", type=Path, default=PATHS.processed / "sft" / "train.jsonl")
    parser.add_argument("--output-dir", type=Path, default=PATHS.checkpoints / "full_smoke")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--max-records", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help=">1 时 Trainer 会对非末位 micro-batch 包 no_sync，FSDP1 在 no_sync 下不回收集齐的整层参数，"
        "9B 模型首个 backward 即 22.8GiB OOM（实测 allocator snapshot：27 个 ~400MiB 无 frame 块 + 4.6GiB 缓冲）。"
        "24GB 卡上只能取 1。",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--no-fsdp", action="store_true", help="Disable automatic FSDP on multi-GPU")
    parser.add_argument("--confirm", action="store_true", help="Acknowledge that full SFT is not the mainline")
    args = parser.parse_args()
    if not args.confirm:
        raise SystemExit("Full SFT is an optional smoke test. Re-run with --confirm.")

    import torch
    from transformers import Trainer, TrainingArguments, set_seed

    from model.hf_loader import load_model, load_processor
    from train.sft_data import CausalCollator, ChatJsonlDataset

    set_seed(DEFAULT_SEED)
    processor = load_processor(args.model)
    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(args.model, text_only=True)
    model.config.use_cache = False

    train_dataset = ChatJsonlDataset(args.train_data, tokenizer, args.max_length, args.max_records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if is_main_process():
        (args.output_dir / "run_config.json").write_text(json.dumps(vars(args), default=str, ensure_ascii=False, indent=2), encoding="utf-8")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_fsdp = world_size > 1 and not args.no_fsdp
    fsdp_config = None
    if use_fsdp:
        layer_classes = sorted({
            module.__class__.__name__
            for name, module in model.named_modules()
            if ".layers." in name and module.__class__.__name__.endswith(("DecoderLayer", "Block"))
        })
        if not layer_classes:
            raise RuntimeError("FSDP requested but no decoder layer class was detected; pass --no-fsdp")
        # root 单元包含 embed_tokens(1017M) + lm_head(1017M)，backward 时 AC 重算 forward
        # 会 unshard 整个 root（3.79GiB 全量缓冲，allocator snapshot 实测），与 bnb 账外
        # scratch（~4.2GiB，raw cudaMalloc 不进 torch 账本）叠加后 4096 必 OOM。
        # 把 Embedding 单独包装后 root 瞬时缓冲降到 ~1.9GiB：峰值 18.77GiB → 16.95GiB。
        layer_classes.append("Embedding")
        fsdp_config = {
            # 实测结论（4×3090 + transformers 5.15/accelerate 1.14）：
            # - FSDP2 会把可训练参数上转 fp32 master 且 Trainer 不注入 MixedPrecisionPolicy，
            #   bnb 优化器又不兼容 DTensor → 9B 全参在 24GB 卡上无法完成一步；
            # - FSDP1 flat-param 分片 + bf16 原生计算 + paged 8-bit Adam 是唯一跑通的组合；
            # - gradient_accumulation_steps>1 时 Trainer 对非末位 micro-batch 包 no_sync，
            #   FSDP1 no_sync 跳过 post-backward 参数回收 → 整层参数在 backward 全程驻留，
            #   allocator snapshot 实测 27×~400MiB 集齐块 + 4.6GiB 缓冲，首个 backward 即 22.8GiB OOM；
            #   因此默认 grad-acc=1。
            # - bnb paged 优化器有 ~4.2GiB 账外 GPU scratch（raw cudaMalloc，不进 torch 账本，
            #   首次 optimizer.step 后常驻），叠加 root 单元 3.79GiB unshard 缓冲后 4096 OOM；
            #   Embedding 单独包装后峰值 16.95GiB@4096，23.56GiB 卡上有 ~1.9GiB 余量。
            "version": 1,
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "fsdp_transformer_layer_cls_to_wrap": ",".join(layer_classes),
            "limit_all_gathers": True,
            # FSDP 原生激活检查点：避免 transformers 通用 checkpoint 在 backward 的冗余 AllGather。
            "activation_checkpointing": True,
            # flat-param 分片（bnb 优化器只支持这种形式；orig-param 下梯度/状态为全模型大小，必 OOM）。
            "use_orig_params": False,
            "backward_prefetch": "BACKWARD_PRE",
        }
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        # transformers ≥5.x 移除了 warmup_ratio；本冒烟固定跑 max_steps 步，按 3% 换算等价 warmup_steps。
        warmup_steps=max(1, int(args.max_steps * 0.03)),
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.max_steps,
        save_total_limit=1,
        gradient_checkpointing=not use_fsdp,
        # 注意：不能开 bf16/fp16。accelerate 在 mixed_precision != "no" 时会把全部可训练参数
        # 上转为 fp32 master（FSDP1/FSDP2 均如此，fsdp_utils.py），且 Trainer 不会注入
        # MixedPrecisionPolicy，导致 9B 模型按 fp32 计算在 24GB 卡上直接 OOM。
        # 关闭 autocast 后参数保持加载时的 bf16 原生计算（每卡 shard 4.17GiB，实测峰值 18.8GiB）。
        bf16=False,
        fp16=False,
        remove_unused_columns=False,
        report_to=[],
        # 8-bit paged Adam：fp32 状态（每卡约 17.8GiB）放不下 24GB 卡；bnb 与 FSDP1 flat-param 兼容。
        optim="paged_adamw_8bit",
        # transformers ≥5.15 已弃用 fsdp 字符串形式；fsdp=True 默认 full_shard auto_wrap。
        fsdp=use_fsdp,
        fsdp_config=fsdp_config,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=CausalCollator(tokenizer),
    )
    trainer.train()
    # FSDP 的 save_model 内部 state_dict() 是全 rank 集合通信（all-gather），必须所有 rank
    # 一起调用，文件只由 rank 0 落盘。只让 rank 0 调用会让其余 rank 直接退出、rank 0 在
    # NCCL 忙等死锁（实测 99.7% CPU 空转、零磁盘 I/O）。
    trainer.save_model(str(args.output_dir / "final"))
    if is_main_process():
        tokenizer.save_pretrained(str(args.output_dir / "final"))
        print(f"saved smoke checkpoint to {args.output_dir / 'final'}")
    if torch.cuda.is_available():
        print(f"[memory] final reserved={torch.cuda.max_memory_reserved() / 2**30:.2f}GiB "
              f"allocated={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB", flush=True)


if __name__ == "__main__":
    main()
