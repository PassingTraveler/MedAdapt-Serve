from __future__ import annotations

import argparse
import json
from pathlib import Path

from model.hf_loader import load_model, load_processor


PREFERRED_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "out_proj",
    "gate_proj", "up_proj", "down_proj", "in_proj_qkv", "in_proj_z",
    "in_proj_b", "in_proj_a",
)


def discover_target_modules(model) -> list[str] | str:
    # QLoRA 时 nn.Linear 已被替换为 bnb.nn.Linear4bit，类名不再是 "Linear"；
    # 两者都要匹配，找不到时回退 PEFT 的 "all-linear"。
    names = {
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if module.__class__.__name__ in ("Linear", "Linear4bit")
    }
    selected = [suffix for suffix in PREFERRED_SUFFIXES if suffix in names]
    if not selected:
        return "all-linear"
    return selected


def logits_of(model, tokenizer, text: str, max_length: int = 64):
    import torch

    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"].to(model.device)
    with torch.no_grad():
        return model(input_ids=ids).logits.float()


def max_abs_diff(a, b) -> float:
    return float((a - b).abs().max().item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Check PEFT LoRA injection and merge/unmerge parity")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, default=Path("out/manifests/lora_parity.json"))
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--text", default="请只输出答案字母。", help="固定输入文本")
    parser.add_argument("--max-length", type=int, default=64)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model

    processor = load_processor(args.model)
    tokenizer = getattr(processor, "tokenizer", processor)
    model = load_model(args.model, qlora=args.qlora, device_map="auto", text_only=True)
    model.eval()
    base_logits = logits_of(model, tokenizer, args.text, args.max_length)

    targets = discover_target_modules(model)
    config = LoraConfig(r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05, target_modules=targets, task_type="CAUSAL_LM")
    peft_model = get_peft_model(model, config)
    peft_model.eval()
    injected_logits = logits_of(peft_model, tokenizer, args.text, args.max_length)
    trainable = sum(parameter.numel() for parameter in peft_model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in peft_model.parameters())

    adapter_dir = args.output.parent / "lora_parity_adapter"
    peft_model.save_pretrained(str(adapter_dir))

    # 保存/恢复一致性：重新加载一份基座并挂载 adapter。
    base_reloaded = load_model(args.model, qlora=args.qlora, device_map="auto", text_only=True)
    reloaded = PeftModel.from_pretrained(base_reloaded, str(adapter_dir))
    reloaded.eval()
    reloaded_logits = logits_of(reloaded, tokenizer, args.text, args.max_length)

    # merge/unmerge parity：bf16 下可 merge；4bit 基座不支持 merge，记录跳过原因。
    merged_info: dict = {"skipped": None, "diff_vs_adapter": None}
    if not args.qlora:
        merged_model = reloaded.merge_and_unload()
        merged_model.eval()
        merged_logits = logits_of(merged_model, tokenizer, args.text, args.max_length)
        merged_info = {"skipped": None, "diff_vs_adapter": max_abs_diff(reloaded_logits, merged_logits)}
    else:
        merged_info = {"skipped": "merge_and_unload is not supported on a 4-bit quantized base", "diff_vs_adapter": None}

    tolerance = 1e-2 if args.qlora else 1e-3
    diffs = {
        "inject_vs_base": max_abs_diff(base_logits, injected_logits),
        "reload_vs_inject": max_abs_diff(injected_logits, reloaded_logits),
        **merged_info,
    }
    result = {
        "model": args.model,
        "qlora": args.qlora,
        "target_modules": targets,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_ratio": trainable / total if total else 0,
        "tolerance": tolerance,
        "logits_max_abs_diff": diffs,
        "checks_passed": {
            # LoRA 零初始化注入后输出应与基座一致（首项在阈值内）。
            "zero_init_parity": diffs["inject_vs_base"] < tolerance,
            "save_reload_parity": diffs["reload_vs_inject"] < tolerance,
            "merge_parity": diffs.get("diff_vs_adapter") is not None and diffs["diff_vs_adapter"] < tolerance,
        },
        "adapter_dir": str(adapter_dir),
    }
    if torch.cuda.is_available():
        result["cuda_device"] = torch.cuda.get_device_name()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
