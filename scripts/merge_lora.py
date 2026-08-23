"""把 LoRA checkpoint merge 回基座并保存（重建 out/models/Qwen--Qwen3.5-9B-Base-LoRA-ckpt1250）。

精简仓库时删除了 merged 权重，此脚本是正式重建路径。用法（在 proj_3 目录）：

  PYTHONPATH=/media/data2/tangzc/proj_3 python scripts/merge_lora.py \
    --base out/models/Qwen--Qwen3.5-9B-Base \
    --adapter out/checkpoints/lora/checkpoint-1250 \
    --output out/models/Qwen--Qwen3.5-9B-Base-LoRA-ckpt1250

加载口径与训练/评测一致：text_only=True（hf_loader，FSDP2 adapter 键
base_model.model.model.layers.* 只在文本视图上匹配，见 2026-08-21 实测）；
merge_and_unload 后以 bf16 保存，键布局 = model.language_model.*（与 base HF 一致）。
vLLM 服务还需多模态外壳 config，重建方式见 serving/quant_export.py 的
rebuild_multimodal_config（--shell-dir 指向 base HF 目录）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from model.hf_loader import load_model, load_processor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True, help="基座 HF 快照目录")
    parser.add_argument("--adapter", type=Path, required=True, help="PEFT checkpoint 目录")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = load_model(
        str(args.base),
        torch_dtype="bfloat16",
        text_only=True,
        peft_adapter=str(args.adapter),
    )
    merged = model.merge_and_unload()
    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.output), safe_serialization=True)
    processor = load_processor(str(args.base))
    processor.save_pretrained(str(args.output))
    print(f"merged saved to {args.output}")


if __name__ == "__main__":
    main()
