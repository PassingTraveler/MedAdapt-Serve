from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

from config import PATHS


def quantize_gptq(model_dir: Path, output_dir: Path, bits: int, group_size: int) -> None:
    """通过 optimum/auto-gptq 导出 W4A16 GPTQ checkpoint（需安装 auto-gptq 与 optimum）。"""
    try:
        from optimum.gptq import GPTQQuantizer
    except ImportError as exc:
        raise SystemExit("GPTQ export requires `pip install auto-gptq optimum`; this is not part of requirements-train/serve.") from exc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    quantizer = GPTQQuantizer(bits=bits, dataset="c4", model_seqlen=1024, group_size=group_size)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=torch.float16, device_map="auto")
    quantized = quantizer.quantize_model(model, tokenizer)
    output_dir.mkdir(parents=True, exist_ok=True)
    quantized.save_pretrained(str(output_dir), safetensors=True)
    tokenizer.save_pretrained(str(output_dir))
    print(f"GPTQ checkpoint saved to {output_dir}")


def verify_quant_dir(model_dir: Path) -> dict:
    """校验一个量化 checkpoint 能被 transformers 真实加载，并给出权重统计。"""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()
    total = 0
    for parameter in model.parameters():
        total += parameter.numel() * parameter.element_size()
    ids = tokenizer("请只输出答案字母。", return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**ids).logits.float()
    finite = bool(torch.isfinite(logits).all().item())
    return {
        "model_dir": str(model_dir),
        "model_type": getattr(config, "model_type", None),
        "weight_bytes": total,
        "logits_finite": finite,
        "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and verify a genuinely loadable quantization format")
    parser.add_argument("--model", type=Path, default=PATHS.models / "Qwen--Qwen3.5-9B-Base")
    parser.add_argument("--format", choices=("gptq",), default="gptq", help="vLLM 可加载的真实量化格式（主线为 W4A16 GPTQ）")
    parser.add_argument("--output-dir", type=Path, default=PATHS.models / "Qwen--Qwen3.5-9B-Base-GPTQ-W4A16")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--verify-only", type=Path, default=None, help="只校验已有量化目录，不导出")
    parser.add_argument("--vllm-dry-run", action="store_true", help="导出后用 vllm serve 命令做参数级冒烟检查")
    args = parser.parse_args()

    result: dict = {}
    if args.verify_only is not None:
        result = verify_quant_dir(args.verify_only)
    else:
        if args.format == "gptq":
            quantize_gptq(args.model, args.output_dir, args.bits, args.group_size)
        else:
            raise SystemExit(f"unsupported format: {args.format}")
        result = verify_quant_dir(args.output_dir)
    if args.vllm_dry_run:
        command = [
            "vllm", "serve", str(args.verify_only or args.output_dir),
            "--max-model-len", "4096", "--quantization", "gptq", "--dry-run",
        ]
        print(" ".join(shlex.quote(item) for item in command))
        try:
            result["vllm_dry_run_returncode"] = subprocess.call(command, timeout=120)
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            result["vllm_dry_run_error"] = str(exc)
    output = PATHS.manifests / "quant_export.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
