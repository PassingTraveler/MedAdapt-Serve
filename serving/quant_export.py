from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from config import PATHS

# 量化校准语料:取本地 CMB 训练集前 N 条(题面+选项)。
# 不用 optimum 的 dataset="c4":那会触发完整 C4 下载(数百 GB),本机不可行;
# 且医学领域内语料校准对下游医疗问答任务更合适。
CALIBRATION_RECORDS = 512


def load_calibration(path: Path, limit: int = CALIBRATION_RECORDS) -> list[str]:
    samples: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        text = f"{record.get('question', '')}\n{record.get('option', '')}".strip()
        if text:
            samples.append(text)
        if len(samples) >= limit:
            break
    if not samples:
        raise SystemExit(f"calibration corpus empty: {path}")
    return samples


def remap_language_model_keys(output_dir: Path) -> None:
    """把量化导出的键幂等规范化为 HF base 布局 model.language_model.*。

    实测（2026-08-21，三个目录逐一核对 safetensors 键）：
    base HF 目录与 vllm serving 目录的键都是 model.language_model.embed_tokens /
    model.language_model.layers.N.* / model.language_model.norm——**没有中间的
    "model" 段**；vllm 0.19.0 从 serving 目录成功加载，即按此布局读取。
    AutoModelForCausalLM（文本视图 Qwen3_5ForCausalLM）保存的键同样带
    language_model 段（merged 目录实测），但 optimum 各版本行为不一，可能保存成
    无容器段的 model.layers.*；两种输入都规范化到 base 布局，已正确的键保持原样。
    逐键重写 model.safetensors（单文件，量化导出时 max_shard_size 已放宽）。
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    file = output_dir / "model.safetensors"
    state: dict[str, object] = {}
    with safe_open(str(file), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            new_key = key
            for old, new in (
                # 纠错：早期版本曾误把 base 布局写成多一段 model（vllm 读取会失败）。
                ("model.language_model.model.", "model.language_model."),
                ("model.embed_tokens", "model.language_model.embed_tokens"),
                ("model.layers.", "model.language_model.layers."),
                ("model.norm", "model.language_model.norm"),
            ):
                if key.startswith(old):
                    new_key = new + key[len(old):]
                    break
            state[new_key] = handle.get_tensor(key)
    save_file(state, str(file))


def rebuild_multimodal_config(source_dir: Path, output_dir: Path) -> None:
    """把文本视图的 config.json 换回多模态外壳（保留 quantization_config）。

    vllm 0.19.0 registry 只有 Qwen3_5ForConditionalGeneration，没有文本视图类；
    权重只含语言部分，配合 --language-model-only 加载。外壳取自源模型目录
    （architectures/text_config/vision_config 原样），量化参数来自导出的 config。
    """
    import json
    import shutil

    source_config = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
    quant_config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    merged_quant = quant_config["quantization_config"]
    # vllm 0.19.0 只认 checkpoint_format 键识别 gptq_v2（optimum 写的是 format；
    # transformers 5.x 的 GPTQConfig 也兼容该 legacy 键，两边无冲突）。
    if merged_quant.get("format") == "gptq_v2" and "checkpoint_format" not in merged_quant:
        merged_quant["checkpoint_format"] = "gptq_v2"
    # vllm 的 GPTQ 只支持 fp16 激活（bf16 直接 ValidationError，实测）；量化导出的
    # activation dtype 本就是 fp16，外壳里的 torch_dtype 需同步，否则 serve 默认
    # dtype=auto 会按 bf16 启动失败。
    source_config["torch_dtype"] = "float16"
    source_config["quantization_config"] = merged_quant
    (output_dir / "config.json").write_text(
        json.dumps(source_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # 多模态外壳还需要 processor 配置（AutoProcessor 在缺 processor_config.json 时
    # 可能直接报错），从源目录原样复制。
    for name in ("processor_config.json", "preprocessor_config.json"):
        source = source_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)


def quantize_gptq(model_dir: Path, output_dir: Path, bits: int, group_size: int,
                  shell_dir: Path | None = None) -> None:
    """通过 optimum + gptqmodel 导出 W4A16 GPTQ checkpoint（需安装 gptqmodel 与 optimum）。

    optimum 2.3.0 起 GPTQ 后端从 auto-gptq 切到 gptqmodel（auto-gptq 0.7.1 在
    transformers 5.x 下无法 import）。gptqmodel 7.3.4 硬性要求 numpy==2.2.6、
    protobuf>=7.34、torchao>=0.16（torchao 0.18 需要 torch 2.9+，本机 torch 2.8
    须配 torchao 0.17.0，实测可导入）。"""
    try:
        from optimum.gptq import GPTQQuantizer
    except ImportError as exc:
        raise SystemExit("GPTQ export requires `pip install gptqmodel optimum`; this is not part of requirements-train/serve.") from exc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    calibration = load_calibration(PATHS.processed / "cmb" / "train.jsonl")
    quantizer = GPTQQuantizer(bits=bits, dataset=calibration, model_seqlen=512, group_size=group_size,
                              # 512:CMB 题面+选项均 <512 token，校准足够；1024 时输入缓存翻倍，
                              # 双卡 24GB 下量化到 block 22/32 实测 OOM。
                              cache_block_outputs=False,
                              # AutoModelForCausalLM 把 qwen3_5 解析成文本视图
                              # Qwen3_5ForCausalLM（transformers 5.x 的 VLM 兼容映射）：
                              # 无视觉塔省 ~2GiB，单卡 24GB 可容纳。多模态类双卡跑
                              # gptqmodel 跨卡 Hessian 合并有已知 bug（源码注释
                              # "multi-3090 using P2P"），实测 OOM。保存后由
                              # remap_language_model_keys + rebuild_multimodal_config
                              # 还原成 vllm 可加载的布局与 config。
                              block_name_to_quantize="model.layers",
                              # 排除 32 维状态投影层（linear_attn.in_proj_a/b，4096→32）：
                              # gptqmodel 加载时按 desc_act=False+对称给它们选
                              # MarlinLinear，out_features 32 不被 64 整除直接抛
                              # NotImplementedError（实测）。这些层共 48 个、
                              # ~6.3M 参数（0.07%），保持 fp16 对精度/体积无实质影响。
                              modules_in_block_to_quantize=[
                                  ["linear_attn.out_proj", "linear_attn.in_proj_qkv",
                                   "linear_attn.in_proj_z", "mlp.gate_proj",
                                   "mlp.up_proj", "mlp.down_proj"],
                              ])
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.float16, device_map="auto")
    # optimum 2.3.0 的 quantize_model 假定 model.config.use_cache 存在；
    # transformers 5.x 的多模态 Qwen3_5Config 顶层无该属性（在 text_config 里）。
    if not hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    quantized = quantizer.quantize_model(model, tokenizer)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 单文件（不按默认 5GB 切片），后续键重映射无需处理分片。
    quantized.save_pretrained(str(output_dir), safetensors=True, max_shard_size="20GB")
    tokenizer.save_pretrained(str(output_dir))
    # 先按文本视图验证（此时键布局与 config 自洽）：若量化权重无法被 transformers
    # 还原或前向不 finite，后续 remap 只会掩盖问题。
    text_view = verify_quant_dir(output_dir)
    if not text_view["logits_finite"]:
        raise SystemExit(f"quantized checkpoint failed transformers verification: {text_view}")
    remap_language_model_keys(output_dir)
    # 多模态外壳（vllm 0.19.0 只有 Qwen3_5ForConditionalGeneration 架构，且外壳里的
    # vision_config 无法从文本视图 config 自构造）。缺省自动选择：源 config 已是
    # 多模态（model_type=qwen3_5）则用自身；否则用 base HF 目录（其 config 与
    # serving 目录实测等价，仅字段顺序不同）。
    if shell_dir is None:
        source_type = json.loads((model_dir / "config.json").read_text(encoding="utf-8")).get("model_type")
        shell_dir = model_dir if source_type == "qwen3_5" else PATHS.models / "Qwen--Qwen3.5-9B-Base"
    rebuild_multimodal_config(shell_dir, output_dir)
    # remap 后按多模态类再验：语言部分权重应原样加载（视觉塔权重缺失只会随机初始化，
    # 不参与文本前向），且 remap 只是布局重写——同一 prompt 的 logits 探针必须完全一致。
    multimodal = verify_quant_dir(output_dir, model_cls="Qwen3_5ForConditionalGeneration")
    probe_delta = max(abs(a - b) for a, b in zip(text_view["logits_probe"], multimodal["logits_probe"]))
    if not multimodal["logits_finite"] or probe_delta > 1e-3:
        raise SystemExit(
            f"key remap broke the checkpoint (logits_finite={multimodal['logits_finite']}, "
            f"max probe delta={probe_delta:.6f})")
    print(f"GPTQ checkpoint saved to {output_dir} (remap verified, probe delta {probe_delta:.2e})")
    return {"text_view_verify": text_view, "multimodal_verify": multimodal, "remap_probe_delta": probe_delta}


def verify_quant_dir(model_dir: Path, model_cls: str = "AutoModelForCausalLM") -> dict:
    """校验一个量化 checkpoint 能被 transformers 真实加载，并给出权重统计。

    model_cls 决定用哪个 Auto 类加载：文本视图键布局（model.layers.*）必须用
    AutoModelForCausalLM；remap 后的 base 布局（model.language_model.model.*）
    必须用多模态类 Qwen3_5ForConditionalGeneration（transformers 5.15 原生）。
    用错类会导致权重键全部不匹配而被静默跳过——logits_finite 仍可能为 True，
    所以调用方必须按布局选类，并配合 logits_probe 对比 remap 前后是否一致。
    """
    import torch
    import transformers
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    cls = getattr(transformers, model_cls)
    model = cls.from_pretrained(str(model_dir), torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()
    total = 0
    for parameter in model.parameters():
        total += parameter.numel() * parameter.element_size()
    ids = tokenizer("请只输出答案字母。", return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**ids).logits.float()
    # 末位 token 的 vocab 前 8 个值作为探针，供 remap 前后对比（布局重写不应改变任何数值）。
    probe = [round(float(value), 6) for value in logits[0, -1, :8]]
    finite = bool(torch.isfinite(logits).all().item())
    return {
        "model_dir": str(model_dir),
        "model_type": getattr(config, "model_type", None),
        "weight_bytes": total,
        "logits_finite": finite,
        "logits_probe": probe,
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
    parser.add_argument("--shell-dir", type=Path, default=None,
                        help="多模态外壳目录（config.json 含 vision_config 的多模态版本）；"
                             "缺省自动选择：源 config 为 qwen3_5 则用源目录，否则用 base HF 目录")
    parser.add_argument("--vllm-load-check", action="store_true",
                        help="导出后离线构造 vLLM LLM 引擎并前向一步,验证可被 vLLM 真实加载"
                             "(本机 pin 的 vllm 0.19.0 无 --dry-run 参数,见 _vllm_load_check.py)")
    parser.add_argument("--vllm-python", default=None,
                        help="vLLM 环境的 python 解释器路径。本脚本本身在 omni 环境跑"
                             "(transformers 5.15 才能加载 Qwen3_5),缺省 sys.executable "
                             "拿不到 vllm,必须显式传 vllm 环境解释器。")
    args = parser.parse_args()

    result: dict = {}
    if args.verify_only is not None:
        result = verify_quant_dir(args.verify_only)
    else:
        if args.format == "gptq":
            result = quantize_gptq(args.model, args.output_dir, args.bits, args.group_size,
                                   shell_dir=args.shell_dir)
        else:
            raise SystemExit(f"unsupported format: {args.format}")
    if args.vllm_load_check:
        check_script = Path(__file__).with_name("_vllm_load_check.py")
        target = args.verify_only or args.output_dir
        command = [
            args.vllm_python or sys.executable, str(check_script),
            "--model", str(target),
            "--quantization", "gptq", "--max-model-len", "4096", "--tensor-parallel-size", "1",
            # 量化 checkpoint 只含语言部分权重，与 serve_vllm.py 一样必须
            # language-model-only（否则 vllm 找不到视觉塔权重直接失败）。
            "--language-model-only",
        ]
        print(" ".join(shlex.quote(item) for item in command))
        try:
            result["vllm_load_check_returncode"] = subprocess.call(command, timeout=600)
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            result["vllm_load_check_error"] = str(exc)
    output = PATHS.manifests / "quant_export.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
