"""vllm 加载冒烟检查:离线构造 LLM + 真实前向一步。

背景:本机 pin 的 vllm 0.19.0 没有 --dry-run 参数,量化导出后无法用
命令级冒烟检查验证可加载性。本脚本用与 serve 相同的
LLM 引擎配置离线加载并生成一个 token,能同时暴露 config 不合法、
权重损坏、量化内核缺失三类问题,比 --dry-run(只验 config)更严格。

退出码约定:0 = 加载+前向成功;1 = 失败(traceback 打到 stderr)。
调用方需自行设置 CUDA_VISIBLE_DEVICES。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# 本机实测：vllm V1 父进程在 config 校验阶段会初始化 CUDA，fork 启动 EngineCore
# 时报 "Cannot re-initialize CUDA in forked subprocess"，必须用 spawn。
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline vLLM load + one-forward smoke check")
    parser.add_argument("--model", required=True)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--language-model-only", action="store_true",
                        help="与 serve_vllm.py 同款:混合模型只跑文本部分")
    parser.add_argument("--dtype", default=None, help="如 bfloat16;缺省由 config torch_dtype 决定")
    args = parser.parse_args()

    started = time.perf_counter()
    from vllm import LLM, SamplingParams

    llm_kwargs = {
        "model": args.model,
        "quantization": args.quantization,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "language_model_only": args.language_model_only,
        # 只做冒烟检查,跳过 CUDA graph 捕获以加快速度。
        "enforce_eager": True,
    }
    # ModelConfig 不接受 dtype=None(缺省必须省略或传 'auto'),只有显式给出才传。
    if args.dtype:
        llm_kwargs["dtype"] = args.dtype
    llm = LLM(**llm_kwargs)
    outputs = llm.generate(
        ["请只输出答案字母。"],
        SamplingParams(max_tokens=8, temperature=0.0),
    )
    text = outputs[0].outputs[0].text
    elapsed = time.perf_counter() - started
    print(json.dumps({
        "status": "ok",
        "generated_text": text,
        "load_seconds": round(elapsed, 1),
    }, ensure_ascii=False))
    sys.exit(0)


if __name__ == "__main__":
    main()
