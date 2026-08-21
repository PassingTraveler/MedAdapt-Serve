from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

# 本机实测：vllm V1 父进程在 config 校验阶段会初始化 CUDA，fork 启动 EngineCore
# 时报 "Cannot re-initialize CUDA in forked subprocess"，必须用 spawn。
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def resolve_vllm_cli() -> str:
    """定位 vllm 命令行入口：PATH 里没有时回退到当前解释器同目录的 vllm
    （用绝对 python 启动本脚本时不依赖 shell 环境）。"""
    found = shutil.which("vllm")
    if found:
        return found
    sibling = Path(sys.executable).with_name("vllm")
    if sibling.exists():
        return str(sibling)
    raise SystemExit("未找到 vllm 命令：请在 vllm 环境内运行或把 vllm 加入 PATH")


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the pinned vLLM text-only service")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1",
                        help="默认只监听本机；需要被其他机器访问时显式传 0.0.0.0（注意无鉴权，"
                             "仅限可信内网）")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    # 压测并发上限 16，默认 64 已留 4 倍余量；同时 CUDA graph 缓冲随 max_num_seqs
    # 缩小（默认值下 9B 模型在 24GB 卡上 profiling 阶段会 OOM，实测）。
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--served-model-name", default=None,
                        help="API 可见的模型名；默认取模型路径 basename，客户端 --model 必须与此一致")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    served_name = args.served_model_name or Path(args.model).name
    command = [
        resolve_vllm_cli(), "serve", args.model,
        "--host", args.host, "--port", str(args.port),
        "--max-model-len", str(args.max_model_len),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-num-seqs", str(args.max_num_seqs),
        "--language-model-only",
        "--served-model-name", served_name,
    ]
    if args.quantization:
        command.extend(["--quantization", args.quantization])
    if args.enable_prefix_caching:
        command.append("--enable-prefix-caching")
    print(" ".join(shlex.quote(item) for item in command))
    if not args.dry_run:
        raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()

