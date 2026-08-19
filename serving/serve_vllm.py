from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


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
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--served-model-name", default=None,
                        help="API 可见的模型名；默认取模型路径 basename，客户端 --model 必须与此一致")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    served_name = args.served_model_name or Path(args.model).name
    command = [
        "vllm", "serve", args.model,
        "--host", args.host, "--port", str(args.port),
        "--max-model-len", str(args.max_model_len),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
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

