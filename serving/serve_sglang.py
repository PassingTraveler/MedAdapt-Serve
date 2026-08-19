from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the optional SGLang text-only service")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1",
                        help="默认只监听本机；需要被其他机器访问时显式传 0.0.0.0（注意无鉴权，"
                             "仅限可信内网）")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--served-model-name", default=None,
                        help="API 可见的模型名；默认取模型路径 basename，客户端 --model 必须与此一致")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    served_name = args.served_model_name or Path(args.model).name
    command = [
        "python", "-m", "sglang.launch_server", "--model-path", args.model,
        "--host", args.host, "--port", str(args.port), "--tp-size", str(args.tp_size),
        "--context-length", str(args.context_length),
        "--served-model-name", served_name,
    ]
    print(" ".join(shlex.quote(item) for item in command))
    if not args.dry_run:
        raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
