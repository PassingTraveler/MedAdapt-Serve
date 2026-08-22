from __future__ import annotations

import argparse
from pathlib import Path

from eval.eval_cmb_generate import run as generate_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-domain entry point; use the same API harness with a fixed benchmark release")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--structured", action="store_true",
                        help="structured outputs:强制输出 {'answer': 'X'} JSON")
    parser.add_argument("--workers", type=int, default=1)
    # 直接复用 generate 评测的 run()，不再改写 sys.argv。
    generate_run(parser.parse_args())


if __name__ == "__main__":
    main()

