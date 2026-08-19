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
    # 直接复用 generate 评测的 run()，不再改写 sys.argv。
    generate_run(parser.parse_args())


if __name__ == "__main__":
    main()

