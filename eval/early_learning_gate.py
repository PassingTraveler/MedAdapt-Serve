from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Record an early-learning gate from a logprob evaluation")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-accuracy-delta", type=float, default=0.0)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))["summary"]
    checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))["summary"]
    delta = float(checkpoint.get("accuracy", 0)) - float(baseline.get("accuracy", 0))
    result = {
        "baseline": baseline,
        "checkpoint": checkpoint,
        "accuracy_delta": delta,
        "observable_direction": delta >= args.min_accuracy_delta,
        "interpretation": "continue_and_check_margin" if delta >= args.min_accuracy_delta else "inspect_train_loss_target_modules_and_labels",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["observable_direction"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

