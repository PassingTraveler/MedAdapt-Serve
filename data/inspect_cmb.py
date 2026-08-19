from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from config import PATHS
from data.utils import option_map, read_json_records


def locate_split(root: Path, split: str) -> Path:
    patterns = {
        "train": "CMB-train-merge.json",
        "val": "CMB-val-merge.json",
        "test": "CMB-test-choice-question-merge.json",
    }
    matches = list(root.rglob(patterns[split]))
    if not matches:
        raise FileNotFoundError(f"Could not locate CMB {split} file below {root}")
    return matches[0]


def inspect(records: list[dict]) -> dict:
    question_types = Counter(str(item.get("question_type", "unknown")) for item in records)
    subjects = Counter(str(item.get("exam_subject", "unknown")) for item in records)
    option_counts = Counter(len(option_map(item)) for item in records)
    lengths = [len(str(item.get("question", ""))) for item in records]
    return {
        "records": len(records),
        "question_type": dict(question_types),
        "top_subjects": subjects.most_common(20),
        "option_count": dict(option_counts),
        "question_chars": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": round(sum(lengths) / len(lengths), 2) if lengths else 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect CMB split sizes and schema")
    parser.add_argument("--root", type=Path, default=PATHS.raw / "cmb")
    parser.add_argument("--output", type=Path, default=PATHS.manifests / "cmb_raw_stats.json")
    args = parser.parse_args()

    result = {"root": str(args.root), "splits": {}}
    for split in ("train", "val", "test"):
        path = locate_split(args.root, split)
        stats = inspect(read_json_records(path))
        stats["path"] = str(path)
        result["splits"][split] = stats
        print(split, json.dumps(stats, ensure_ascii=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

