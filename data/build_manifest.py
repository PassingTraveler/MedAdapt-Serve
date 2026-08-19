from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from config import PATHS
from data.utils import iter_jsonl


def summarize(path: Path) -> dict:
    records = list(iter_jsonl(path))
    lengths = []
    subjects = Counter()
    for record in records:
        messages = record.get("messages")
        if isinstance(messages, list):
            text = "\n".join(str(item.get("content", "")) for item in messages if isinstance(item, dict))
        else:
            # 非 chat 格式（如 processed cmb 原始记录）：用题干 + 选项文本统计长度。
            parts = [str(record.get("question") or "")]
            parts.extend(str(value) for value in (record.get("option") or {}).values())
            text = "\n".join(part for part in parts if part)
        lengths.append(len(text))
        if record.get("exam_subject"):
            subjects[str(record["exam_subject"])] += 1
    return {
        "path": str(path),
        "records": len(records),
        "text_chars": {"min": min(lengths) if lengths else 0, "max": max(lengths) if lengths else 0, "mean": round(sum(lengths) / len(lengths), 2) if lengths else 0},
        "top_subjects": subjects.most_common(20),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a compact data manifest")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=PATHS.manifests / "dataset_summary.json")
    args = parser.parse_args()
    result = {"datasets": [summarize(path) for path in args.paths]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

