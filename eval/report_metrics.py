from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def bootstrap(values: list[float], seed: int, rounds: int = 2000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = []
    for _ in range(rounds):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(rounds * 0.025)], means[int(rounds * 0.975)]


def attach_fields(rows: list[dict], data_path: Path) -> list[dict]:
    """按 index 顺序把题干元信息（题型/大类/学科）并入结果行，用于分组统计。"""
    records: list[dict] = []
    with data_path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    enriched = []
    for row in rows:
        row = dict(row)
        index = row.get("index")
        if isinstance(index, int) and 0 <= index < len(records):
            record = records[index]
            for field in ("question_type", "exam_class", "exam_subject"):
                if field in record:
                    row[field] = record[field]
        enriched.append(row)
    return enriched


def group_stats(rows: list[dict], field: str, seed: int) -> dict:
    groups: dict[str, list[float]] = {}
    for row in rows:
        if not row.get("scored", True):
            continue
        groups.setdefault(str(row.get(field) or "unknown"), []).append(float(row.get("correct", False)))
    result = {}
    for name, values in groups.items():
        low, high = bootstrap(values, seed)
        result[name] = {"n": len(values), "accuracy": sum(values) / len(values), "ci95": [low, high]}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize result JSON and bootstrap accuracy interval")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=None, help="Optional source jsonl to attach question_type/exam_class for group stats")
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = payload.get("results", [])
    if args.data is not None and args.data.exists():
        rows = attach_fields(rows, args.data)
    values = [float(row.get("correct", False)) for row in rows if row.get("scored", True)]
    low, high = bootstrap(values, args.seed)
    summary = dict(payload.get("summary", {}))
    summary.update({"bootstrap_95_low": low, "bootstrap_95_high": high})
    for field in ("question_type", "exam_class"):
        if any(field in row for row in rows):
            summary[f"by_{field}"] = group_stats(rows, field, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
