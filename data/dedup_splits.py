from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import PATHS
from data.inspect_cmb import locate_split
from data.utils import answer_letters, canonical_question_key, option_map, read_json_records, write_jsonl


def normalize_record(record: dict, split: str, index: int) -> dict:
    result = dict(record)
    result["_split"] = split
    result["_source_index"] = index
    # 规范化选项：丢弃空白选项（raw 中存在 18,694 条带空选项的记录）、
    # 只保留 A-F 字母（raw 中存在 6 选项题目，见 README 2.1）。
    options = option_map(record)
    if options:
        result["option"] = options
    if "answer" in record:
        result["_original_answer"] = record.get("answer")
        letters = answer_letters(record.get("answer"))
        kept = [letter for letter in letters if letter in options]
        if len(kept) != len(letters):
            result["_answer_trimmed"] = True
        result["answer"] = "".join(kept)
    result["_canonical_key"] = canonical_question_key(result)
    return result


def dedup(records: list[dict]) -> tuple[list[dict], dict]:
    seen: dict[str, tuple[str, int]] = {}
    kept: list[dict] = []
    dropped = []
    for record in records:
        key = record["_canonical_key"]
        if key in seen:
            dropped.append({"key": key, "duplicate_of": seen[key]})
            continue
        seen[key] = (record["_split"], record["_source_index"])
        kept.append(record)
    return kept, {"input": len(records), "kept": len(kept), "dropped": len(dropped), "examples": dropped[:100]}


def filter_train_cross_split(train_records: list[dict], protected: dict[str, tuple[str, int]]) -> tuple[list[dict], list[dict]]:
    """训练集若与受保护的评测集（val/test）撞题，删训练侧副本而不是动评测集。"""
    cross_duplicates = []
    output = []
    for record in train_records:
        key = record["_canonical_key"]
        if key in protected:
            record["_cross_split_duplicate"] = True
            cross_duplicates.append({"key": key, "current": ("train", record["_source_index"]),
                                     "protected": protected[key]})
        else:
            output.append(record)
    return output, cross_duplicates


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate CMB splits and audit cross-split overlap")
    parser.add_argument("--root", type=Path, default=PATHS.raw / "cmb")
    parser.add_argument("--output-dir", type=Path, default=PATHS.processed / "cmb")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_records: dict[str, list[dict]] = {}
    normalization_drops: dict[str, list[dict]] = {}
    for split in ("train", "val", "test"):
        path = locate_split(args.root, split)
        raw = read_json_records(path)
        records: list[dict] = []
        for i, item in enumerate(raw):
            record = normalize_record(item, split, i)
            if split != "test" and not record.get("answer"):
                # 答案字母缺失、超出 A-F（raw 中存在 G+ 答案）或全部指向已丢弃的
                # 空选项：这类样本无法在 answer-only 框架下学习/评测，丢弃并留痕。
                record["_drop_reason"] = "answer_letters_missing_or_unsupported"
                normalization_drops.setdefault(split, []).append(record)
                continue
            records.append(record)
        split_records[split] = records

    unique_by_split: dict[str, list[dict]] = {}
    summary: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        if split == "train":
            unique, stats = dedup(split_records[split])
        else:
            # Keep official evaluation rows unchanged for benchmark comparability;
            # report within-split duplicates instead of deleting them.
            unique = split_records[split]
            seen: set[str] = set()
            duplicate_count = 0
            for record in unique:
                key = record["_canonical_key"]
                if key in seen:
                    record["_within_split_duplicate"] = True
                    duplicate_count += 1
                else:
                    seen.add(key)
            stats = {
                "input": len(unique),
                "kept": len(unique),
                "dropped": 0,
                "within_split_duplicates": duplicate_count,
                "examples": [],
            }
        trimmed = sum(1 for record in unique if record.get("_answer_trimmed"))
        drops = normalization_drops.get(split, [])
        stats["normalization_dropped"] = len(drops)
        stats["normalization_drop_examples"] = [
            {"key": record["_canonical_key"], "split": split, "source_index": record["_source_index"],
             "original_answer": record.get("_original_answer"), "question": str(record.get("question", ""))[:80]}
            for record in drops[:20]
        ]
        stats["answer_trimmed"] = trimmed
        unique_by_split[split] = unique
        summary[split] = stats

    # Evaluation splits are protected. If a question exists in val/test, remove the
    # training-side copy instead of shrinking the official evaluation split.
    protected: dict[str, tuple[str, int]] = {}
    for split in ("test", "val"):
        for record in unique_by_split[split]:
            protected.setdefault(record["_canonical_key"], (split, record["_source_index"]))

    # 真正的 val ↔ test 跨 split 重叠（此前实现把 split 内部重复误标为跨 split 重复）。
    val_seen: dict[str, tuple[str, int]] = {}
    for record in unique_by_split["val"]:
        val_seen.setdefault(record["_canonical_key"], ("val", record["_source_index"]))
    test_seen: dict[str, tuple[str, int]] = {}
    for record in unique_by_split["test"]:
        test_seen.setdefault(record["_canonical_key"], ("test", record["_source_index"]))
    val_test_overlap = [
        {"key": key, "val": val_seen[key], "test": test_seen[key]}
        for key in sorted(set(val_seen) & set(test_seen))
    ]

    train_output, train_cross_duplicates = filter_train_cross_split(unique_by_split["train"], protected)

    summary["train"]["cross_split_overlap"] = len(train_cross_duplicates)
    summary["train"]["cross_split_examples"] = train_cross_duplicates[:100]
    summary["val"]["cross_split_overlap"] = len(val_test_overlap)
    summary["val"]["cross_split_examples"] = val_test_overlap[:100]
    summary["test"]["cross_split_overlap"] = len(val_test_overlap)
    summary["test"]["cross_split_examples"] = val_test_overlap[:100]

    summary["train"]["output_kept"] = len(train_output)
    summary["val"]["output_kept"] = len(unique_by_split["val"])
    summary["test"]["output_kept"] = len(unique_by_split["test"])

    write_jsonl(args.output_dir / "train.jsonl", train_output)
    write_jsonl(args.output_dir / "val.jsonl", unique_by_split["val"])
    write_jsonl(args.output_dir / "test.jsonl", unique_by_split["test"])

    manifest = {"source_root": str(args.root), "output_dir": str(args.output_dir), "splits": summary}
    manifest_path = PATHS.manifests / "cmb_dedup.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
