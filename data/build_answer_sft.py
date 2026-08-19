from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from config import DEFAULT_SEED, PATHS
from data.utils import ANSWER_LETTERS, answer_letters, iter_jsonl, option_map, write_jsonl


def shuffle_record(record: dict, rng: random.Random) -> tuple[dict[str, str], dict[str, str]]:
    options = option_map(record)
    old_letters = list(options)
    shuffled = old_letters[:]
    rng.shuffle(shuffled)
    new_options = {ANSWER_LETTERS[i]: options[old_letter] for i, old_letter in enumerate(shuffled)}
    mapping = {old_letter: ANSWER_LETTERS[i] for i, old_letter in enumerate(shuffled)}
    return new_options, mapping


def make_example(record: dict, seed: int, shuffle: bool) -> dict | None:
    rng = random.Random(seed)
    original_options = option_map(record)
    if shuffle and len(original_options) >= 2:
        options, mapping = shuffle_record(record, rng)
    else:
        options, mapping = original_options, {letter: letter for letter in original_options}
    original_answer = answer_letters(record.get("answer"))
    answer = sorted(
        {mapping.get(letter) for letter in original_answer if mapping.get(letter) in options},
        key=ANSWER_LETTERS.index,
    )
    if not answer:
        # 答案字母在选项规范化/重排后无对应文本，无法构造可学习样本。
        return None
    question = str(record.get("question", "")).strip()
    option_text = "\n".join(f"{letter}. {text}" for letter, text in options.items())
    prompt = f"{question}\n\n选项：\n{option_text}\n\n请只输出答案字母。"
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "答案：" + ",".join(answer)},
        ],
        "answer": answer,
        "original_answer": original_answer,
        "question_type": record.get("question_type", ""),
        "exam_subject": record.get("exam_subject", ""),
        "source_split": record.get("_split", "train"),
        "source_index": record.get("_source_index"),
        "canonical_key": record.get("_canonical_key"),
        "option_mapping": mapping,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build auditable answer-only SFT JSONL")
    parser.add_argument("--input", type=Path, default=PATHS.processed / "cmb" / "train.jsonl")
    parser.add_argument("--output", type=Path, default=PATHS.processed / "sft" / "train.jsonl")
    parser.add_argument("--max-records", type=int, default=80000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-shuffle", action="store_true")
    args = parser.parse_args()

    records = list(iter_jsonl(args.input))
    rng = random.Random(args.seed)
    rng.shuffle(records)
    records = records[: args.max_records]
    output = []
    dropped = 0
    for i, record in enumerate(records):
        example = make_example(record, args.seed + i, not args.no_shuffle)
        if example is None:
            dropped += 1
            continue
        output.append(example)
    count = write_jsonl(args.output, output)
    stats = {
        "input": str(args.input),
        "output": str(args.output),
        "records": count,
        "dropped_invalid": dropped,
        "seed": args.seed,
        "shuffle": not args.no_shuffle,
    }
    # 每次运行追加一条记录，避免 val 运行覆盖 train 运行的审计信息。
    manifest_path = PATHS.manifests / "sft_build.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        history = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(history, dict) or not isinstance(history.get("runs"), list):
            # 兼容旧版单条扁平格式：作为第一条历史记录保留。
            history = {"runs": [history] if isinstance(history, dict) and history else []}
    else:
        history = {"runs": []}
    history["runs"].append(stats)
    manifest_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

