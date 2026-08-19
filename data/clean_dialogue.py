from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import PATHS
from data.utils import normalize_text, short_hash, write_jsonl


def normalize_messages(record: dict) -> list[dict[str, str]]:
    messages = record.get("messages")
    if isinstance(messages, list):
        result = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).lower()
            raw_content = message.get("content")
            content = str(raw_content).strip() if raw_content is not None else ""
            if role in {"system", "user", "assistant"} and content:
                result.append({"role": role, "content": content})
        return result
    instruction = record.get("instruction") or record.get("question") or record.get("input")
    answer = record.get("output") or record.get("answer") or record.get("response")
    if instruction and answer:
        return [{"role": "user", "content": str(instruction)}, {"role": "assistant", "content": str(answer)}]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean optional medical dialogue JSONL")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PATHS.processed / "dialogue.jsonl")
    parser.add_argument("--max-records", type=int, default=20000)
    args = parser.parse_args()

    seen: set[str] = set()
    output = []
    for raw_line in args.input.read_text(encoding="utf-8-sig").splitlines():
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        messages = normalize_messages(record)
        if len(messages) < 2 or messages[-1]["role"] != "assistant":
            continue
        key = short_hash("\n".join(normalize_text(item["content"]) for item in messages))
        if key in seen:
            continue
        seen.add(key)
        output.append({"messages": messages, "source_hash": key})
        if len(output) >= args.max_records:
            break
    count = write_jsonl(args.output, output)
    print(json.dumps({"input": str(args.input), "output": str(args.output), "records": count}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

