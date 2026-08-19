from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator


# CMB 原始数据中存在 6 选项题目（A-F，train 1,150 条），只支持 A-E 会静默丢失
# 答案为 F 的记录（269 条）并截掉真实的干扰项 F。
ANSWER_LETTERS = tuple("ABCDEF")
MAX_OPTION_LETTERS = len(ANSWER_LETTERS)


def read_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        return []
    parsed = json.loads(text)
    if isinstance(parsed, dict):
        for key in ("data", "items", "questions", "records"):
            if isinstance(parsed.get(key), list):
                parsed = parsed[key]
                break
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return [item for item in parsed if isinstance(item, dict)]


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            yield item


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def normalize_text(value: Any) -> str:
    text = str(value or "").replace("\u3000", " ")
    return re.sub(r"\s+", " ", text).strip().lower()


def option_map(record: dict[str, Any]) -> dict[str, str]:
    options = record.get("option") or record.get("options") or {}
    if isinstance(options, list):
        options = {ANSWER_LETTERS[i]: value for i, value in enumerate(options[:MAX_OPTION_LETTERS])}
    if not isinstance(options, dict):
        return {}
    result: dict[str, str] = {}
    for key, value in options.items():
        letter = str(key).strip().upper().strip("()[]")
        text = str(value or "").strip()
        if letter in ANSWER_LETTERS and text:
            result[letter] = text
    return dict(sorted(result.items(), key=lambda item: ANSWER_LETTERS.index(item[0])))


def answer_letters(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        raw = "".join(str(item) for item in value)
    else:
        raw = str(value or "")
    letters = [letter for letter in re.findall(r"[A-Fa-f]", raw.upper())]
    return sorted(set(letters), key=ANSWER_LETTERS.index)


def canonical_question_key(record: dict[str, Any]) -> str:
    question = normalize_text(record.get("question") or record.get("prompt"))
    options = option_map(record)
    payload = question + "\n" + "\n".join(
        f"{key}:{normalize_text(value)}" for key, value in options.items()
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

