from __future__ import annotations

import re
from typing import Any

from data.utils import answer_letters, option_map


def prompt_from_record(record: dict[str, Any]) -> str:
    question = str(record.get("question", "")).strip()
    options = option_map(record)
    option_text = "\n".join(f"{letter}. {text}" for letter, text in options.items())
    return f"{question}\n\n选项：\n{option_text}\n\n请只输出答案字母。"


# 单个答案字母串：分隔符可省略，因此 "答案：BCDE" 能整体捕获为多选，
# "答案：B。解析…" 会在句号处自然停止。
LETTER_RUN = r"[A-Fa-f](?:[\s,、/&和]*[A-Fa-f])*"


def parse_prediction(text: str) -> list[str]:
    text = str(text or "").strip()
    # 首选明确的关键词定位。
    match = re.search(r"(?:答案|answer)\s*[:：是为]?\s*(" + LETTER_RUN + r")", text, re.I)
    if match:
        return answer_letters(match.group(1))
    # 次选：正确答案/选择/应选等措辞；"选(?!项)" 防止命中 "选项" 一词。
    match = re.search(r"(?:正确答案|应选|选择|选(?!项))\s*[:：是为]?\s*(" + LETTER_RUN + r")", text, re.I)
    if match:
        return answer_letters(match.group(1))
    # 兜底：短输出直接取全部字母（如模型只回 "B" 或 "B、C"）；
    # 长文本取最后一段字母串（避免吞入题干回显的选项字母）。
    runs = re.findall(LETTER_RUN, text)
    if not runs:
        return []
    if len(text) <= 20:
        return answer_letters("".join(runs))
    return answer_letters(runs[-1])


def gold_answer(record: dict[str, Any]) -> list[str]:
    return answer_letters(record.get("answer"))


def exact_match(prediction: str, record: dict[str, Any]) -> bool:
    return parse_prediction(prediction) == gold_answer(record)

