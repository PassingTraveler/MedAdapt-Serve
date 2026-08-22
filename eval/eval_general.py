from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import requests


def extract_number(text: str) -> float | None:
    """提取最终数值：优先 GSM8K 的 '#### N'，否则取最后一个出现的数字。"""
    match = re.search(r"####\s*(-?\d+(?:[.,]\d+)?)", str(text or ""))
    if match:
        return float(match.group(1).replace(",", "").replace("，", ""))
    numbers = re.findall(r"-?\d+(?:[.,]\d+)?", str(text or ""))
    if not numbers:
        return None
    return float(numbers[-1].replace(",", "").replace("，", ""))


def normalize_gold(value) -> str | None:
    """把 gold 统一成可比较的字符串；缺失返回 None。"""
    if isinstance(value, (list, tuple)):
        parts = [normalize_gold(item) for item in value]
        if any(part is None for part in parts):
            return None
        return ",".join(part for part in parts if part is not None)
    if value is None or str(value).strip() == "":
        return None
    return str(value).strip()


def is_correct_answer(prediction: str, gold: str) -> bool:
    # 两侧都能提取出数值时按数值精确比较，避免 "4" in "42" 之类子串误报；
    # 否则退化为子串匹配（适用于短语型答案）。
    pred_number = extract_number(prediction)
    gold_number = extract_number(gold)
    if pred_number is not None and gold_number is not None:
        return abs(pred_number - gold_number) < 1e-9
    return gold in prediction


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a generic JSONL prompt/answer set through an API")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="生成上限;GSM8K 0-shot 推理在 256 下约 29% 截断"
                             "(实测 380/1319 条顶满上限),取中间推理数字造成假阴性")
    args = parser.parse_args()

    rows = []
    correct = 0
    skipped = 0
    index = 0
    for line in args.data.read_text(encoding="utf-8-sig").splitlines():
        if args.limit is not None and index >= args.limit:
            break
        if not line.strip():
            continue
        item = json.loads(line)
        prompt = item.get("prompt") or item.get("question") or item.get("messages")
        gold = normalize_gold(item.get("answer") if "answer" in item else item.get("target"))
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}]
        response = requests.post(args.base_url.rstrip("/") + "/chat/completions", json={"model": args.model, "messages": messages, "temperature": 0, "max_tokens": args.max_tokens}, timeout=300)
        response.raise_for_status()
        prediction = response.json()["choices"][0]["message"]["content"].strip()
        if gold is None:
            # 记录缺少答案：不参与准确率统计，避免空 gold 恒判对。
            skipped += 1
            rows.append({"index": index, "prediction": prediction, "gold": None, "correct": False, "skipped": True})
            index += 1
            continue
        is_correct = is_correct_answer(prediction, gold)
        correct += int(is_correct)
        rows.append({"index": index, "prediction": prediction, "gold": gold, "correct": is_correct})
        index += 1
    summary = {"records": len(rows), "scored": len(rows) - skipped, "skipped_missing_gold": skipped,
               "accuracy": correct / (len(rows) - skipped) if len(rows) - skipped else 0,
               "model": args.model, "data": str(args.data), "max_tokens": args.max_tokens}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

