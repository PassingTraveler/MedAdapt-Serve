from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

from config import PATHS
from data.utils import iter_jsonl
from eval.cmb_common import exact_match, gold_answer, prompt_from_record


def request_one(base_url: str, model: str, prompt: str, max_tokens: int) -> str:
    response = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        headers={"Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": max_tokens},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def run(args) -> None:
    results = []
    correct = 0
    scored = 0
    start = time.perf_counter()
    for index, record in enumerate(iter_jsonl(args.data)):
        if args.limit is not None and index >= args.limit:
            break
        prediction = request_one(args.base_url, args.model, prompt_from_record(record), 32)
        gold = gold_answer(record)
        if not gold:
            # CMB 官方 test 不公开答案：只收集预测，不参与准确率统计。
            results.append({"index": index, "scored": False, "prediction": prediction,
                            "canonical_key": record.get("_canonical_key")})
            continue
        is_correct = exact_match(prediction, record)
        correct += int(is_correct)
        scored += 1
        results.append({"index": index, "scored": True, "prediction": prediction, "correct": is_correct,
                        "gold": gold, "canonical_key": record.get("_canonical_key")})
        if (index + 1) % 100 == 0:
            print(f"{index + 1} examples acc={correct / max(scored, 1):.4f}")
    summary = {"data": str(args.data), "model": args.model, "records": len(results), "scored": scored,
               "accuracy": correct / scored if scored else 0, "seconds": time.perf_counter() - start}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CMB through an OpenAI-compatible server")
    parser.add_argument("--data", type=Path, default=PATHS.processed / "cmb" / "test.jsonl")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, default=PATHS.eval / "cmb_generate.json")
    parser.add_argument("--limit", type=int, default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

