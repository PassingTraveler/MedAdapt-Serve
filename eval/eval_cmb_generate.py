from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from config import PATHS
from data.utils import answer_letters, iter_jsonl
from eval.cmb_common import exact_match, gold_answer, prompt_from_record

# structured outputs 的 JSON schema：强制模型输出 {"answer": "X"}。
# 实测(M6 CMExam):base 与 LoRA 均无视"请只输出答案字母"指令,自由生成
# 复述/思考链,32 token 内截断在题干阶段 → 准确率低于随机水平(11.9%,
# 5 选 1 随机 ~20%)。structured 约束下两模型均输出 {"answer":"C"} 且答对。
# 对多选(CMB 部分题 gold 为 "BCDE" 型)同样成立:answer 可含多个字母。
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
STRUCTURED_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer_letter",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "pattern": "^[" + LETTERS + "]{1,8}$"},
            },
            "required": ["answer"],
        },
    },
}


def request_one(base_url: str, model: str, prompt: str, max_tokens: int,
                structured: bool = False) -> str:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_tokens}
    if structured:
        body["response_format"] = STRUCTURED_SCHEMA
    response = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        headers={"Content-Type": "application/json"},
        json=body,
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def run(args) -> None:
    records = list(iter_jsonl(args.data))
    if args.limit is not None:
        records = records[: args.limit]
    start = time.perf_counter()

    def process(index: int, record: dict) -> dict:
        prediction = request_one(args.base_url, args.model, prompt_from_record(record),
                                 args.max_tokens, structured=args.structured)
        gold = gold_answer(record)
        if not gold:
            # CMB 官方 test 不公开答案：只收集预测，不参与准确率统计。
            return {"index": index, "scored": False, "prediction": prediction,
                    "canonical_key": record.get("_canonical_key")}
        is_correct = exact_match(prediction, record)
        if args.structured:
            # structured 输出为 {"answer": "X"} JSON,parse_prediction 的
            # 关键词正则处理不了引号/换行,直接取 JSON 的 answer 字段。
            parsed = None
            try:
                parsed = json.loads(prediction).get("answer")
            except (json.JSONDecodeError, AttributeError):
                parsed = None
            is_correct = answer_letters(parsed) == gold
        return {"index": index, "scored": True, "prediction": prediction, "correct": is_correct,
                "gold": gold, "canonical_key": record.get("_canonical_key")}

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(lambda pair: process(*pair), enumerate(records)))
    else:
        results = [process(index, record) for index, record in enumerate(records)]
        for index, row in enumerate(results):
            scored_rows = [r for r in results[: index + 1] if r["scored"]]
            if (index + 1) % 100 == 0 and scored_rows:
                acc = sum(r["correct"] for r in scored_rows) / len(scored_rows)
                print(f"{index + 1} examples acc={acc:.4f}")

    scored = [r for r in results if r["scored"]]
    correct = sum(r["correct"] for r in scored)
    summary = {"data": str(args.data), "model": args.model, "records": len(results), "scored": len(scored),
               "accuracy": correct / len(scored) if scored else 0, "seconds": time.perf_counter() - start,
               "structured_outputs": args.structured, "max_tokens": args.max_tokens}
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
    parser.add_argument("--max-tokens", type=int, default=32,
                        help="生成上限;32 下 base 模型自由生成时截断在题干阶段"
                             "(M6 CMExam 实测 11.9% 低于随机),需配合 --structured 使用")
    parser.add_argument("--structured", action="store_true",
                        help="structured outputs:强制输出 {'answer': 'X'} JSON"
                             "(vLLM 0.19.0 response_format=json_schema,实测生效)")
    parser.add_argument("--workers", type=int, default=1, help="并发请求线程数(压测并发上限 16)")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

