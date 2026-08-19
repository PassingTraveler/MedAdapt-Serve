from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from config import PATHS
from data.utils import iter_jsonl, option_map
from eval.cmb_common import gold_answer, prompt_from_record
from model.hf_loader import load_model, load_processor


def score_candidate(model, tokenizer, prompt: str, candidate: str, max_length: int) -> float:
    import torch

    prompt_messages = [{"role": "user", "content": prompt}]
    full_messages = prompt_messages + [{"role": "assistant", "content": "答案：" + candidate}]
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
    full = tokenizer(full_text, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=max_length)
    input_ids = full["input_ids"].to(model.device)
    attention_mask = full["attention_mask"].to(model.device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1]
    labels = input_ids[:, 1:]
    start = max(int(prompt_ids.shape[1]) - 1, 0)
    end = labels.shape[1]
    if end <= start:
        # prompt 已占满 max_length，候选部分无 token 可评分。
        return 0.0
    token_log_probs = logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    # 按 token 数归一化，保证 1 个字母的候选与 4 个字母的候选可比。
    return float(token_log_probs[:, start:end].sum().item() / (end - start))


def select_candidates(letters: list[str], question_type: str | None) -> list[str]:
    """候选空间只由题型决定，绝不用 gold 宽度——用 gold 等于提前告诉模型"选几个"。
    单选只比较单个字母；多选比较全部 2..n 元组合，模型选错个数会被真实判错。
    question_type 来自数据集的官方题型字段（单项选择题/多项选择题）。"""
    if question_type and "多" in str(question_type):
        combos: list[str] = []
        for width in range(2, len(letters) + 1):
            combos.extend(",".join(group) for group in itertools.combinations(letters, width))
        return combos
    return letters


def main() -> None:
    parser = argparse.ArgumentParser(description="Score CMB candidate answers with local model logprob")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, default=PATHS.processed / "cmb" / "val.jsonl")
    parser.add_argument("--output", type=Path, default=PATHS.eval / "cmb_logprob.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()

    processor = load_processor(args.model)
    tokenizer = getattr(processor, "tokenizer", processor)
    model = load_model(args.model, device_map="auto")
    model.eval()
    results = []
    correct = 0
    scored = 0
    by_type: dict[str, list[int]] = {}
    for index, record in enumerate(iter_jsonl(args.data)):
        if args.limit is not None and index >= args.limit:
            break
        gold = gold_answer(record)
        letters = list(option_map(record))
        if not gold or not letters:
            results.append({"index": index, "scored": False, "reason": "missing_gold_or_options",
                            "gold": gold, "canonical_key": record.get("_canonical_key")})
            continue
        prompt = prompt_from_record(record)
        candidates = select_candidates(letters, record.get("question_type"))
        scores = {candidate: score_candidate(model, tokenizer, prompt, candidate, args.max_length) for candidate in candidates}
        prediction = max(scores, key=scores.get) if scores else ""
        is_correct = prediction.split(",") == gold
        correct += int(is_correct)
        scored += 1
        by_type.setdefault(str(record.get("question_type") or "unknown"), []).append(int(is_correct))
        results.append({"index": index, "scored": True, "question_type": record.get("question_type"),
                        "scores": scores, "prediction": prediction,
                        "gold": gold, "correct": is_correct, "canonical_key": record.get("_canonical_key")})
        if (index + 1) % 20 == 0:
            print(f"{index + 1} examples acc={correct / max(scored, 1):.4f}")
    summary = {"data": str(args.data), "model": args.model, "records": len(results), "scored": scored,
               "accuracy": correct / scored if scored else 0,
               "accuracy_by_type": {key: sum(values) / len(values) for key, values in by_type.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
