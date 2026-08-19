from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from data.utils import iter_jsonl
from eval.cmb_common import exact_match


def load_prediction_records(path: Path) -> list[dict]:
    """兼容仓库内评测输出：eval_cmb_generate / eval_cmb_logprob 写的是
    {"summary": ..., "results": [...]} 单个 JSON；也兼容历史 JSONL 用法。"""
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return list(iter_jsonl(path))
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return payload["results"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported predictions format: {path}")


def group_reward_variance(path: Path, group_size: int) -> dict:
    records = load_prediction_records(path)
    groups = []
    for offset in range(0, len(records), group_size):
        group = records[offset: offset + group_size]
        rewards = []
        for item in group:
            prediction = str(item.get("prediction", ""))
            gold = item.get("gold") if "gold" in item else item.get("answer")
            if gold is None:
                # 无 gold 的行（如 CMB test）不参与方差统计。
                continue
            rewards.append(float(exact_match(prediction, {"answer": gold})))
        if rewards:
            groups.append({"size": len(rewards), "mean": statistics.mean(rewards), "stdev": statistics.pstdev(rewards)})
    return {"groups": len(groups), "nonzero_variance_groups": sum(item["stdev"] > 0 for item in groups), "examples": groups[:10]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether an answer reward has useful group variance")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=8)
    args = parser.parse_args()
    result = group_reward_variance(args.predictions, args.group_size)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
