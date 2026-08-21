"""从训练未见数据(train 中未进 SFT 的样本)固定种子抽样,构造 held-out 评测集。

官方 test 集不含答案(CMB held-out 设计),无法做准确率评测;train 去重后 239,690 条中
SFT 只用了 80,000 条(seed 20260818 抽样),其余 159,690 条训练完全未见且带答案,
是干净的 held-out 集。本脚本按 canonical_key 排除训练集后固定种子抽样。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from config import DEFAULT_SEED, PATHS
from data.utils import iter_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build held-out eval set from unseen train records")
    parser.add_argument("--input", type=Path, default=PATHS.processed / "cmb" / "train.jsonl")
    parser.add_argument("--sft", type=Path, default=PATHS.processed / "sft" / "train.jsonl")
    parser.add_argument("--output", type=Path, default=PATHS.processed / "cmb" / "heldout_5000.jsonl")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    trained = set()
    for line in Path(args.sft).read_text(encoding="utf-8").splitlines():
        try:
            trained.add(json.loads(line)["canonical_key"])
        except (json.JSONDecodeError, KeyError):
            continue
    heldout = [r for r in iter_jsonl(args.input) if r.get("_canonical_key") not in trained]
    rng = random.Random(args.seed)
    rng.shuffle(heldout)
    sample = heldout[: args.n]
    count = write_jsonl(args.output, sample)
    n_multi = sum(1 for r in sample if "多" in str(r.get("question_type", "")))
    stats = {
        "input": str(args.input),
        "sft_excluded": str(args.sft),
        "trained_keys": len(trained),
        "heldout_total": len(heldout),
        "records": count,
        "multi_choice": n_multi,
        "seed": args.seed,
    }
    manifest_path = PATHS.manifests / "heldout_build.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        history = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(history, dict) or not isinstance(history.get("runs"), list):
            history = {"runs": [history] if isinstance(history, dict) and history else []}
    else:
        history = {"runs": []}
    history["runs"].append(stats)
    manifest_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
