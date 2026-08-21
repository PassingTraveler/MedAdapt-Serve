"""配对显著性分析:两个评测结果 JSON(同一数据顺序)之间的 acc 差异。

输出:总/单选/多选各自的 acc、Δ、bootstrap 95% CI(配对重采样,默认 10,000 次)、
配对 McNemar 精确 p 值(base 错→LoRA 对 与 base 对→LoRA 错 的符号检验)。
用法:python -m eval.analyze_significance --base <json> --lora <json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from config import DEFAULT_SEED


def load(path: Path) -> list[tuple[int, int, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [(r["index"], int(r["correct"]), str(r.get("question_type") or "unknown"))
            for r in data["results"] if r.get("scored")]


def analyze(base: np.ndarray, lora: np.ndarray, seed: int, label: str) -> None:
    n = len(base)
    acc_b, acc_l, delta = base.mean(), lora.mean(), lora.mean() - base.mean()
    b_to_l = int(((base == 0) & (lora == 1)).sum())  # base 错 → lora 对
    l_to_b = int(((base == 1) & (lora == 0)).sum())  # base 对 → lora 错
    p = binomtest(min(b_to_l, l_to_b), b_to_l + l_to_b, 0.5).pvalue if (b_to_l + l_to_b) else 1.0
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(10000):
        # 配对 bootstrap:必须对同一条记录联合重采样,独立抽样会破坏配对、人为拉宽 CI。
        s = rng.integers(0, n, n)
        diffs.append(lora[s].mean() - base[s].mean())
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"{label}: n={n} base={acc_b * 100:.2f}% lora={acc_l * 100:.2f}% Δ={delta * 100:+.2f}pt "
          f"bootstrap95%CI[{lo * 100:+.2f},{hi * 100:+.2f}] McNemar_p={p:.4g} "
          f"(base错→lora对 {b_to_l} / base对→lora错 {l_to_b})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired significance analysis for two eval JSONs")
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--lora", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    base_records = load(args.base)
    lora_records = load(args.lora)
    if [b[0] for b in base_records] != [lr[0] for lr in lora_records]:
        raise SystemExit("两个结果文件的记录顺序不一致,无法配对分析")
    base = np.array([b[1] for b in base_records], dtype=int)
    lora = np.array([lr[1] for lr in lora_records], dtype=int)
    types = np.array([b[2] for b in base_records])

    analyze(base, lora, args.seed, "全部")
    for question_type in sorted(set(types)):
        mask = types == question_type
        if mask.sum() > 0:
            analyze(base[mask], lora[mask], args.seed, question_type)


if __name__ == "__main__":
    main()
