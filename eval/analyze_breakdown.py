"""heldout 评测按考试类型/科目维度的 acc breakdown(README 3.1 报告项)。

输入:base/lora 两个 heldout logprob JSON + heldout 数据(经 canonical_key join
拿到 exam_type/exam_class 分类)。输出 markdown 表格:每类样本数、base/lora
acc、Δ,以及 McNemar p(样本 >= 30 才算,小类只报数字)。
用法:eval/analyze_breakdown.py --output out/eval/heldout_breakdown.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import PATHS


def load_records(path: Path) -> dict[str, dict]:
    """index -> {correct, question_type}"""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(r["index"]): r for r in data["results"] if r.get("scored", True)}


def meta_by_key(path: Path) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        meta[record["_canonical_key"]] = {
            "exam_type": record.get("exam_type", "-"),
            "exam_class": record.get("exam_class", "-"),
            "question_type": record.get("question_type", "-"),
        }
    return meta


def mcnemar_p(tb: int, fb: int) -> float | None:
    """配对 McNemar(双侧):(base 错→lora 对)= tb,(base 对→lora 错)= fb。
    小样本用精确二项;大样本用连续性校正卡方。
    """
    n = tb + fb
    if n == 0:
        return None
    if n < 30:
        from math import comb
        p = sum(comb(n, k) for k in range(min(tb, fb) + 1)) / (2 ** n) * 2
        return min(p, 1.0)
    z = abs(tb - fb) - 1
    from math import sqrt
    return 2 * (1 - __import__("scipy").stats.norm.cdf(z / sqrt(tb + fb)))


def fmt_acc(value: float, n: int) -> str:
    return f"{value * 100:.2f}%({n})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=PATHS.eval / "base_heldout5000_logprob.json")
    parser.add_argument("--lora", type=Path, default=PATHS.eval / "lora_heldout5000_logprob_ckpt1250.json")
    parser.add_argument("--data", type=Path, default=PATHS.processed / "cmb" / "heldout_5000.jsonl")
    parser.add_argument("--output", type=Path, default=PATHS.eval / "heldout_breakdown.md")
    args = parser.parse_args()

    base = load_records(args.base)
    lora = load_records(args.lora)
    meta = meta_by_key(args.data)

    # 先验证两个评测的 index 一一对应(断点续跑/失败重试可能产生顺序差异)。
    common = sorted(set(base) & set(lora))
    if len(common) != len(base) or len(common) != len(lora):
        print(f"WARN: base {len(base)} lora {len(lora)} common {len(common)}, 结果可能不配对")
    # 分类经 canonical_key 关联(base 任一记录的 key 即数据里的 _canonical_key)。
    sample_key = base[common[0]]["canonical_key"]
    assert sample_key in meta, f"canonical_key {sample_key!r} 不在数据 meta 中"

    lines = ["# heldout 5,000 条 breakdown(base vs LoRA)", ""]
    for dim in ("exam_type", "exam_class"):
        lines.append(f"## {dim}")
        lines.append("| 类别 | 样本数 | base acc | LoRA acc | Δ | McNemar p |")
        lines.append("|---|---|---|---|---|---|")
        groups: dict[str, list[str]] = {}
        for index in common:
            key = base[index]["canonical_key"]
            groups.setdefault(meta.get(key, {}).get(dim, "-"), []).append(index)
        for name in sorted(groups, key=lambda k: -len(groups[k])):
            idx = groups[name]
            b_ok = [i for i in idx if base[i]["correct"]]
            l_ok = [i for i in idx if lora[i]["correct"]]
            tb = sum(1 for i in idx if not base[i]["correct"] and lora[i]["correct"])
            fb = sum(1 for i in idx if base[i]["correct"] and not lora[i]["correct"])
            p = mcnemar_p(tb, fb)
            p_str = "—" if p is None else f"{p:.2e}"
            delta = len(l_ok) / len(idx) - len(b_ok) / len(idx)
            lines.append(f"| {name} | {len(idx)} | {fmt_acc(len(b_ok)/len(idx), len(b_ok))} | "
                         f"{fmt_acc(len(l_ok)/len(idx), len(l_ok))} | {delta*100:+.2f}pt | {p_str} |")
        lines.append("")
    lines.append(f"共 {len(common)} 条配对;base 总 acc {len([i for i in common if base[i]['correct']])/len(common)*100:.2f}%, "
                 f"LoRA 总 acc {len([i for i in common if lora[i]['correct']])/len(common)*100:.2f}%")

    text = "\n".join(lines) + "\n"
    args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
