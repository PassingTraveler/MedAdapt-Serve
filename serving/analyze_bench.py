"""汇总固定 trace 压测各档结果(README 5.3 表格素材)。

从 out/bench/bench_<model>_c<conc>_<sampling>.json 读取各档 summary;
GPU 显存/功耗来自 gpu_monitor JSONL:本机部署 bf16 serve 在物理 GPU 4、
GPTQ serve 在 GPU 5,monitor 全程覆盖两条服务,按 GPU 取全程均值。
缺档会明确标 MISSING(不静默跳过)。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# 本机实测部署:bf16 serve 物理 GPU 4,GTPQ serve 物理 GPU 5(见 out/serving/serve_*.log)。
GPU_FOR_MODEL = {"bf16": "4", "gptq": "5"}


def fmt(value) -> str:
    return "-" if value is None else f"{value:.1f}"


def stats_of(summary: dict, key: str) -> dict:
    value = summary.get(key) or {}
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-dir", type=Path, default=Path("out/bench"))
    parser.add_argument("--monitor", type=Path, default=Path("out/bench/gpu_monitor_bf16.jsonl"))
    parser.add_argument("--models", nargs="+", default=("bf16", "gptq"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    monitor: list[dict] = []
    if args.monitor.exists():
        monitor = [json.loads(line) for line in args.monitor.read_text(encoding="utf-8").splitlines() if line.strip()]

    # 每模型对应 GPU 的全过程均值
    gpu_stats: dict[str, dict] = {}
    for model, gpu in GPU_FOR_MODEL.items():
        samples = []
        for row in monitor:
            if gpu in row.get("gpus", {}):
                samples.append(row["gpus"][gpu])
        if samples:
            n = len(samples)
            gpu_stats[model] = {
                "mem_mib": round(sum(s["mem_used_mib"] for s in samples) / n),
                "power_w": round(sum(s["power_w"] for s in samples) / n, 1),
            }
    print(f"# GPU monitor 全程均值: {json.dumps(gpu_stats, ensure_ascii=False)} "
          f"(采样 {len(monitor)} 点, {monitor[0]['ts']:.0f}s~{monitor[-1]['ts']:.0f}s)\n")

    rows: list[dict] = []
    for model in args.models:
        for sampling in ("greedy", "seeded"):
            for conc in (1, 4, 8, 16):
                path = args.bench_dir / f"bench_{model}_c{conc}_{sampling}.json"
                row = {"model": model, "sampling": sampling, "conc": conc}
                if not path.exists():
                    row["status"] = "MISSING"
                    rows.append(row)
                    print(f"| {model} {sampling} c{conc} | MISSING |")
                    continue
                summary = json.loads(path.read_text(encoding="utf-8")).get("summary", {})
                ttft, tpot, e2e = (stats_of(summary, k) for k in ("ttft_ms", "tpot_ms", "e2e_ms"))
                row.update({
                    "status": "ok",
                    "requests": summary.get("requests"),
                    "failed": summary.get("failed"),
                    "error_rate": summary.get("error_rate"),
                    "ttft_p50": fmt(ttft.get("p50")), "ttft_p95": fmt(ttft.get("p95")),
                    "ttft_p99": fmt(ttft.get("p99")),
                    "tpot_p50": fmt(tpot.get("p50")), "tpot_p95": fmt(tpot.get("p95")),
                    "tpot_p99": fmt(tpot.get("p99")),
                    "e2e_p50": fmt(e2e.get("p50")), "e2e_p99": fmt(e2e.get("p99")),
                    "out_tok_s": round(summary.get("completion_tokens_per_second") or 0, 1),
                    "wall_s": round(summary.get("wall_clock_seconds") or 0),
                    "req_s": round(summary.get("requests_per_second") or 0, 2),
                })
                rows.append(row)
                print(f"| {model} {sampling} c{conc} | ok | {row['requests']} req / {row['failed']} fail "
                      f"({row['error_rate']}) | TTFT {row['ttft_p50']}/{row['ttft_p95']}/{row['ttft_p99']}ms | "
                      f"TPOT {row['tpot_p50']}/{row['tpot_p95']}/{row['tpot_p99']}ms | "
                      f"e2e {row['e2e_p50']}/{row['e2e_p99']}ms | {row['out_tok_s']} tok/s | {row['wall_s']}s |")

    if args.output:
        args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
