"""压测期间的 GPU 显存/功耗采集器（README 5.3 指标之一）。

用法:与 bench.py 并行启动，压测结束后 kill 或传 --duration 自动退出。
输出 JSONL:{"ts": 相对秒, "gpus": {<物理GPU编号>: {"mem_used_mib": ..., "power_w": ...}}}
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def poll(gpu_ids: str) -> dict[int, dict]:
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,power.draw",
         "--format=csv,noheader,nounits", "-i", gpu_ids],
        capture_output=True, text=True, timeout=15,
    )
    gpus: dict[int, dict] = {}
    for line in proc.stdout.strip().splitlines():
        index, mem, power = (part.strip() for part in line.split(","))
        gpus[int(index)] = {"mem_used_mib": float(mem), "power_w": float(power)}
    return gpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll GPU memory/power during a benchmark")
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=None,
                        help="采集时长（秒）；缺省持续到被 kill")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with args.output.open("w", encoding="utf-8") as handle:
        while True:
            try:
                gpus = poll(args.gpus)
                handle.write(json.dumps(
                    {"ts": round(time.perf_counter() - started, 1), "gpus": gpus}) + "\n")
                handle.flush()
            except (subprocess.SubprocessError, OSError, ValueError):
                # 单次采集失败（如 GPU 驱动瞬时抖动）跳过，不中断监控。
                pass
            if args.duration is not None and time.perf_counter() - started >= args.duration:
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
