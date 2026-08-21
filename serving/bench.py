from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def stream_request(base_url: str, model: str, row: dict, timeout: int = 180,
                   temperature: float = 0.0, seed: int | None = None) -> dict:
    started = time.perf_counter()
    first_token = None
    output = []
    error = None
    usage = None
    body = {"model": model, "messages": row["messages"], "temperature": temperature,
            "max_tokens": row["expected_output_tokens"], "stream": True,
            # OpenAI 兼容接口的 usage 回传：拿服务端真实 token 计数（vLLM 支持）。
            "stream_options": {"include_usage": True}}
    # README 5.3：greedy 和固定 seed 的 sampling 分开跑；seeded 模式显式传 seed。
    if seed is not None:
        body["seed"] = seed
    try:
        response = requests.post(
            base_url.rstrip("/") + "/chat/completions",
            headers={"Content-Type": "application/json"},
            json=body,
            stream=True,
            timeout=timeout,
        )
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                continue
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if data.get("usage"):
                # vllm 的 usage-only 结束块 choices=[]，跳过（否则下一行取 [0] 越界）。
                usage = data["usage"]
                continue
            choices = data.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {}).get("content")
            if delta:
                # 只有首个内容块计 TTFT；usage-only 结束块不计。
                if first_token is None:
                    first_token = time.perf_counter()
                output.append(delta)
    except (requests.RequestException, OSError) as exc:
        # 单个请求失败/超时只记录，不中断整个压测（README 5.3 要求记录失败/超时）。
        error = f"{type(exc).__name__}: {exc}"
    ended = time.perf_counter()
    output_text = "".join(output)
    output_chars = len(output_text)
    completion_tokens = int(usage["completion_tokens"]) if usage and usage.get("completion_tokens") else None
    e2e_ms = (ended - started) * 1000
    ttft_ms = (first_token - started) * 1000 if first_token else None
    # TPOT 以服务端真实 completion_tokens 为口径；无 usage 回退时降级为字符口径并留痕。
    if completion_tokens and first_token is not None:
        tpot_ms = (ended - first_token) * 1000 / completion_tokens
        token_source = "usage"
    else:
        tpot_ms = (e2e_ms - ttft_ms) / output_chars if (ttft_ms is not None and output_chars) else None
        token_source = "chars"
    return {
        "id": row["id"],
        "input_bucket": row["input_bucket"],
        "prefix_mode": row.get("prefix_mode", "fixed"),
        "reuse_group": row["reuse_group"],
        "started": started,
        "ended": ended,
        "ttft_ms": ttft_ms,
        "e2e_ms": e2e_ms,
        "tpot_ms": tpot_ms,
        "token_source": token_source,
        "completion_tokens": completion_tokens,
        "output_chars": output_chars,
        "error": error,
    }


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((len(values) - 1) * p)))
    return values[index]


def summarize(name: str, measured: list[dict], wall_clock_seconds: float | None) -> dict:
    values = [row[name] for row in measured if row.get(name) is not None]
    return {
        "mean": statistics.mean(values) if values else None,
        "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-trace streaming latency benchmark")
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--formal-per-group", type=int, default=100, help="Formal measured requests per (input_bucket, reuse_group)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sampling", choices=("greedy", "seeded"), default="greedy",
                        help="greedy=temperature 0；seeded=固定 seed 的随机采样（README 5.3 分开跑）")
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="仅 seeded 模式生效；greedy 固定为 0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    temperature = 0.0 if args.sampling == "greedy" else args.temperature
    seed = None if args.sampling == "greedy" else args.seed
    rows = [json.loads(line) for line in args.workload.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = rows[: args.limit] if args.limit else rows

    # 按 (input_bucket, prefix_mode, reuse_group) 分组；预热与正式都覆盖全部组合（README 5.3）。
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["input_bucket"], row.get("prefix_mode", "fixed"), row["reuse_group"]), []).append(row)

    # 预热：每组固定 --warmup 次，行不足时循环复用（README 5.3：每组预热 20 次）。
    for key, group in sorted(groups.items()):
        for i in range(args.warmup):
            stream_request(args.base_url, args.model, group[i % len(group)],
                           temperature=temperature, seed=seed)

    # 正式：每组 --formal-per-group 次，行不足时循环复用（保持固定 trace 可复现）。
    formal_rows = []
    for key, group in sorted(groups.items()):
        for i in range(args.formal_per_group):
            formal_rows.append(group[i % len(group)])

    measured = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(stream_request, args.base_url, args.model, row,
                                   temperature=temperature, seed=seed) for row in formal_rows]
        for future in as_completed(futures):
            measured.append(future.result())

    failed = [row for row in measured if row["error"]]
    started = min((row["started"] for row in measured), default=None)
    ended = max((row["ended"] for row in measured), default=None)
    wall_clock_seconds = (ended - started) if (started is not None and ended is not None) else None
    total_chars = sum(row["output_chars"] for row in measured)
    total_tokens = sum(row["completion_tokens"] for row in measured if row["completion_tokens"])
    token_sources = {row["token_source"] for row in measured}
    summary = {
        "model": args.model,
        "workload": str(args.workload),
        "concurrency": args.concurrency,
        # 采样口径留痕（README 5.3：greedy 与固定 seed 的 sampling 分开报告）。
        "sampling": args.sampling,
        "temperature": temperature,
        "seed": seed,
        "stop": "max_tokens",
        "requests": len(measured),
        "failed": len(failed),
        "error_rate": len(failed) / len(measured) if measured else 0,
        # 主口径：服务端真实 token 指标（usage 回传）。chars 字段仅作辅助保留。
        "token_sources": sorted(token_sources),
        "ttft_ms": summarize("ttft_ms", measured, wall_clock_seconds),
        "e2e_ms": summarize("e2e_ms", measured, wall_clock_seconds),
        "tpot_ms": summarize("tpot_ms", measured, wall_clock_seconds),
        # 并发下吞吐分母用墙钟时间，而不是各请求 e2e 之和。
        "completion_tokens_per_second": total_tokens / wall_clock_seconds if wall_clock_seconds else 0,
        "output_chars_per_second": total_chars / wall_clock_seconds if wall_clock_seconds else 0,
        "requests_per_second": len(measured) / wall_clock_seconds if wall_clock_seconds else 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "requests": measured}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
