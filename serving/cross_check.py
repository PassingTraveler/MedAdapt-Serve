"""vLLM 服务 vs 本地 transformers 前向的正确性对照。

在 omni 环境运行(transformers 侧加载 merged 模型),对照对象是正在运行的
vLLM 服务。对每个样本:同一 prompt 下两边的 greedy token 序列必须逐位一致,
每个生成 token 的 logprob 差值必须落在 bf16 数值差异量级(~1e-2)内;
差异来源是 bf16 GEMM/softmax 实现不同,不属于正确性问题。

logprob 对比用教师强制(vllm 的 token 序列喂 transformers 前向逐位算),
greedy 分歧后自生成侧的前缀不同,逐位 diff 会混入跨分支噪声(实测一条
near-tie 翻转曾造成 1.096 nats 的假差异);greedy 一致性单独记录
first_divergence_at。

vLLM 侧用 /completions + logprobs=1(不套 chat template,两边 tokenize 口径
一致);transformers 侧 argmax 在 bf16 logits 上取(与推理一致),
logprob 用 fp32 log_softmax 计算(vLLM 同样在 fp32 算 logprob)。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

from config import PATHS


def load_prompts(path: Path, limit: int) -> list[str]:
    prompts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        question = json.loads(line).get("question", "").strip()
        if question:
            prompts.append(question)
        if len(prompts) >= limit:
            break
    return prompts


def vllm_complete(base_url: str, model: str, prompt: str, max_tokens: int) -> dict:
    response = requests.post(
        base_url.rstrip("/") + "/completions",
        json={"model": model, "prompt": prompt, "max_tokens": max_tokens,
              "temperature": 0, "logprobs": 1},
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()
    choice = data["choices"][0]
    return {
        "text": choice["text"],
        "token_logprobs": choice["logprobs"]["token_logprobs"],
        "usage": data.get("usage"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-check vLLM serving vs transformers forward")
    parser.add_argument("--model-dir", required=True, help="transformers 侧模型目录（merged）")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True, help="服务端 served-model-name")
    parser.add_argument("--samples", type=Path, default=PATHS.processed / "cmb" / "val.jsonl")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path, default=PATHS.eval / "cross_check.json")
    parser.add_argument("--dtype", default="bfloat16",
                        help="transformers 侧加载 dtype;量化模型必须 float16(vllm GPTQ 侧即 "
                             "fp16 激活,bf16 加载实测 lm_head matmul dtype 冲突)")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    # 量化模型(gptqmodel TritonV2Linear 等)不支持先 cpu 加载再 .to("cuda")
    # (实测 ValueError: does not support device: cpu),必须 device_map 在加载时
    # 分发到 GPU;bf16 模型两种方式数值等价,统一走 device_map。
    model = (AutoModelForCausalLM.from_pretrained(args.model_dir,
                                                  torch_dtype=getattr(torch, args.dtype),
                                                  device_map="auto").eval())
    input_device = model.device

    results = []
    for index, prompt in enumerate(load_prompts(args.samples, args.limit)):
        served = vllm_complete(args.base_url, args.model, prompt, args.max_tokens)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(prompt + served["text"], add_special_tokens=False)["input_ids"]
        new_ids = full_ids[len(prompt_ids):]

        # 1) greedy 自生成:两边各自 argmax,比较 token 序列是否逐位一致。
        ids = torch.tensor([prompt_ids], device=input_device)
        chosen: list[int] = []
        with torch.no_grad():
            for _ in range(len(new_ids)):
                logits = model(input_ids=ids).logits
                top = int(torch.argmax(logits[0, -1]).item())
                chosen.append(top)
                ids = torch.cat([ids, torch.tensor([[top]], device=input_device)], dim=1)

        # 2) 教师强制:vllm 生成的序列喂 transformers 前向,逐位对比 logprob。
        # 两边前缀一致,全序列 diff 都有意义(greedy 分歧后自生成侧 diff 无意义,
        # 会混入跨分支噪声,不能用来判数值对齐)。
        ids = torch.tensor([prompt_ids], device=input_device)
        forced_logprobs: list[float] = []
        with torch.no_grad():
            for t in range(len(new_ids)):
                logits = model(input_ids=ids).logits
                logprob = float(torch.log_softmax(logits[0, -1].float(), dim=-1)[new_ids[t]].item())
                forced_logprobs.append(logprob)
                ids = torch.cat([ids, torch.tensor([[new_ids[t]]], device=input_device)], dim=1)

        vllm_logprobs = served["token_logprobs"]
        aligned = len(vllm_logprobs) == len(new_ids)
        diffs = [abs(vllm_logprobs[t] - forced_logprobs[t])
                 for t in range(min(len(vllm_logprobs), len(new_ids)))]
        first_divergence_at = next((t for t, (a, b) in enumerate(zip(chosen, new_ids))
                                    if a != b), None)
        results.append({
            "index": index,
            "prompt_chars": len(prompt),
            "tokens_generated": len(new_ids),
            "vllm_text": served["text"],
            "transformers_text": tokenizer.decode(chosen, skip_special_tokens=True),
            "text_equal": served["text"] == tokenizer.decode(chosen, skip_special_tokens=True),
            "token_sequence_equal": chosen == new_ids,
            "first_divergence_at": first_divergence_at,
            "tokenizer_alignment_ok": aligned,
            "usage": served["usage"],
            "mean_abs_logprob_diff": sum(diffs) / len(diffs) if diffs else None,
            "max_abs_logprob_diff": max(diffs) if diffs else None,
            "per_token_logprob_diff": diffs,  # 教师强制下逐位对比
        })
        print(f"[{index}] tokens={len(new_ids)} text_equal={results[-1]['text_equal']} "
              f"seq_equal={results[-1]['token_sequence_equal']} "
              f"mean|Δlogp|={results[-1]['mean_abs_logprob_diff']:.5f} "
              f"max|Δlogp|={results[-1]['max_abs_logprob_diff']:.5f}", flush=True)

    summary = {
        "model_dir": args.model_dir,
        "served_model": args.model,
        "samples": len(results),
        "all_text_equal": all(r["text_equal"] for r in results),
        "all_token_sequences_equal": all(r["token_sequence_equal"] for r in results),
        "all_alignment_ok": all(r["tokenizer_alignment_ok"] for r in results),
        "mean_abs_logprob_diff": sum(r["mean_abs_logprob_diff"] for r in results) / len(results),
        "max_abs_logprob_diff": max(r["max_abs_logprob_diff"] for r in results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "results": results},
                                      ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
