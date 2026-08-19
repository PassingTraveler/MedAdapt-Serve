from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import PATHS


SYSTEM_PREFIX = "你是一个严谨的中文医疗知识助手。仅用于教学和模型评测，不提供个体诊疗建议。"
TOOL_SCHEMA = "可用工具：\n- lookup_guideline(query): 查询固定教学资料\n- calculate(value): 执行确定性计算"
# 长前缀高复用档的共享正文：所有同组请求的 system 消息完全一致。
LONG_GUIDELINE = (
    "医疗知识考核须知：本考核面向模型能力评测，不构成任何诊疗建议。"
    "答题时请先阅读题干与全部选项，再给出最符合临床证据的答案。"
    "涉及药物剂量时参考最新版药典与指南，不依赖单一教材表述。"
    "多选题目按字母顺序输出全部正确选项，单选题只输出一个字母。"
    "不确定时优先选择证据等级最高的诊断或处理方案，并避免过度检查。"
    "传染病题目按国家法定传染病分类标准作答；中医题目按统编教材术语作答。"
)
PREFIX_MODES = ("none", "fixed", "long")
FILLER_UNIT = "请基于给定题干判断最符合的选项。"


def load_optional_tokenizer():
    """有本地权重目录且装了 transformers 时用真实 tokenizer 计量，否则按字符近似。"""
    try:
        from transformers import AutoTokenizer
    except Exception:
        return None
    for candidate in (PATHS.models / "Qwen--Qwen3.5-9B-Base",):
        if (candidate / "tokenizer.json").exists():
            try:
                return AutoTokenizer.from_pretrained(str(candidate))
            except Exception:
                return None
    return None


def measure_tokens(tokenizer, messages: list[dict]) -> int | None:
    if tokenizer is None:
        return None
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    return len(encoded["input_ids"] if isinstance(encoded, dict) else encoded)


def sized_filler(tokenizer, target_tokens: int) -> str:
    """生成接近 target_tokens 的填充文本；无 tokenizer 时按中文字符≈1 token 近似。"""
    if tokenizer is None:
        return FILLER_UNIT * max(1, target_tokens // 20)
    unit_tokens = max(1, len(tokenizer(FILLER_UNIT, add_special_tokens=False)["input_ids"]))
    return FILLER_UNIT * max(1, target_tokens // unit_tokens)


def make_prompt(size: int, reuse_group: int, index: int, prefix_mode: str, tokenizer) -> dict:
    if prefix_mode == "none":
        system_content = None
    elif prefix_mode == "fixed":
        system_content = SYSTEM_PREFIX + "\n" + TOOL_SCHEMA + f"\n共享前缀组：{reuse_group}"
    else:
        system_content = SYSTEM_PREFIX + "\n" + TOOL_SCHEMA + "\n" + LONG_GUIDELINE + f"\n共享前缀组：{reuse_group}"
    messages: list[dict] = []
    if system_content is not None:
        messages.append({"role": "system", "content": system_content})
    user_message = {"role": "user", "content": f"{sized_filler(tokenizer, size)}样本编号 {index}。请输出简短答案。"}
    messages.append(user_message)
    # 用真实 tokenizer 迭代逼近目标 token 数（重复单元近似线性，几轮内收敛到 5%）。
    for _ in range(8):
        tokens = measure_tokens(tokenizer, messages)
        if tokens is None or abs(tokens - size) / size <= 0.05:
            break
        unit_tokens = max(1, len(tokenizer(FILLER_UNIT, add_special_tokens=False)["input_ids"]))
        static_share = tokens - unit_tokens * user_message["content"].count(FILLER_UNIT)
        repeat = max(1, (size - static_share) // unit_tokens)
        user_message["content"] = FILLER_UNIT * repeat + f"样本编号 {index}。请输出简短答案。"
    input_tokens = measure_tokens(tokenizer, messages)
    input_chars = sum(len(str(item["content"])) for item in messages)
    return {
        "id": index,
        "messages": messages,
        "input_bucket": size,
        "prefix_mode": prefix_mode,
        "reuse_group": reuse_group,
        # 无 tokenizer 时用字符数近似（中文≈1 token/字符），此时 input_chars 与之一致。
        "input_tokens": input_tokens if input_tokens is not None else input_chars,
        "input_chars": input_chars,
        "expected_output_tokens": 32 if size == 256 else 128 if size == 1024 else 512,
    }


def resolve_tokenizer(char_fallback: bool):
    """桶大小按 token 定义，必须用真实 tokenizer 计量；只有显式 --char-fallback 才允许
    字符近似（此时压测口径降级，input_tokens 为近似值）。"""
    tokenizer = load_optional_tokenizer()
    if tokenizer is None and not char_fallback:
        raise SystemExit(
            "workload 需要真实 tokenizer 计量输入 token 数（未找到本地 tokenizer 或未安装 "
            "transformers）。若只需无 torch 环境的字符近似 trace，显式传 --char-fallback。"
        )
    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic serving workload trace")
    parser.add_argument("--output", type=Path, default=PATHS.bench / "workload.jsonl")
    parser.add_argument("--per-bucket", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--char-fallback", action="store_true",
                        help="无 tokenizer 时用字符数近似 token（口径降级，仅用于无 torch 环境冒烟）")
    args = parser.parse_args()
    tokenizer = resolve_tokenizer(args.char_fallback)
    rows = []
    index = 0
    for size in (256, 1024, 3072):
        for prefix_mode in PREFIX_MODES:
            for i in range(args.per_bucket):
                rows.append(make_prompt(size, i % 10, index, prefix_mode, tokenizer))
                index += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    token_note = "measured with local tokenizer" if tokenizer is not None else "approximated by chars (tokenizer unavailable)"
    print(f"wrote {len(rows)} requests to {args.output}; input_tokens {token_note}")


if __name__ == "__main__":
    main()
