"""构建通用保持评测数据(README M6):GSM8K test 1,319 条 → JSONL。

输出 data/processed/general/gsm8k_test.jsonl,每行 {"prompt", "answer"}:
- prompt 为英文原题 + 逐步求解指令(0-shot,口径留痕在 manifest);
- answer 保留官方推理与 "#### N" 结尾(eval_general.extract_number 优先取 N)。

HF 下载走 HF_ENDPOINT(本机直连 HF 限流,默认 hf-mirror)。
"""
from __future__ import annotations

import json

from config import PATHS

PROMPT_SUFFIX = "\n\nPlease solve it step by step, and put the final answer after '####'."


def main() -> None:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    assert len(ds) == 1319, f"GSM8K test 行数 {len(ds)} != 1319"

    out_dir = PATHS.processed / "general"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gsm8k_test.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for item in ds:
            question = str(item["question"]).strip()
            answer = str(item["answer"]).strip()
            handle.write(json.dumps({
                "prompt": question + PROMPT_SUFFIX,
                "answer": answer,
                "_source": "openai/gsm8k@main/test",
            }, ensure_ascii=False) + "\n")

    manifest = {
        "source": "openai/gsm8k (main config, test split)",
        "records": len(ds),
        "prompt_mode": "0-shot, 英文原题 + 逐步求解 + #### 收尾指令",
        "output": str(out_path),
    }
    manifest_dir = PATHS.manifests
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "gsm8k_build.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
