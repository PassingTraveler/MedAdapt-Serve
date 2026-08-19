from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.build_answer_sft import make_example
from data.utils import iter_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small data smoke test on a JSON file")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args()
    records = []
    for index, record in enumerate(iter_jsonl(args.input)):
        if index >= args.limit:
            break
        records.append(record)
    output = [example for index, record in enumerate(records) if (example := make_example(record, 20260818 + index, shuffle=True)) is not None]
    assert output and all(
        item["answer"] and item["messages"][-1]["content"] == "答案：" + ",".join(item["answer"]) for item in output
    )
    print(json.dumps({"input": str(args.input), "records": len(output), "first": output[0]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
