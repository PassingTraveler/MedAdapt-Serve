"""构建跨题库评测数据(README M6):CMExam test 6,811 条 → CMB 归一格式 JSONL。

CMExam(NeurIPS 2023)源自中国国家医师资格考试(NMLE),与 CMB 出题源
不同,选它做跨题库(README 计划的两候选 MedBench/CMMLU 实测均不可用:
opencompass 版 MedBench 官方 test answer=null;ECNU 版只公开几百题
Resident;CMMLU test 无答案、dev 仅 5 题/子集)。

关键步骤:与 CMB train 的 canonical_key 零重叠验证——两库都源自医师
考试,可能重复;重叠题必须剔除,否则"跨题库"名不副实(README 评审项:
CMB 背题或跨 split 近重复)。

输入:原始 CMExam test.json(JSONL,每行 {Question, Options[{key,value}],
Answer, Explanation});镜像 fzkuji/CMExam(hf-mirror 可直连)。
输出:data/processed/cmexam/test.jsonl,CMB 归一格式(question/option
dict/answer/question_type/exam_type)。
"""
from __future__ import annotations

import json
from pathlib import Path

import requests

from config import PATHS
from data.utils import canonical_question_key, iter_jsonl

CMEXAM_URL = "https://hf-mirror.com/datasets/fzkuji/CMExam/resolve/main/test.json"


def fetch_raw() -> Path:
    out_dir = PATHS.raw / "cmexam"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "test.json"
    if not raw_path.exists() or raw_path.stat().st_size < 1_000_000:
        r = requests.get(CMEXAM_URL, timeout=180)
        r.raise_for_status()
        raw_path.write_bytes(r.content)
        print(f"downloaded {raw_path.stat().st_size} bytes to {raw_path}")
    return raw_path


def convert(row: dict) -> dict:
    question = str(row.get("Question", "")).strip()
    options = {}
    for opt in row.get("Options") or []:
        letter = str(opt.get("key", "")).strip().upper()
        value = str(opt.get("value", "")).strip()
        if letter and value:
            options[letter] = value
    answer = str(row.get("Answer", "")).strip()
    return {
        "question": question,
        "option": options,
        "answer": answer,
        "question_type": "单项选择题" if len(answer) <= 1 else "多项选择题",
        "exam_type": "医师资格考试(NMLE)",
        "_source": "CMExam@NeurIPS2023/test",
    }


def main() -> None:
    raw_path = fetch_raw()
    rows = []
    for line in raw_path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    print(f"raw rows: {len(rows)}")
    assert len(rows) == 6811, f"CMExam test 行数 {len(rows)} != 6811"

    # CMB train 的 canonical_key 集合(文件已带 _canonical_key)。
    train_keys: set[str] = set()
    for record in iter_jsonl(PATHS.processed / "cmb" / "train.jsonl"):
        train_keys.add(record["_canonical_key"])
    print(f"cmb train keys: {len(train_keys)}")

    converted, overlap = [], []
    for i, row in enumerate(rows):
        record = convert(row)
        key = canonical_question_key(record)
        if key in train_keys:
            overlap.append({"index": i, "question": record["question"][:60]})
            continue
        record["_canonical_key"] = key
        converted.append(record)

    out_dir = PATHS.processed / "cmexam"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for record in converted:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "source": "CMExam (NeurIPS 2023, 中国国家医师资格考试 NMLE)",
        "mirror": "fzkuji/CMExam@hf-mirror",
        "raw_rows": len(rows),
        "kept": len(converted),
        "overlap_with_cmb_train": len(overlap),
        "overlap_samples": overlap[:5],
        "output": str(out_path),
        "rationale": ("README 计划候选 MedBench/CMMLU 实测不可用(官方 test 无答案/"
                      "仅数百题),CMExam 同为医师考试出题但全部公开答案,"
                      "且与 CMB train canonical_key 零重叠验证后保留。"),
    }
    (PATHS.manifests / "cmexam_build.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
