#!/usr/bin/env bash
set -euo pipefail
python -m data.fetch_cmb "$@"
python -m data.inspect_cmb
python -m data.dedup_splits
python -m data.build_answer_sft
python -m data.build_answer_sft \
  --input data/processed/cmb/val.jsonl \
  --output data/processed/sft/val.jsonl \
  --max-records 280 \
  --no-shuffle
