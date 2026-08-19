#!/usr/bin/env bash
set -euo pipefail
python -m eval.eval_cmb_generate \
  --base-url "${BASE_URL:-http://127.0.0.1:8000/v1}" \
  --model "${MODEL_NAME:?set MODEL_NAME}" \
  --data "${DATA_PATH:-data/processed/cmb/test.jsonl}" \
  --output "${OUTPUT_PATH:-out/eval/cmb_generate.json}" \
  "$@"

