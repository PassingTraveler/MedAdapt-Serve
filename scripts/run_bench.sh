#!/usr/bin/env bash
set -euo pipefail
python -m serving.workload --output "${WORKLOAD_PATH:-out/bench/workload.jsonl}" --per-bucket "${PER_BUCKET:-100}"
python -m serving.bench \
  --workload "${WORKLOAD_PATH:-out/bench/workload.jsonl}" \
  --base-url "${BASE_URL:-http://127.0.0.1:8000/v1}" \
  --model "${MODEL_NAME:?set MODEL_NAME}" \
  --concurrency "${CONCURRENCY:-1}" \
  --output "${OUTPUT_PATH:-out/bench/result.json}" \
  "$@"

