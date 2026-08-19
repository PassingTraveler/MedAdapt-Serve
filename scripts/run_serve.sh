#!/usr/bin/env bash
set -euo pipefail
python -m serving.serve_vllm \
  --model "${MODEL_PATH:?set MODEL_PATH}" \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --tensor-parallel-size "${TP_SIZE:-1}" \
  --enable-prefix-caching \
  "$@"

