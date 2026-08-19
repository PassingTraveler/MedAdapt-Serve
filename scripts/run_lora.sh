#!/usr/bin/env bash
set -euo pipefail
python -m train.train_lora \
  --model "${MODEL_ID:-Qwen/Qwen3.5-9B-Base}" \
  --train-data "${TRAIN_DATA:-data/processed/sft/train.jsonl}" \
  --eval-data "${EVAL_DATA:-data/processed/sft/val.jsonl}" \
  --output-dir "${OUTPUT_DIR:-out/checkpoints/lora}" \
  --max-length "${MAX_LENGTH:-4096}" \
  --per-device-batch-size "${PER_DEVICE_BATCH_SIZE:-1}" \
  --gradient-accumulation-steps "${GRAD_ACCUM:-16}" \
  --eval-steps "${EVAL_STEPS:-500}" \
  "$@"
