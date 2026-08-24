#!/usr/bin/env bash
# LoRA 训练守护进程（运行在 setsid+tmux 独立 session，与 SSH/VS Code 生命周期完全无关）：
# - 每 5 分钟检查 4 个训练 worker 是否存活；
# - 死亡后宽限 10 分钟（排除保存间隙/部分退出的误判），仍死则清理残留并重启：
#   有 checkpoint 用 --resume-from-checkpoint 续训，否则从头开始；
# - 最多自动重启 2 次，之后放弃并留日志等待人工处理。
# 日志：out/checkpoints/lora_watchdog.log；训练日志：out/checkpoints/lora_console.log
set -u
# 可移植性：路径/GPU 通过环境变量覆盖，默认值对应 vip2023-2 开发机。
PROJ=${PROJ_DIR:-/media/data2/tangzc/proj_3}
LOG=$PROJ/out/checkpoints/lora_console.log
WLOG=$PROJ/out/checkpoints/lora_watchdog.log
CKPT_DIR=$PROJ/out/checkpoints/lora
MODEL=$PROJ/out/models/Qwen--Qwen3.5-9B-Base
CONDA_SH=${CONDA_SH:-/media/data2/tangzc/miniconda3/etc/profile.d/conda.sh}
DEVICES=${LORA_GPUS:-4,5,6,7}
WORKER_PAT="python3.10 -u -m train.train_lora"
TORCHRUN_PAT="torchrun --standalone"

exec >> "$WLOG" 2>&1
echo "[watchdog $(date '+%F %T')] started (pid $$)"

restarts=0
while true; do
  sleep 300
  # pgrep -c 零匹配时会打印 0 并返回退出码 1，不能用 || echo 0（会得到 "0\n0" 破坏整数比较）
  workers=$(pgrep -fc "$WORKER_PAT" 2>/dev/null || true)
  workers=$(printf '%s' "$workers" | tr -dc '0-9')
  [ -z "$workers" ] && workers=0
  [ "$workers" -ge 4 ] && continue
  echo "[watchdog $(date '+%F %T')] workers=$workers (<4), grace 600s"
  sleep 600
  # pgrep -c 零匹配时会打印 0 并返回退出码 1，不能用 || echo 0（会得到 "0\n0" 破坏整数比较）
  workers=$(pgrep -fc "$WORKER_PAT" 2>/dev/null || true)
  workers=$(printf '%s' "$workers" | tr -dc '0-9')
  [ -z "$workers" ] && workers=0
  [ "$workers" -ge 4 ] && continue

  ts=$(date '+%F %T')
  if [ "$restarts" -ge 2 ]; then
    echo "[watchdog $ts] already restarted 2 times, giving up; manual resume required"
    break
  fi
  echo "[watchdog $ts] training dead (workers=$workers); cleaning up and restarting"
  pkill -f "$WORKER_PAT" 2>/dev/null || true
  pkill -f "$TORCHRUN_PAT" 2>/dev/null || true
  sleep 60

  resume=""
  latest=$(ls -d "$CKPT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1)
  [ -n "$latest" ] && resume="--resume-from-checkpoint $latest"
  echo "[watchdog $(date '+%F %T')] relaunch $([ -n "$resume" ] && echo "with $resume" || echo 'from scratch')"

  # shellcheck disable=SC1090
  source "$CONDA_SH" && conda activate omni
  cd "$PROJ" || exit 1
  # setsid：训练进程树脱离 watchdog 的会话/进程组，tmux kill-session 的 SIGHUP 不再波及训练
  # （torchrun elastic agent 会自己安装 SIGHUP 处理器，nohup 挡不住；2026-08-20 曾因此误杀训练）。
  PYTHONPATH=$PROJ CUDA_VISIBLE_DEVICES=$DEVICES setsid nohup torchrun --standalone --nproc_per_node=4 \
    -m train.train_lora --model "$MODEL" \
    --train-data data/processed/sft/train.jsonl \
    --eval-data data/processed/sft/val.jsonl \
    --output-dir out/checkpoints/lora \
    --max-length 4096 $resume \
    >> "$LOG" 2>&1 &
  sleep 600
  restarts=$((restarts + 1))
done
