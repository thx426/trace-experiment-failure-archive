#!/usr/bin/env bash
# 抓训练步指标行(pg_loss/grad_norm/reward/entropy),用法: step_metrics.sh <arm>
W=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM
ARM=${1:?arm}
LOG=$W/runs/classic_alf7b_${ARM}_smoke.launch.log
echo "===== $ARM ====="
grep -o 'step:[0-9]* - .*' "$LOG" | while IFS= read -r line; do
  step=$(echo "$line" | grep -o '^step:[0-9]*')
  vals=$(echo "$line" | tr ' ' '\n' | grep -E 'pg_loss:|grad_norm:|actor/entropy:|actor/loss:|reward/mean|won/mean|env_steps/mean|invalid_format/mean|response_length/mean' | tr '\n' ' ')
  echo "$step  $vals"
done
echo "--- Traceback/Killed 计数:"
grep -cE "Traceback|killed by signal" "$LOG"
echo "--- ckpt:"
ls -d $W/runs/classic_alf7b_${ARM}_smoke/checkpoints/* 2>/dev/null
