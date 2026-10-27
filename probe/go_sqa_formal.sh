#!/bin/bash
# SearchQA 正式双臂发车序列(2026-08-13 用户令)。前提:GPU 已清场。
set -e
W=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM
echo "── GPU 清场核验 ──"
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "{s+=\$1} END {print s}")
[ "$used" -lt 100 ] || { echo "FATAL: GPU 未清场(合计 ${used}MiB)"; exit 1; }
echo "── 发正式双臂 ──"
bash $W/probe/launch_dual_sqa7b.sh formal
sleep 30
echo "── 起 watchdog(参数化;stop/pid 已预清)──"
setsid env RUN_PREFIX=classic_sqa7b CFG_STEM=searchqa_7b RAY_TAG=sqa TOTAL_STEPS=2650 \
  bash $W/probe/watchdog_dual.sh > /dev/null 2>&1 < /dev/null &
echo "watchdog pid=$!"
sleep 5
tail -3 $W/runs/watchdog_classic_sqa7b.log 2>/dev/null || echo "(watchdog log 尚未写)"
echo "GO-SQA-DONE"
