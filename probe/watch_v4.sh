#!/usr/bin/env bash
# 稳定性看护 v4:在步进/长度之外加熵坍缩与格式服从两条告警线(2026-08-16 SQA g237
# 熵断崖后新增)。零额外 IO——只读已有日志的尾部,不写任何测试文件。
W=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM
SQA_LOG=$W/runs/classic_sqa7b_pgs.launch.log
MATH_LOG=$(ls -S /tmp/ray_math_pgs/ray/session_latest/logs/worker-*.out 2>/dev/null | head -1)

# 从一行 verl console 指标里抠一个键
pick() { grep -o "$2:[0-9.]*" <<<"$1" | tail -1 | cut -d: -f2; }
# 取日志里最后一条含 actor/entropy 的 step 行
lastrow() { grep -o "step:[0-9]* - .*actor/entropy[^\\\\]*" "$1" 2>/dev/null | tail -1; }

fmt() { awk -v v="$1" -v s="$2" 'BEGIN{if(v=="")printf"-";else printf "%.*f", s, v}'; }

dead_s=0; dead_m=0
while :; do
  sq_step=$(ls -d $W/runs/classic_sqa7b_pgs/rollouts/step_* 2>/dev/null | grep -v val | sed 's#.*step_##' | sort -n | tail -1)
  ma_step=$(ls -d $W/runs/classic_math7b_pgs/rollouts/step_* 2>/dev/null | grep -v val | sed 's#.*step_##' | sort -n | tail -1)

  sr=$(lastrow "$SQA_LOG"); mr=$(lastrow "$MATH_LOG")
  s_ent=$(pick "$sr" "actor/entropy");  m_ent=$(pick "$mr" "actor/entropy")
  s_bad=$(pick "$sr" "invalid_format/mean"); m_bad=$(pick "$mr" "invalid_format/mean")
  s_gate=$(pick "$sr" "steer_gate_open_rate/mean"); m_gate=$(pick "$mr" "steer_gate_open_rate/mean")
  s_iv=$(pick "$sr" "steer_count/mean");   m_iv=$(pick "$mr" "steer_count/mean")
  s_won=$(pick "$sr" "won/mean");          m_won=$(pick "$mr" "won/mean")
  s_len=$(pick "$sr" "response_length/mean"); m_len=$(pick "$mr" "response_length/mean")

  # 进程存活(缺一即告警)+ 活跃度(判停滞只能看产物增量,不能看进程名)
  alive=$(pgrep -fc "[m]ain_ppo" 2>/dev/null || echo 0)
  fs=$(find $W/runs/classic_sqa7b_pgs/rollouts -name "*.json" ! -size 0 -newermt "-16 minutes" 2>/dev/null | wc -l)
  fm=$(find $W/runs/classic_math7b_pgs/rollouts -name "*.json" ! -size 0 -newermt "-16 minutes" 2>/dev/null | wc -l)
  [ "${fs:-0}" -eq 0 ] && dead_s=$((dead_s+1)) || dead_s=0
  [ "${fm:-0}" -eq 0 ] && dead_m=$((dead_m+1)) || dead_m=0

  line="[$(date +%H:%M)] SQA=$sq_step ent=$(fmt "$s_ent" 3) bad=$(awk -v v="$s_bad" 'BEGIN{printf "%.1f%%", 100*v}') gate=$(awk -v v="$s_gate" 'BEGIN{printf "%.0f%%", 100*v}') iv=$(fmt "$s_iv" 0) won=$(awk -v v="$s_won" 'BEGIN{printf "%.0f%%", 100*v}') len=$(fmt "$s_len" 0)"
  line="$line | MATH=$ma_step ent=$(fmt "$m_ent" 3) bad=$(awk -v v="$m_bad" 'BEGIN{printf "%.1f%%", 100*v}') iv=$(fmt "$m_iv" 0) won=$(awk -v v="$m_won" 'BEGIN{printf "%.0f%%", 100*v}') len=$(fmt "$m_len" 0)"

  flag=""
  awk -v v="$s_ent" 'BEGIN{exit !(v!="" && v<0.05)}' && flag="$flag ⚠SQA熵<0.05"
  awk -v v="$m_ent" 'BEGIN{exit !(v!="" && v<0.05)}' && flag="$flag ⚠MATH熵<0.05"
  awk -v v="$s_bad" 'BEGIN{exit !(v!="" && v>0.10)}' && flag="$flag ⚠SQA格式失守>10%"
  awk -v v="$m_bad" 'BEGIN{exit !(v!="" && v>0.10)}' && flag="$flag ⚠MATH格式失守>10%"
  [ "$alive" -lt 2 ] && flag="$flag ⚠训练进程只剩${alive}个"
  [ $dead_s -ge 2 ] && flag="$flag ⚠SQA停滞${dead_s}轮"
  [ $dead_m -ge 2 ] && flag="$flag ⚠MATH停滞${dead_m}轮"

  echo "$line$flag"
  sleep 900
done
