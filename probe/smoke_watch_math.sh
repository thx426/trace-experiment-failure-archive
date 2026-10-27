#!/bin/bash
# Math 双臂冒烟看护(smoke_watch.sh 的 math 参数版):val=1304、steps=1024
W=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM
L1=$W/runs/classic_math7b_pgs_smoke.launch.log
L2=$W/runs/classic_math7b_grpo_smoke.launch.log
tail -n +1 -F "$L1" "$L2" 2>/dev/null \
 | stdbuf -oL tr "\r" "\n" \
 | grep -E --line-buffered "Traceback|RuntimeError|ValueError|TypeError|KeyError|AssertionError|NotImplementedError|EngineDeadError|out of memory|unexpected pos|malformed|failed twice|Address already in use|val-core" \
 | awk "{k=\$0; gsub(/[0-9]+/,\"#\",k); if(seen[k]++<1){n=length(\$0); print (n>360? substr(\$0,n-359) : \$0); fflush()}}" &
TP=$!
declare -A last
end1=0; end2=0
for i in $(seq 1 55); do
  sleep 60
  for a in pgs grpo; do
    d=$W/runs/classic_math7b_${a}_smoke
    n=$(ps -eo args | grep -ac "[l]aunch_train.py.*math_7b_${a}" || true)
    v=$(find $d/rollouts/step_00000_val -type f ! -size 0 2>/dev/null | wc -l)
    s1=$(find $d/rollouts/step_00001 -type f ! -size 0 2>/dev/null | wc -l)
    s2=$(find $d/rollouts/step_00002 -type f ! -size 0 2>/dev/null | wc -l)
    vq=$(( v==0 ? 0 : (v<1304 ? 1 : 2) )); q1=$(( s1==0 ? 0 : (s1<1024 ? 1 : 2) )); q2=$(( s2==0 ? 0 : (s2<1024 ? 1 : 2) ))
    st="$a alive=$n valq=$vq s1q=$q1 s2q=$q2"
    if [ "$st" != "${last[$a]}" ]; then echo "[$a] alive=$n val=$v/1304 step1=$s1/1024 step2=$s2/1024"; last[$a]="$st"; fi
    if [ "$n" = 0 ]; then
      if [ "$a" = pgs ] && [ $end1 = 0 ]; then end1=1; echo "PGS-EXIT val=$v s1=$s1 s2=$s2"; fi
      if [ "$a" = grpo ] && [ $end2 = 0 ]; then end2=1; echo "GRPO-EXIT val=$v s1=$s1 s2=$s2"; fi
    fi
  done
  if [ $end1 = 1 ] && [ $end2 = 1 ]; then echo "SMOKE-BOTH-TERMINAL"; kill $TP 2>/dev/null; exit 0; fi
done
echo "SMOKE-TIMEOUT-55min"; kill $TP 2>/dev/null
