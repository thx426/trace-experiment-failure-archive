#!/bin/bash
# Math 文本 TIR 协议探针:Qwen2.5-7B-Instruct bare greedy @ valid_strat60(6源×10)
# 产物:runs/probe_math_text/strat60/{episodes.jsonl,summary.json,manifest.json}
W=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM
D=/fs/fast/u2021000902/wushang/data
PYX=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD/env_sglang_train/bin/python
M=/fs/fast/u2021000902/wushang/model/Qwen/Qwen2.5-7B-Instruct
R=$W/runs/probe_math_text
mkdir -p $R
# 编译前置入口(硬规则)
source /fs/fast/u2021000902/wushang/workspace/TRACE-OPD-PGS/env_pgs_sglang_a800.sh >/dev/null 2>&1
# ★ 必须在 source 之后:入口脚本会把 PYTHONPATH 覆写到 PGS 树
# (TRACE-OPD-PGS/src)——写在前面会让 -m trace_opd... 静默解析到另一棵树
# 的旧代码(2026-08-13 实锤:argparse 报不认识 --temperature 才暴露)。
export PYTHONPATH=$W/src

# 1) 7B server(CVD=7 / port 8254;文本协议走 /v1/completions 自解析路径,无需 tool parser)
setsid env CUDA_VISIBLE_DEVICES=7 $PYX -m sglang.launch_server \
  --model-path $M --served-model-name qwen25-7b --port 8254 \
  --mem-fraction-static 0.85 --context-length 32768 \
  > $R/server.log 2>&1 < /dev/null &
SRV=$!
echo "SERVER-PID $SRV"
ok=0
for i in $(seq 1 90); do
  curl -s -m 2 http://127.0.0.1:8254/v1/models >/dev/null 2>&1 && { ok=1; break; }
  sleep 5
done
[ $ok = 1 ] || { echo "SERVER-FAILED(450s 未就绪)"; tail -5 $R/server.log; exit 1; }
echo "SERVER-READY"

# 2) 真实往返冒烟门(张量层判据,不信 is_available)
curl -s -m 30 http://127.0.0.1:8254/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen25-7b","prompt":"1+1=","max_tokens":4,"temperature":0}' \
  | grep -q '"text"' && echo "ROUNDTRIP-OK" || { echo "ROUNDTRIP-FAILED"; exit 1; }

# 3) run_eval(text 协议在 math.yaml 中显式;greedy;并发 16)
$PYX -m trace_opd.evalm.run_eval --config $W/configs/classic/math.yaml \
  --model-path $M --api-base http://127.0.0.1:8254/v1 --served-model qwen25-7b \
  --parquet $D/classic_math/${PQ:-valid_strat60.parquet} --arm bare --temperature 0.0 \
  --parallel 16 --out-dir $R/${TAG:-strat60} --force > $R/${TAG:-strat60}.launch.log 2>&1
rc=$?
echo "EVAL-EXIT rc=$rc"
tail -12 $R/strat60.launch.log
# 4) 清 server(按 PGID,含 sglang 子进程)
kill -- -$SRV 2>/dev/null; sleep 3; kill -9 -- -$SRV 2>/dev/null
echo "PROBE-DONE rc=$rc"
