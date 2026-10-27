#!/bin/bash
# Phase3 首轮修复重发:alf seen/unseen + math(补 ALFWORLD env var;math-verify 已装)
W=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM
D=/fs/fast/u2021000902/wushang/data
PYX=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD/env_sglang_train/bin/python
M=/fs/fast/u2021000902/wushang/model/Qwen/Qwen2.5-7B-Instruct
R=$W/runs/classic_probe
export PYTHONPATH=$W/src
export ALFWORLD_DATA=/fs/fast/u2021000902/wushang/workspace/data/alfworld_data
export ALFWORLD_WORKER_PYTHON=/fs/fast/u2021000902/wushang/workspace/alfworld-venv/bin/python

run() { local name=$1 port=$2 pq=$3 cfg=$4
  $PYX -m trace_opd.evalm.run_eval --config $W/configs/classic/$cfg \
    --model-path $M --api-base http://127.0.0.1:$port/v1 --served-model qwen25-7b \
    --parquet $pq --arm bare --temperature 0.0 \
    --out-dir $R/$name > $R/${name}.launch.log 2>&1
  echo "EXIT $name rc=$?"
}
(run alf_vseen30_bare_7b 8250 $D/classic/classic_alfworld_data/valid_seen_head30.parquet alfworld.yaml) &
(run alf_vunseen30_bare_7b 8256 $D/classic/classic_alfworld_data/valid_unseen_head30.parquet alfworld.yaml) &
(run math_val60_bare_7b 8254 $D/classic_math/valid_strat60.parquet math.yaml) &
wait
echo FIX-ALL-DONE
