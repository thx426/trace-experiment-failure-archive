#!/bin/bash
# Phase3 首轮:四域 bare rollout(7B,greedy,定版数据抽样)五路并行
W=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM
D=/fs/fast/u2021000902/wushang/data
PYX=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD/env_sglang_train/bin/python
M=/fs/fast/u2021000902/wushang/model/Qwen/Qwen2.5-7B-Instruct
R=$W/runs/classic_probe
export PYTHONPATH=$W/src

run() { local name=$1 port=$2 pq=$3 cfg=$4
  $PYX -m trace_opd.evalm.run_eval --config $W/configs/classic/$cfg \
    --model-path $M --api-base http://127.0.0.1:$port/v1 --served-model qwen25-7b \
    --parquet $pq --arm bare --temperature 0.0 \
    --out-dir $R/$name > $R/${name}.launch.log 2>&1
  echo "EXIT $name rc=$?"
}

(run alf_vseen30_bare_7b 8250 $D/classic/classic_alfworld_data/valid_seen_head30.parquet alfworld.yaml) &
(for i in $(seq 60); do curl -s -m 2 http://127.0.0.1:8256/v1/models >/dev/null 2>&1 && break; sleep 10; done
 run alf_vunseen30_bare_7b 8256 $D/classic/classic_alfworld_data/valid_unseen_head30.parquet alfworld.yaml) &
(run sqa_valid63_bare_7b 8253 $D/classic/classic_searchqa_data/valid_strat63.parquet searchqa.yaml) &
(run math_val60_bare_7b 8254 $D/classic_math/valid_strat60.parquet math.yaml) &
(run ws_v3test60_bare_7b 8255 $D/classic/classic_webshop_data_v3/test_head60.parquet webshop.yaml) &
wait
echo ALL-DONE
