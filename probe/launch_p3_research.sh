#!/bin/bash
# Phase3 门C-R:35B researcher 批量会话,五路并行(sqa14/math12/ws12/alf12x2)
W=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM
D=/fs/fast/u2021000902/wushang/data
PYX=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD/env_sglang_train/bin/python
M=/fs/fast/u2021000902/wushang/model/Qwen/Qwen2.5-7B-Instruct
R35="http://HOST_IP:8880/v1,http://HOST_IP:8881/v1"
O=$W/runs/classic_research
export PYTHONPATH=$W/src TRACE_35B_KEY=token-abc123
export ALFWORLD_DATA=/fs/fast/u2021000902/wushang/workspace/data/alfworld_data
export ALFWORLD_WORKER_PYTHON=/fs/fast/u2021000902/wushang/workspace/alfworld-venv/bin/python
cd $W

rg() { local name=$1 cfg=$2 pq=$3 n=$4 exb=$5 conc=$6; shift 6
  $PYX probe/research_gen_docs.py --config $W/configs/classic/$cfg \
    --tasks-parquet $pq --n-tasks $n --model-path $M \
    --executor-api-base $exb --executor-model qwen25-7b \
    --researcher-api-base $R35 --researcher-model qwen3.6-35b-a3b \
    --out-dir $O/$name --concurrency $conc "$@" > $O/${name}.launch.log 2>&1
  echo "EXIT $name rc=$?"
}

(rg sqa14 searchqa.yaml $D/classic/classic_searchqa_data/valid_strat63.parquet 14 http://127.0.0.1:8253/v1 12) &
(rg math12 math.yaml $D/classic_math/valid_strat60.parquet 12 http://127.0.0.1:8254/v1 8) &
(rg ws12 webshop.yaml $D/classic/classic_webshop_data_v3/test_head60.parquet 12 http://127.0.0.1:8250/v1 3 --webshop-goals $D/classic/classic_webshop_data_v3/webshop_goals_seed0.jsonl) &
(rg alf_seen12 alfworld.yaml $D/classic/classic_alfworld_data/valid_seen_strat30.parquet 12 http://127.0.0.1:8256/v1 6 --alf-gtwalk-dir $D/classic/classic_alfworld_data/gtwalk_valid_seen_strat30) &
(rg alf_unseen12 alfworld.yaml $D/classic/classic_alfworld_data/valid_unseen_strat30.parquet 12 http://127.0.0.1:8257/v1 6 --alf-gtwalk-dir $D/classic/classic_alfworld_data/gtwalk_valid_unseen_strat30) &
wait
echo RESEARCH-ALL-DONE
