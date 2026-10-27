#!/bin/bash
# Phase3 门C-R 第二段:doc 臂 eval(35B researcher docs → 7B,greedy)
W=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM
D=/fs/fast/u2021000902/wushang/data
PYX=/fs/fast/u2021000902/wushang/workspace/TRACE-OPD/env_sglang_train/bin/python
M=/fs/fast/u2021000902/wushang/model/Qwen/Qwen2.5-7B-Instruct
O=$W/runs/classic_research
R=$W/runs/classic_probe
export PYTHONPATH=$W/src
export ALFWORLD_DATA=/fs/fast/u2021000902/wushang/workspace/data/alfworld_data
export ALFWORLD_WORKER_PYTHON=/fs/fast/u2021000902/wushang/workspace/alfworld-venv/bin/python

flt() { $PYX - "$1" "$2" "$3" <<'PYEOF'
import sys, pandas as pd
from pathlib import Path
pq, docdir, out = sys.argv[1:4]
df = pd.read_parquet(pq)
def safe(t): return t.replace("/", "__").replace(" ", "_")
dd = Path(docdir)
keep = df.extra_info.apply(lambda e: (dd / (safe(e["task_id"]) + ".md")).exists())
df[keep].to_parquet(out)
print(f"[flt] {out}: {int(keep.sum())}/{len(df)}")
PYEOF
}

run() { local name=$1 port=$2 pq=$3 cfg=$4 docdir=$5; shift 5
  rm -rf $R/$name
  $PYX -m trace_opd.evalm.run_eval --config $W/configs/classic/$cfg \
    --model-path $M --api-base http://127.0.0.1:$port/v1 --served-model qwen25-7b \
    --parquet $pq --arm doc --doc-dir $docdir --temperature 0.0 "$@" \
    --out-dir $R/$name > $R/${name}.launch.log 2>&1
  echo "EXIT $name rc=$?"
}

flt $D/classic_math/valid_strat60.parquet $O/math12/docs /tmp/math12_doc.parquet
flt $D/classic/classic_webshop_data_v3/test_head60.parquet $O/ws12/docs /tmp/ws12_doc.parquet
flt $D/classic/classic_alfworld_data/valid_seen_strat30.parquet $O/alf_seen12/docs /tmp/alfs12_doc.parquet
flt $D/classic/classic_alfworld_data/valid_unseen_strat30.parquet $O/alf_unseen12/docs /tmp/alfu12_doc.parquet

(run math12_doc35b_7b 8254 /tmp/math12_doc.parquet math.yaml $O/math12/docs) &
(run ws12_doc35b_7b 8250 /tmp/ws12_doc.parquet webshop.yaml $O/ws12/docs --parallel 8) &
(run alf_seen12_doc35b_7b 8256 /tmp/alfs12_doc.parquet alfworld.yaml $O/alf_seen12/docs) &
(run alf_unseen12_doc35b_7b 8257 /tmp/alfu12_doc.parquet alfworld.yaml $O/alf_unseen12/docs) &
wait
echo DOCARM-ALL-DONE
