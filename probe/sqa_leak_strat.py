#!/usr/bin/env python3
"""SearchQA 泄漏分层配对:答案在/不在 doc 的 task 的 steered-bare gap。"""
import json, glob, os, collections, re
from math import comb
import pandas as pd

run = "/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM/runs/classic_sqa7b_pgs"
gt = {}
df = pd.read_parquet("/fs/fast/u2021000902/wushang/data/classic/classic_searchqa_data/train_trainer.parquet")
for _, row in df.iterrows():
    ei = dict(row["extra_info"])
    ga = ei.get("golden_answers")
    ga = [ga] if isinstance(ga, str) else (list(ga) if ga is not None else [])
    gt[str(ei["task_id"])] = [str(x) for x in ga]

def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

# task → doc 泄漏标志(取该 task 任一会话 doc)
leak = {}
for f in glob.glob(os.path.join(run, "research", "*", "*", "doc.md")):
    tdir = f.split("/research/")[1].split("/")[0]
    tid = tdir.rsplit("--", 1)[0].replace("__", "/", 1)
    ans = [norm(a) for a in gt.get(tid, []) if len(norm(a)) >= 3]
    if not ans:
        leak[tid] = None; continue
    d = norm(open(f, encoding="utf-8", errors="replace").read())
    leak[tid] = any(a in d for a in ans)

cells = {True: [], False: [], None: []}
for d in sorted(glob.glob(os.path.join(run, "rollouts", "step_*"))):
    base = os.path.basename(d)
    if "val" in base or int(base.split("_")[1]) < 2:
        continue
    by = collections.defaultdict(list)
    for f in glob.glob(os.path.join(d, "*.json")):
        if os.path.getsize(f) == 0: continue
        try: r = json.load(open(f))
        except Exception: continue
        by[r["task_id"]].append(r)
    for t, rs in by.items():
        if len(rs) != 8: continue
        s = [x for x in rs if (x.get("steering") or {}).get("steer_arm") == 1]
        b = [x for x in rs if (x.get("steering") or {}).get("steer_arm") == 0]
        if len(s) != 4 or len(b) != 4: continue
        sw = sum(1 for x in s if x.get("won")) / 4
        bw = sum(1 for x in b if x.get("won")) / 4
        cells[leak.get(t)].append((sw, bw))

for k, name in [(True, "答案在 doc "), (False, "答案不在 doc"), (None, "答案短/无doc")]:
    v = cells[k]
    if not v:
        print(f"{name}: 0 task"); continue
    n = len(v)
    sw = sum(x for x, _ in v) / n; bw = sum(x for _, x in v) / n
    pos = sum(1 for x, y in v if x > y); neg = sum(1 for x, y in v if x < y)
    m = pos + neg
    p = 2 * sum(comb(m, i) for i in range(max(pos, neg), m + 1)) / (2 ** m) if m else 1.0
    print(f"{name}: {n:4d} task  steered={100*sw:.1f}%  bare={100*bw:.1f}%  gap={100*(sw-bw):+.2f}pp  "
          f"{pos}胜/{n-pos-neg}平/{neg}负  p={min(p,1):.1e}")
