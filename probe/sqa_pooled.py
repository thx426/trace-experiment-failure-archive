#!/usr/bin/env python3
"""跨步池化配对分析 + doc 泄漏率(SearchQA 正式跑)。"""
import json, glob, os, sys, collections
from math import comb

run = "/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM/runs/classic_sqa7b_pgs"
# 1) 跨步池化(干预步 >=2)
pos = neg = tie = 0
gaps = []
steps_used = []
for d in sorted(glob.glob(os.path.join(run, "rollouts", "step_*"))):
    base = os.path.basename(d)
    if "val" in base:
        continue
    n = int(base.split("_")[1])
    if n < 2:
        continue
    by = collections.defaultdict(list)
    for f in glob.glob(os.path.join(d, "*.json")):
        if os.path.getsize(f) == 0:
            continue
        try:
            r = json.load(open(f))
        except Exception:
            continue
        by[r["task_id"]].append(r)
    full = 0
    for t, rs in by.items():
        if len(rs) != 8:
            continue
        s = [x for x in rs if (x.get("steering") or {}).get("steer_arm") == 1]
        b = [x for x in rs if (x.get("steering") or {}).get("steer_arm") == 0]
        if len(s) != 4 or len(b) != 4:
            continue
        full += 1
        sw = sum(1 for x in s if x.get("won"))
        bw = sum(1 for x in b if x.get("won"))
        if sw > bw: pos += 1
        elif sw < bw: neg += 1
        else: tie += 1
        gaps.append((sw - bw) / 4)
    if full:
        steps_used.append((n, full))
m = pos + neg
p = 2 * sum(comb(m, i) for i in range(max(pos, neg), m + 1)) / (2 ** m) if m else 1.0
print(f"池化步: {[s for s,_ in steps_used]}(各 {steps_used[0][1] if steps_used else 0} task)")
print(f"配对 task 总数 = {pos+neg+tie}:steered 胜 {pos} / 平 {tie} / 负 {neg}")
print(f"非平局中胜率 {pos}/{m} = {100*pos/max(m,1):.1f}%,双边符号检验 p = {min(p,1.0):.2e}")
print(f"平均组内 gap = {100*sum(gaps)/max(len(gaps),1):+.2f}pp")

# 2) doc 泄漏率(golden answers 是否出现在 doc 里)
import pandas as pd, re
gt = {}
df = pd.read_parquet("/fs/fast/u2021000902/wushang/data/classic/classic_searchqa_data/train_trainer.parquet")
for _, row in df.iterrows():
    ei = dict(row["extra_info"])
    ga = ei.get("golden_answers")
    if ga is None: ga = []
    elif isinstance(ga, str): ga = [ga]
    else: ga = list(ga)
    gt[str(ei["task_id"])] = [str(x) for x in ga]
def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())
y = n = u = 0
docs = glob.glob(os.path.join(run, "research", "*", "*", "doc.md"))
for f in docs:
    tdir = f.split("/research/")[1].split("/")[0]
    tid = tdir.rsplit("--", 1)[0].replace("__", "/", 1)
    golden = gt.get(tid, [])
    ok_ans = [norm(a) for a in golden if len(norm(a)) >= 3]
    if not ok_ans:
        u += 1; continue
    d = norm(open(f, encoding="utf-8", errors="replace").read())
    if any(a in d for a in ok_ans): y += 1
    else: n += 1
print(f"\ndoc 总数={len(docs)}  答案入 doc: {y}  不在: {n}  太短不判: {u}  ⇒ 入doc率 {100*y/max(y+n,1):.0f}%")
