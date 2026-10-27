#!/usr/bin/env python3
"""跨步池化的 steering 配对分析(通用版,SQA/Math 都用)。
用法: pooled_steer.py <run_dir> --since "YYYY-MM-DD HH:MM:SS" [--min-step 2]

只统计:①该时间点之后落盘的文件(resume 会往同一 step 目录追加,必须切分);
②副本齐 8 且 4+4 分臂的 task(半写状态与残缺组会污染配对)。
"""
import json, glob, os, sys, collections, datetime
from math import comb

run = sys.argv[1].rstrip("/")
since = None
if "--since" in sys.argv:
    since = datetime.datetime.strptime(sys.argv[sys.argv.index("--since") + 1],
                                       "%Y-%m-%d %H:%M:%S").timestamp()
min_step = int(sys.argv[sys.argv.index("--min-step") + 1]) if "--min-step" in sys.argv else 2

pos = neg = tie = 0
gaps, steps_used = [], []
st_w = st_n = ba_w = ba_n = 0
iv_lens, doc_words = [], []

for d in sorted(glob.glob(os.path.join(run, "rollouts", "step_*"))):
    base = os.path.basename(d)
    if "val" in base:
        continue
    try:
        n = int(base.split("_")[1])
    except Exception:
        continue
    if n < min_step:
        continue
    by = collections.defaultdict(list)
    for f in glob.glob(os.path.join(d, "*.json")):
        if os.path.getsize(f) == 0:
            continue
        if since and os.path.getmtime(f) < since:
            continue
        try:
            by[json.load(open(f))["task_id"]].append(json.load(open(f)))
        except Exception:
            pass
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
        st_w += sw; st_n += 4; ba_w += bw; ba_n += 4
        if sw > bw: pos += 1
        elif sw < bw: neg += 1
        else: tie += 1
        gaps.append((sw - bw) / 4)
        for x in s:
            iv_lens.append(len((x.get("steering") or {}).get("steer_positions") or []))
        for x in rs:
            w = (x.get("research") or {}).get("research_doc_words")
            if isinstance(w, (int, float)):
                doc_words.append(w)
    if full:
        steps_used.append((n, full))

if not steps_used:
    print("无可用数据(检查 --since 与 min-step)"); sys.exit(0)
m = pos + neg
p = 2 * sum(comb(m, i) for i in range(max(pos, neg), m + 1)) / (2 ** m) if m else 1.0
print(f"池化步: {[s for s, _ in steps_used]}(各步合格 task: {[c for _, c in steps_used]})")
print(f"steered {st_w}/{st_n} = {100*st_w/max(st_n,1):.1f}%   "
      f"bare {ba_w}/{ba_n} = {100*ba_w/max(ba_n,1):.1f}%   "
      f"gap = {100*(st_w/max(st_n,1) - ba_w/max(ba_n,1)):+.2f}pp")
print(f"逐 task 配对: {pos} 胜 / {tie} 平 / {neg} 负;非平局胜率 {100*pos/max(m,1):.1f}%,"
      f"符号检验 p = {min(p,1.0):.2e}")
print(f"平均组内 gap = {100*sum(gaps)/max(len(gaps),1):+.2f}pp")
if iv_lens:
    iv_lens.sort()
    print(f"干预 token: 中位={iv_lens[len(iv_lens)//2]}  p90={iv_lens[int(len(iv_lens)*.9)]}  "
          f"为0占比={100*sum(1 for x in iv_lens if x == 0)/len(iv_lens):.1f}%")
if doc_words:
    doc_words.sort()
    print(f"doc 词数中位 = {doc_words[len(doc_words)//2]:.0f}")
