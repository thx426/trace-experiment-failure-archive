#!/usr/bin/env python3
"""数学 doc 的答案泄漏量化:答案入 doc 率 + steered/bare gap 按泄漏分层。
用法: math_doc_leak.py <run_dir>(含 research/ 与 rollouts/step_00002)"""
import json, glob, sys, os, collections, re

run = sys.argv[1].rstrip("/")

# 1) task → doc 文本(取每 task 最新版本)
docs = {}
for f in sorted(glob.glob(os.path.join(run, "research", "*", "*", "doc.md"))):
    task_dir = f.split("/research/")[1].split("/")[0]
    docs[task_dir] = open(f, encoding="utf-8", errors="replace").read()

# 2) step2 rollout:task_id/ground_truth/won/steer_arm
recs = []
for f in glob.glob(os.path.join(run, "rollouts", "step_00002", "*.json")):
    if os.path.getsize(f) == 0:
        continue
    try:
        recs.append(json.load(open(f)))
    except Exception:
        pass
print(f"docs={len(docs)}  step2 rollouts={len(recs)}")

def norm(s):
    s = str(s).strip().replace(" ", "").replace("$", "").replace("\\!", "")
    s = re.sub(r"\\boxed\{(.*)\}", r"\1", s)
    return s

def answer_in_doc(gt, doc):
    """粗判:归一化后的 gt 是否以子串出现在 doc 中(长度<2 的 gt 不判,太易误中)。"""
    g = norm(gt)
    if len(g) < 2:
        return None
    d = doc.replace(" ", "").replace("$", "")
    return g in d

# rollout 记录里 ground_truth 在 extra? 训练 rollout 顶层无 extra_info——从
# trajectory 无法取 gt。改从 parquet 拿。
import pandas as pd
gt_map = {}
for pq in ["/fs/fast/u2021000902/wushang/data/classic_math/train.parquet"]:
    df = pd.read_parquet(pq)
    for _, row in df.iterrows():
        ei = dict(row["extra_info"])
        gt_map[str(ei["task_id"])] = str(ei["ground_truth"])

by_task = collections.defaultdict(list)
for r in recs:
    by_task[r.get("task_id")].append(r)

leak_y, leak_n, leak_u = [], [], []
for tid, rs in by_task.items():
    if len(rs) != 8:
        continue
    # research 目录名 = task_id 的安全化(dapo/xxx → dapo__xxx--hash);前缀匹配
    safe = tid.replace("/", "__")
    doc = ""
    for k, v in docs.items():
        if k.startswith(safe + "--") or k == safe:
            doc = v; break
    gt = gt_map.get(tid, "")
    st = [r for r in rs if (r.get("steering") or {}).get("steer_arm") == 1]
    ba = [r for r in rs if (r.get("steering") or {}).get("steer_arm") == 0]
    if not st or not ba or not doc:
        continue
    sw = sum(1 for r in st if r.get("won")) / len(st)
    bw = sum(1 for r in ba if r.get("won")) / len(ba)
    a = answer_in_doc(gt, doc)
    cell = {"tid": tid, "sw": sw, "bw": bw}
    (leak_y if a else (leak_n if a is False else leak_u)).append(cell)

def report(name, cells):
    if not cells:
        print(f"{name}: 0 task")
        return
    n = len(cells)
    sw = sum(c["sw"] for c in cells) / n
    bw = sum(c["bw"] for c in cells) / n
    pos = sum(1 for c in cells if c["sw"] > c["bw"])
    neg = sum(1 for c in cells if c["sw"] < c["bw"])
    print(f"{name}: {n} task  steered={100*sw:.1f}%  bare={100*bw:.1f}%  "
          f"gap={100*(sw-bw):+.1f}pp  配对 {pos}胜/{n-pos-neg}平/{neg}负")

print("\n=== 答案入 doc 分层(step2,副本齐全 task)===")
report("答案在 doc  ", leak_y)
report("答案不在 doc", leak_n)
report("gt 太短不判 ", leak_u)
tot = len(leak_y) + len(leak_n)
if tot:
    print(f"\n答案入 doc 率 = {len(leak_y)}/{tot} = {100*len(leak_y)/tot:.0f}%(可判 task 中)")
