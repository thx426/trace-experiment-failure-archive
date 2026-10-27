#!/usr/bin/env python3
"""分布干预(steering)效果:只统计**副本已落齐 8 条**的 task,避免半写状态污染。
用法: steer_effect.py <rollouts/step_XXXXX>"""
import json, glob, sys, os, collections

d = sys.argv[1]
recs = []
for f in glob.glob(os.path.join(d, "*.json")):
    try:
        recs.append(json.load(open(f)))
    except Exception:
        pass
by_task = collections.defaultdict(list)
for r in recs:
    by_task[r.get("task_id")].append(r)
full = {t: v for t, v in by_task.items() if len(v) == 8}
print(f"落盘 {len(recs)} 行 / {len(by_task)} task;其中副本齐全(8)的 task = {len(full)}")
if not full:
    sys.exit(0)

st_won = st_n = ba_won = ba_n = 0
pos_lens, doc_words, paired = [], [], []
for t, rs in full.items():
    s = [r for r in rs if (r.get("steering") or {}).get("steer_arm") == 1]
    b = [r for r in rs if (r.get("steering") or {}).get("steer_arm") == 0]
    if not s or not b:
        continue
    sw = sum(1 for r in s if r.get("won")); bw = sum(1 for r in b if r.get("won"))
    st_won += sw; st_n += len(s); ba_won += bw; ba_n += len(b)
    paired.append((sw / len(s)) - (bw / len(b)))
    for r in s:
        pos_lens.append(len((r.get("steering") or {}).get("steer_positions") or []))
    for r in rs:
        w = (r.get("research") or {}).get("research_doc_words")
        if isinstance(w, (int, float)):
            doc_words.append(w)

print(f"\n=== steered vs bare(组内配对,同任务同权重)===")
print(f"  steered : {st_won}/{st_n} = {100*st_won/max(st_n,1):.1f}%")
print(f"  bare    : {ba_won}/{ba_n} = {100*ba_won/max(ba_n,1):.1f}%")
gap = st_won / max(st_n, 1) - ba_won / max(ba_n, 1)
print(f"  gap     : {gap*100:+.1f}pp")

if paired:
    pos = sum(1 for x in paired if x > 0); neg = sum(1 for x in paired if x < 0)
    tie = len(paired) - pos - neg
    print(f"\n=== 逐 task 配对方向(n={len(paired)})===")
    print(f"  steered 更好 {pos}  |  持平 {tie}  |  bare 更好 {neg}")
    if pos + neg:
        # 符号检验(双边)
        from math import comb
        k, m = max(pos, neg), pos + neg
        p = 2 * sum(comb(m, i) for i in range(k, m + 1)) / (2 ** m)
        print(f"  非平局中 steered 胜 {pos}/{pos+neg},符号检验 p≈{min(p,1.0):.4f}")

if pos_lens:
    pos_lens.sort()
    print(f"\n=== 干预强度(steered 行的被干预 token 数)===")
    print(f"  中位={pos_lens[len(pos_lens)//2]}  p90={pos_lens[int(len(pos_lens)*.9)]}  max={max(pos_lens)}  为0的占比={100*sum(1 for x in pos_lens if x==0)/len(pos_lens):.1f}%")
if doc_words:
    doc_words.sort()
    print(f"  doc 词数中位={doc_words[len(doc_words)//2]:.0f}")
