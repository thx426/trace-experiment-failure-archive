#!/usr/bin/env python3
"""avg@3 口径的 val 读数:按 task 聚合 3 次采样,给 mean@3 / best@3 / worst@3
+ 一致性画像(3 次全对 / 全错 / 不一致的占比 —— 这是 greedy 单次给不出的信息)。
用法: val_avg3.py <rollouts/step_XXXXX_val> [--by-bench] [--since HH:MM]

★ --since 是必需的读数纪律(2026-08-14 实证):resume 会把新一轮 val 追加
进**同一个 step_XXXXX_val 目录**,口径切换(greedy@1 → avg@3)后该目录会
同时含两批数据(实测 SQA step_00050_val 有 10240 = 4096 旧 greedy + 6144
新 avg@3)。不按时间切分会把两种口径混算,结论必错。"""
import json, glob, os, sys, collections, datetime

d = sys.argv[1]
by_bench = "--by-bench" in sys.argv
since = None
if "--since" in sys.argv:
    hhmm = sys.argv[sys.argv.index("--since") + 1]
    today = datetime.date.today()
    h, m = hhmm.split(":")
    since = datetime.datetime.combine(today, datetime.time(int(h), int(m))).timestamp()
recs, dropped = [], 0
for f in glob.glob(os.path.join(d, "*.json")):
    if os.path.getsize(f) == 0:
        continue
    if since is not None and os.path.getmtime(f) < since:
        dropped += 1
        continue
    try:
        recs.append(json.load(open(f)))
    except Exception:
        pass
if since is not None:
    print(f"[--since {hhmm}] 排除 {dropped} 个旧口径文件")

by_task = collections.defaultdict(list)
task_bench = {}
for r in recs:
    t = r.get("task_id")
    by_task[t].append(1 if r.get("won") else 0)
    # bench 归属:优先 task_type(各域 builder 写入);SearchQA 的 task_id 形如
    # `2wikimultihopqa_dev_10006`(无 "/"),Math 形如 `aime24/10`。两种都兼容。
    if t not in task_bench:
        tt = r.get("task_type")
        if tt and tt not in ("searchqa", "math"):
            task_bench[t] = str(tt)
        elif "/" in str(t):
            task_bench[t] = str(t).split("/")[0]
        else:
            task_bench[t] = str(t).rsplit("_", 2)[0] if str(t).count("_") >= 2 else str(t)

full = {t: v for t, v in by_task.items() if len(v) >= 3}
part = len(by_task) - len(full)
print(f"读入 {len(recs)} 行 / {len(by_task)} task(副本齐 3 的 {len(full)},未齐 {part} —— 未齐者已排除)")
if not full:
    sys.exit(0)

def stats(tasks):
    n = len(tasks)
    if not n:
        return None
    mean = sum(sum(v[:3]) / 3 for v in tasks.values()) / n
    best = sum(1 for v in tasks.values() if max(v[:3])) / n
    worst = sum(1 for v in tasks.values() if min(v[:3])) / n
    allc = sum(1 for v in tasks.values() if sum(v[:3]) == 3)
    alln = sum(1 for v in tasks.values() if sum(v[:3]) == 0)
    return n, mean, best, worst, allc, alln

n, mean, best, worst, allc, alln = stats(full)
print(f"\n=== 总体(n={n} task × 3 次)===")
print(f"  mean@3  = {100*mean:.2f}%   ← 主指标(与 greedy@1 口径不可直比)")
print(f"  best@3  = {100*best:.2f}%   (至少一次答对 ≈ pass@3 上界)")
print(f"  worst@3 = {100*worst:.2f}%  (三次全对 ≈ 稳定掌握下界)")
print(f"  一致性:三次全对 {allc} ({100*allc/n:.1f}%) | 全错 {alln} ({100*alln/n:.1f}%) | "
      f"不一致 {n-allc-alln} ({100*(n-allc-alln)/n:.1f}%)")
print(f"  ⇒ best-worst 跨度 {100*(best-worst):.1f}pp = 采样不确定带宽")

if by_bench:
    groups = collections.defaultdict(dict)
    for t, v in full.items():
        groups[task_bench.get(t, "?")][t] = v
    print(f"\n{'bench':>12} {'mean@3':>8} {'best@3':>8} {'worst@3':>8} {'n':>5}")
    for b in sorted(groups):
        s = stats(groups[b])
        print(f"{b:>12} {100*s[1]:>7.1f}% {100*s[2]:>7.1f}% {100*s[3]:>7.1f}% {s[0]:>5}")
