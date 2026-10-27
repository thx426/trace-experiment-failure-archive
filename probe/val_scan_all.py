#!/usr/bin/env python3
import os, re, sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

ROOT = "/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM/runs"
RUNS = ["classic_alf7b_pgs","classic_alf7b_grpo","classic_alf3b_pgs","classic_alf3b_grpo",
        "classic_ws3b_pgs","classic_ws3b_grpo","classic_sqa7b_pgs","classic_sqa7b_grpo",
        "classic_math7b_pgs","classic_ssb3b_pgs"]

WON_RE = re.compile(rb'"won":\s*(true|false)')
SPLIT_RE = re.compile(rb'"split":\s*"([^"]*)"')
TASK_RE = re.compile(rb'"task_id":\s*("[^"]*"|\d+)')

def scan_dir(args):
    run, valdir = args
    step = int(valdir.split("step_")[1].split("_")[0])
    path = os.path.join(ROOT, run, "rollouts", valdir)
    agg = defaultdict(lambda: [0, 0, set()])
    try: fns = os.listdir(path)
    except OSError: return run, step, {}
    for fn in fns:
        if not fn.endswith(".json"): continue
        fp = os.path.join(path, fn)
        try:
            if os.path.getsize(fp) == 0: continue
            data = open(fp, "rb").read()
        except OSError: continue
        w = WON_RE.search(data)
        if not w: continue
        s = SPLIT_RE.search(data)
        t = TASK_RE.search(data)
        split = s.group(1).decode() if s else "?"
        tid = t.group(1).decode() if t else fn
        a = agg[split]; a[0] += 1; a[1] += int(w.group(1) == b"true"); a[2].add(tid)
    return run, step, {k: (v[0], v[1], len(v[2])) for k, v in agg.items()}

jobs = []
for run in RUNS:
    ro = os.path.join(ROOT, run, "rollouts")
    if not os.path.isdir(ro): continue
    for d in sorted(os.listdir(ro)):
        if d.endswith("_val"):
            jobs.append((run, d))

print("run,step,split,n_traj,n_won,rate_pct,n_task", flush=True)
with ProcessPoolExecutor(max_workers=16) as ex:
    for run, step, agg in ex.map(scan_dir, jobs):
        for split, (n, w, nt) in sorted(agg.items()):
            print(f"{run},{step},{split},{n},{w},{w/n*100:.2f},{nt}", flush=True)
