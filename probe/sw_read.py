#!/usr/bin/env python3
"""只读 swanlab 指标(绝不 sync——sync 会终结云端 run)。用法: sw_read.py <exp_name>"""
import sys, os
os.environ.setdefault("SWANLAB_API_KEY", "58KT7T652vKUKobNhHb50")
import swanlab

name = sys.argv[1]
api = swanlab.OpenApi()
projs = api.list_projects()
exps = api.list_experiments(project="trace_opd")
rows = [e for e in (exps.data if hasattr(exps, "data") else exps) if name in str(e.get("name", ""))]
if not rows:
    print("未找到实验:", name)
    sys.exit(1)
e = sorted(rows, key=lambda r: str(r.get("createdAt", "")))[-1]
print("experiment:", e.get("name"), "| cuid:", e.get("cuid"), "| state:", e.get("state"))

KEYS = ["actor/pg_loss", "actor/grad_norm", "actor/entropy", "critic/score/mean",
        "critic/rewards/mean", "critic/advantages/mean", "response_length/mean",
        "actor/pgs_steered_alpha"]
for k in KEYS:
    try:
        s = api.get_metrics(exp_id=e.get("cuid"), keys=[k])
        d = s.to_dict("records") if hasattr(s, "to_dict") else s
        vals = [(r.get("step"), r.get(k)) for r in d if r.get(k) is not None]
        print(f"  {k:<28} {vals[-4:] if vals else '(无)'}")
    except Exception as ex:
        print(f"  {k:<28} 读取失败: {type(ex).__name__}: {str(ex)[:80]}")
