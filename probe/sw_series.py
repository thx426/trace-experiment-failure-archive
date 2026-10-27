#!/usr/bin/env python3
"""从 swanlab 拉关键指标时间序列(只读,绝不 sync)。
用法: sw_series.py <exp_name> [--tail N] [--every K]
输出:step / won / entropy / invalid_format / steer_count / theta / resp / val
"""
import sys, os
os.environ.setdefault("SWANLAB_API_KEY", "58KT7T652vKUKobNhHb50")
import swanlab

name = sys.argv[1]
tail = int(sys.argv[sys.argv.index("--tail") + 1]) if "--tail" in sys.argv else 40
every = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 1

api = swanlab.OpenApi()
exps = api.list_experiments(project="trace_opd")
rows = [e for e in (exps.data if hasattr(exps, "data") else exps)
        if name in str(e.get("name", ""))]
if not rows:
    print("未找到实验:", name); sys.exit(1)
e = sorted(rows, key=lambda r: str(r.get("createdAt", "")))[-1]
print(f"experiment: {e.get('name')} | state: {e.get('state')}")

SPEC = [
    ("won",    "rollout_extra/won/mean",              lambda v: f"{100*v:5.1f}"),
    ("ent",    "actor/entropy",                       lambda v: f"{v:6.3f}"),
    ("badfmt", "rollout_extra/invalid_format/mean",   lambda v: f"{100*v:6.2f}"),
    ("ivtok",  "rollout_extra/steer_count/mean",      lambda v: f"{v:6.1f}"),
    ("gate",   "rollout_extra/steer_gate_open_rate/mean", lambda v: f"{100*v:5.1f}"),
    ("theta",  "rollout_extra/steer_entropy_theta/mean",  lambda v: f"{v:5.2f}"),
    ("resp",   "response_length/mean",                lambda v: f"{v:6.0f}"),
    ("grad",   "actor/grad_norm",                     lambda v: f"{v:6.3f}"),
    ("trunc",  "rollout_extra/truncated/mean",        lambda v: f"{100*v:5.1f}"),
]
series = {}
for label, key, _ in SPEC:
    try:
        s = api.get_metrics(exp_id=e.get("cuid"), keys=[key])
        d = s.to_dict("records") if hasattr(s, "to_dict") else s
        series[label] = {int(r["step"]): r[key] for r in d if r.get(key) is not None}
    except Exception as ex:
        series[label] = {}
        print(f"  ! {key} 读取失败: {type(ex).__name__}: {str(ex)[:70]}")

# val 指标(键名可能带 mean@3 等后缀,先探测)
val_keys = []
try:
    all_keys = api.get_summary(exp_id=e.get("cuid"))
    kd = all_keys.to_dict() if hasattr(all_keys, "to_dict") else all_keys
    val_keys = [k for k in kd if "val" in str(k).lower()][:12]
except Exception as ex:
    print(f"  ! summary 读取失败: {type(ex).__name__}")
if val_keys:
    print("  val 相关键:", ", ".join(val_keys))

steps = sorted(set().union(*[set(v) for v in series.values() if v]) or {0})
steps = [s for s in steps if s % every == 0][-tail:]
print(" step " + " ".join(f"{l:>6}" for l, _, _ in SPEC))
for s in steps:
    cells = []
    for label, _, fmt in SPEC:
        v = series[label].get(s)
        cells.append(fmt(v) if v is not None else "     -")
    print(f"{s:>5} " + " ".join(cells))
