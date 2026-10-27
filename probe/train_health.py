#!/usr/bin/env python3
"""训练侧健康指标时间序列(判"训练崩了"还是"评测抖了")。"""
import re, sys, glob

f = sys.argv[1] if len(sys.argv) > 1 else None
if not f:
    cands = glob.glob("/tmp/ray_math_pgs/ray/session_latest/logs/worker-*.out")
    f = max(cands, key=lambda p: __import__("os").path.getsize(p))
txt = open(f, errors="replace").read().replace("\r", "\n")
K = [
    ("won",   r"rollout_extra/won/mean:([0-9.]+)",        lambda x: f"{100*x:5.1f}"),
    ("ent",   r"actor/entropy:([0-9.]+)",                 lambda x: f"{x:6.3f}"),
    ("grad",  r"actor/grad_norm:([0-9.]+)",               lambda x: f"{x:7.3f}"),
    ("resp",  r"response_length/mean:([0-9.]+)",          lambda x: f"{x:6.0f}"),
    ("clip",  r"response_length/clip_ratio:([0-9.]+)",    lambda x: f"{x:6.3f}"),
    ("gate",  r"steer_gate_open_rate/mean:([0-9.]+)",     lambda x: f"{100*x:5.1f}"),
    ("theta", r"steer_entropy_theta/mean:([0-9.]+)",      lambda x: f"{x:5.2f}"),
    ("ivtok", r"steer_count/mean:([0-9.]+)",              lambda x: f"{x:5.0f}"),
    ("trunc", r"rollout_extra/truncated/mean:([0-9.]+)",  lambda x: f"{100*x:5.1f}"),
    ("alpha", r"actor/pgs_steered_alpha:([0-9.]+)",       lambda x: f"{x:5.3f}"),
]
print(" step " + " ".join(f"{n:>6}" for n, _, _ in K))
for m in re.finditer(r"step:(\d+) - (.*?)(?=\n)", txt):
    s, b = int(m.group(1)), m.group(2)
    if "actor/entropy" not in b:
        continue
    cells = []
    for _n, pat, fmt in K:
        mm = re.search(pat, b)
        cells.append(fmt(float(mm.group(1))) if mm else "     -")
    print(f"{s:>5} " + " ".join(cells))
