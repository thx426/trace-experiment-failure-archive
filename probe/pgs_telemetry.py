#!/usr/bin/env python3
"""从 launch log 提取 PGS 逐步机制遥测。用法: pgs_telemetry.py <launch.log>"""
import re, sys

txt = open(sys.argv[1], errors="replace").read().replace("\r", "\n")
keys = {
    "bare": r"won_bare_rate/mean:([0-9.]+)",
    "steer": r"won_steered_rate/mean:([0-9.]+)",
    "gap": r"steer_arm_gap/mean:([-0-9.e]+)",
    "gate": r"steer_gate_open_rate/mean:([0-9.]+)",
    "theta": r"steer_entropy_theta/mean:([0-9.]+)",
    "alpha": r"pgs_steered_alpha:([0-9.]+)",
    "rsucc": r"research_final_success/mean:([0-9.]+)",
    "dwords": r"research_doc_words/mean:([0-9.]+)",
    "won": r"rollout_extra/won/mean:([0-9.]+)",
    "ent": r"actor/entropy:([0-9.]+)",
    "count": r"steer_count/mean:([0-9.]+)",
}
hdr = ["step", "won%", "bare%", "steer%", "gap_pp", "gate%", "theta", "alpha", "rsucc%", "docw", "entropy", "steer_tok"]
print(("{:>5} " * len(hdr)).format(*hdr))
for m in re.finditer(r"step:(\d+) - (.*?)(?=\n)", txt):
    step, body = int(m.group(1)), m.group(2)
    if "won_bare_rate" not in body:
        continue
    v = {}
    for k, pat in keys.items():
        mm = re.search(pat, body)
        v[k] = float(mm.group(1)) if mm else float("nan")
    row = [step, 100*v["won"], 100*v["bare"], 100*v["steer"], 100*v["gap"],
           100*v["gate"], v["theta"], v["alpha"], 100*v["rsucc"], v["dwords"], v["ent"], v["count"]]
    print("{:>5} {:>5.1f} {:>5.1f} {:>5.1f} {:>+5.1f} {:>5.1f} {:>5.2f} {:>5.3f} {:>5.1f} {:>5.0f} {:>5.2f} {:>5.0f}".format(*row))
