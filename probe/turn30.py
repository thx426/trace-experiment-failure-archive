#!/usr/bin/env python3
"""30 轮硬上限的代价评估 + response_length 需求外推(基于 step0 val 真实 274 条)。"""
import json, glob, sys, os

d = sys.argv[1]
recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "*.json")))]
won = [r for r in recs if r.get("won")]

print("=== 30 轮硬上限对『已赢轨迹』的影响 ===")
print(f"  当前赢 {len(won)} 条,其 assistant_turns 分布:")
tw = sorted(r.get("assistant_turns", 0) for r in won)
print("   ", tw)
for cap in (20, 25, 30, 35, 40, 50):
    keep = sum(1 for t in tw if t <= cap)
    print(f"  cap={cap:3d} → 保住 {keep:2d}/{len(won)} 条赢轨迹 ({100*keep/len(won):.0f}%)")
print()

print("=== 赢轨迹里 env_steps(有效步)分布 ===")
ew = sorted(r.get("env_steps", 0) for r in won)
print("   ", ew)
print()

print("=== 30 轮口径下 response_length 需求外推 ===")
# 每条轨迹的实测 tok/turn,乘以 30
per = sorted(r.get("response_token_len", 0) / max(r.get("assistant_turns", 1), 1) for r in recs)
n = len(per)
for lab, idx in (("med", n // 2), ("p90", int(n * .90)), ("p99", int(n * .99)), ("max", n - 1)):
    v = per[idx]
    print(f"  {lab:>3} tok/turn = {v:6.1f}  → 30 轮需 {v*30:7.0f} tok")
# 直接看前 30 轮的实际累计(有 steps 明细的)
tot30 = []
for r in recs:
    st = (r.get("trajectory") or {}).get("steps") or []
    if len(st) >= 30:
        # 用整条的 tok/turn × 30 近似(steps 无逐轮 token 计数)
        tot30.append(r.get("response_token_len", 0) / max(r.get("assistant_turns", 1), 1) * 30)
tot30.sort()
if tot30:
    print(f"  实跑满≥30轮的 {len(tot30)} 条:30轮估算 p99={tot30[int(len(tot30)*.99)]:.0f} max={tot30[-1]:.0f}")
print()
print("=== 无效轮浪费统计(30 轮上限下的价值)===")
waste = [(r.get("assistant_turns", 0) - r.get("env_steps", 0)) for r in recs]
waste.sort()
print(f"  废轮数 med={waste[len(waste)//2]} p75={waste[int(len(waste)*.75)]} p95={waste[int(len(waste)*.95)]} max={waste[-1]}")
print(f"  废轮占比(总): {100*sum(waste)/sum(r.get('assistant_turns',0) for r in recs):.1f}%")
