#!/usr/bin/env python3
"""截断诊断:瓶颈到底是 response_length / assistant_turns / env_steps 哪一个。
用法: trunc_diag.py <rollout_dir>"""
import json, glob, sys, os, statistics as st

d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, "*.json")))
recs = []
for f in files:
    r = json.load(open(f))
    r["_base"] = os.path.basename(f)
    recs.append(r)
print(f"n={len(recs)}\n")


def q(xs, p):
    xs = sorted(xs)
    if not xs: return 0
    return xs[min(int(len(xs) * p), len(xs) - 1)]


def dump(tag, sub):
    if not sub: return
    tl = [r.get("response_token_len", 0) for r in sub]
    tu = [r.get("assistant_turns", 0) for r in sub]
    es = [r.get("env_steps", 0) for r in sub]
    per = [r.get("response_token_len", 0) / max(r.get("assistant_turns", 1), 1) for r in sub]
    inv = sum(1 for r in sub if r.get("invalid_format"))
    print(f"--- {tag}  n={len(sub)}")
    for nm, xs in (("resp_tok", tl), ("turns", tu), ("env_steps", es), ("tok/turn", per)):
        print("    %-9s min=%6.0f p25=%6.0f med=%6.0f p75=%6.0f p95=%6.0f max=%6.0f"
              % (nm, min(xs), q(xs, .25), q(xs, .5), q(xs, .75), q(xs, .95), max(xs)))
    print(f"    invalid_format={inv}  outcome分布={dict((o, sum(1 for r in sub if r.get('outcome')==o)) for o in set(r.get('outcome') for r in sub))}")
    print()


T = [r for r in recs if r.get("truncated")]
N = [r for r in recs if not r.get("truncated")]
W = [r for r in recs if r.get("won")]
dump("全部", recs)
dump("truncated=True", T)
dump("truncated=False", N)
dump("won=True", W)

# 撞哪堵墙
RESP_CAP, TURN_CAP, STEP_CAP = 28672, 60, 50
print("=== 截断样本撞墙归因 ===")
hit_resp = sum(1 for r in T if r.get("response_token_len", 0) >= RESP_CAP * 0.97)
hit_turn = sum(1 for r in T if r.get("assistant_turns", 0) >= TURN_CAP)
hit_step = sum(1 for r in T if r.get("env_steps", 0) >= STEP_CAP)
print(f"  n_truncated={len(T)}")
print(f"  撞 response_length(>=97% of {RESP_CAP}): {hit_resp}  ({100*hit_resp/max(len(T),1):.1f}%)")
print(f"  撞 assistant_turns(>={TURN_CAP}):        {hit_turn}  ({100*hit_turn/max(len(T),1):.1f}%)")
print(f"  撞 env_steps(>={STEP_CAP}):              {hit_step}  ({100*hit_step/max(len(T),1):.1f}%)")
print()

# 真实能力口径:未截断子集里的胜率
print("=== 真实能力口径(记忆 truncation-masks-capability)===")
for tag, sub in (("seen", [r for r in recs if r["_base"].startswith("valid_seen__")]),
                 ("unseen", [r for r in recs if r["_base"].startswith("valid_unseen__")])):
    nt = [r for r in sub if not r.get("truncated")]
    w_all = sum(1 for r in sub if r.get("won"))
    w_nt = sum(1 for r in nt if r.get("won"))
    w_t = sum(1 for r in sub if r.get("won") and r.get("truncated"))
    print(f"  {tag}: 名义 {w_all}/{len(sub)}={100*w_all/max(len(sub),1):.1f}%  |  "
          f"未截断子集 {w_nt}/{len(nt)}={100*w_nt/max(len(nt),1):.1f}%  |  "
          f"『赢&截断』={w_t}  |  截断率={100*(len(sub)-len(nt))/max(len(sub),1):.1f}%")
print()

# 若把 max_tokens_per_turn 压到 512/384,50 轮总量估算
print("=== 假想预算(按实测 tok/turn 分布外推 50 轮总需求)===")
per_all = [r.get("response_token_len", 0) / max(r.get("assistant_turns", 1), 1) for r in recs]
for p, lab in ((.5, "med"), (.75, "p75"), (.95, "p95")):
    v = q(per_all, p)
    print(f"  {lab} tok/turn={v:6.0f}  → 50轮需 {v*50:7.0f} tok  (预算 {RESP_CAP})"
          f"  {'✗超' if v*50 > RESP_CAP else '✓够'}")
