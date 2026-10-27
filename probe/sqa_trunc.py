#!/usr/bin/env python3
"""SearchQA 截断归因:撞的是 env_steps(4)、assistant_turns(6) 还是 token 预算?
并给出"截断的都是些什么轨迹"。用法: sqa_trunc.py <val_dir>"""
import json, glob, sys, os, collections

d = sys.argv[1]
recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "*.json")))]
T = [r for r in recs if r.get("truncated")]
N = [r for r in recs if not r.get("truncated")]
print(f"n={len(recs)}  截断={len(T)} ({100*len(T)/len(recs):.1f}%)  未截断={len(N)}")

STEP_CAP, TURN_CAP, TOK_CAP = 4, 6, 8192
hit_step = sum(1 for r in T if r.get("env_steps", 0) >= STEP_CAP)
hit_turn = sum(1 for r in T if r.get("assistant_turns", 0) >= TURN_CAP)
hit_tok = sum(1 for r in T if r.get("response_token_len", 0) >= TOK_CAP * 0.97)
print(f"\n=== 撞墙归因(截断样本 n={len(T)})===")
print(f"  撞 env_steps(>={STEP_CAP})        : {hit_step:>5} ({100*hit_step/max(len(T),1):.1f}%)")
print(f"  撞 assistant_turns(>={TURN_CAP})  : {hit_turn:>5} ({100*hit_turn/max(len(T),1):.1f}%)")
print(f"  撞 token 预算(>=97% of {TOK_CAP}) : {hit_tok:>5} ({100*hit_tok/max(len(T),1):.1f}%)")


def q(xs, p):
    xs = sorted(xs)
    return xs[min(int(len(xs) * p), len(xs) - 1)] if xs else 0


for tag, sub in (("截断组", T), ("未截断组", N), ("赢的组", [r for r in recs if r.get("won")])):
    if not sub:
        continue
    st = [r.get("env_steps", 0) for r in sub]
    tu = [r.get("assistant_turns", 0) for r in sub]
    tk = [r.get("response_token_len", 0) for r in sub]
    inv = [r.get("invalid_format", 0) for r in sub]
    print(f"\n--- {tag} n={len(sub)}")
    print(f"    env_steps  med={q(st,.5)} p95={q(st,.95)} max={max(st)}")
    print(f"    turns      med={q(tu,.5)} p95={q(tu,.95)} max={max(tu)}")
    print(f"    resp_tok   med={q(tk,.5)} p95={q(tk,.95)} max={max(tk)}")
    print(f"    invalid_format 均值={sum(inv)/len(sub):.2f}  有违规的占比={100*sum(1 for x in inv if x)/len(sub):.1f}%")
    print(f"    outcome: {dict(collections.Counter(r.get('outcome') for r in sub))}")

print(f"\n=== 截断是否吞掉了本可答对的轨迹 ===")
w_all = sum(1 for r in recs if r.get("won"))
w_nt = sum(1 for r in N if r.get("won"))
w_t = sum(1 for r in T if r.get("won"))
print(f"  名义 EM      {w_all}/{len(recs)} = {100*w_all/len(recs):.1f}%")
print(f"  未截断子集   {w_nt}/{len(N)} = {100*w_nt/max(len(N),1):.1f}%")
print(f"  『赢&截断』  {w_t}  (>0 说明截断前已 submit 且答对)")

# 截断轨迹的最后一步在干嘛
last_acts = collections.Counter()
for r in T[:400]:
    steps = (r.get("trajectory") or {}).get("steps") or []
    if steps:
        a = str(steps[-1].get("action_norm") or steps[-1].get("action_raw") or "")[:40]
        last_acts[a.split("(")[0][:24] or "(空)"] += 1
print(f"\n=== 截断轨迹最后一个动作(前 400 条)===")
for k, v in last_acts.most_common(8):
    print(f"  {v:>5}  {k!r}")
