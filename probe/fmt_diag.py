#!/usr/bin/env python3
"""格式违规诊断:模型到底输出了什么导致 invalid_format。
用法: fmt_diag.py <rollout_dir> [n_samples]"""
import json, glob, sys, os, collections

d = sys.argv[1]
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
files = sorted(glob.glob(os.path.join(d, "*.json")))
recs = [json.load(open(f)) for f in files]

r0 = recs[0]
tr = r0.get("trajectory")
print("trajectory 类型:", type(tr).__name__)
if isinstance(tr, dict):
    print("keys:", sorted(tr.keys()))
    for k, v in tr.items():
        print("  %-18s %s len=%s" % (k, type(v).__name__, len(v) if hasattr(v, "__len__") else "-"))
        if isinstance(v, list) and v:
            print("      [0] =", json.dumps(v[0], ensure_ascii=False)[:400])
print()


def turns_of(rec):
    """返回 [(role, text), ...]"""
    t = rec.get("trajectory")
    if isinstance(t, list):
        return [(x.get("role", "?"), str(x.get("content", ""))) if isinstance(x, dict) else ("?", str(x)) for x in t]
    if isinstance(t, dict):
        for key in ("messages", "turns", "steps", "history", "conversation"):
            v = t.get(key)
            if isinstance(v, list):
                out = []
                for x in v:
                    if isinstance(x, dict):
                        role = x.get("role") or x.get("type") or "?"
                        c = x.get("content")
                        if c is None:
                            c = json.dumps({k: vv for k, vv in x.items() if k not in ("role", "type")}, ensure_ascii=False)
                        out.append((role, str(c)))
                    else:
                        out.append(("?", str(x)))
                return out
    return []


deg = [r for r in recs if r.get("outcome") == "DEGENERATE:format"]
print(f"=== DEGENERATE:format {len(deg)} 条,展示 {min(NS,len(deg))} ===\n")
for r in deg[:NS]:
    print("#" * 72)
    print("task:", r.get("task_id"), "| turns:", r.get("assistant_turns"),
          "| env_steps:", r.get("env_steps"), "| trunc:", r.get("truncated"),
          "| resp_tok:", r.get("response_token_len"))
    tt = turns_of(r)
    if not tt:
        print("  !! turns_of 空,原始 trajectory:", json.dumps(r.get("trajectory"), ensure_ascii=False)[:1500])
    for role, txt in (tt[:3] + ([("...", "...")] if len(tt) > 7 else []) + tt[-4:]):
        print(f"  [{role}] {txt[:600]}")
    print()

# 也看一条"轮数打满但没赢"的普通 FAILED
fail60 = [r for r in recs if r.get("outcome") == "FAILED" and r.get("truncated") and r.get("assistant_turns", 0) >= 60]
print(f"=== FAILED & turns>=60 共 {len(fail60)} 条,展示 1 条尾段 ===\n")
for r in fail60[:1]:
    print("task:", r.get("task_id"), "| env_steps:", r.get("env_steps"))
    for role, txt in turns_of(r)[-8:]:
        print(f"  [{role}] {txt[:500]}")
