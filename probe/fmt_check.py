#!/usr/bin/env python3
"""v5 格式服从性专项:逐轮 valid 率 + 抽轨迹展示。用法: fmt_check.py <dir> [n]"""
import json, glob, sys, os, collections

d = sys.argv[1]
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
files = sorted(glob.glob(os.path.join(d, "*.json")))
recs = [json.load(open(f)) for f in files]
print(f"n={len(recs)}")

tot_turns = tot_valid = tot_acted = 0
inv_dist = collections.Counter()
deg = 0
for r in recs:
    st = (r.get("trajectory") or {}).get("steps") or []
    tot_turns += len(st)
    tot_valid += sum(1 for s in st if s.get("valid"))
    tot_acted += sum(1 for s in st if s.get("acted"))
    inv_dist[r.get("invalid_format", 0)] += 1
    if str(r.get("outcome", "")).startswith("DEGENERATE"):
        deg += 1

print(f"总轮数={tot_turns}  acted(有<action>块)={tot_acted} ({100*tot_acted/max(tot_turns,1):.1f}%)  valid(引擎接受)={tot_valid} ({100*tot_valid/max(tot_turns,1):.1f}%)")
print(f"DEGENERATE:format 条数={deg}/{len(recs)}")
print("每条轨迹 invalid_format 次数分布:", dict(sorted(inv_dist.items())[:8]))
print()

# 抽一条赢的 + 一条输的,展示逐轮
won = [r for r in recs if r.get("won")]
lost = [r for r in recs if not r.get("won")]
print(f"won={len(won)}  lost={len(lost)}")
for tag, pool in (("WON", won), ("LOST", lost)):
    for r in pool[:NS if tag == "WON" else 1]:
        st = (r.get("trajectory") or {}).get("steps") or []
        print("#" * 72)
        print(f"[{tag}] {r.get('task_id')}  turns={r.get('assistant_turns')} env_steps={r.get('env_steps')} invalid={r.get('invalid_format')}")
        show = st[:4] + (["..."] if len(st) > 8 else []) + st[-4:] if len(st) > 8 else st
        for s in show:
            if s == "...":
                print("   ...")
                continue
            th = (s.get("thought") or "").replace("\n", " ")[:120]
            print(f"  t{s.get('t'):>2} valid={'Y' if s.get('valid') else 'n'} act={s.get('action_norm') or '∅':<28} | {th}")
        print()
