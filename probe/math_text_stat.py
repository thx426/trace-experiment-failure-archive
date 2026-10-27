#!/usr/bin/env python3
"""Math 文本 TIR 探针读数:逐 bench EM / void-turn / 代码使用 / 截断。
用法: math_text_stat.py <out_dir 含 episodes.jsonl>"""
import json, sys, collections

path = sys.argv[1].rstrip("/") + "/episodes.jsonl"
rows = [json.loads(l) for l in open(path)]
print(f"episodes = {len(rows)}")

by_bench = collections.defaultdict(list)
for r in rows:
    by_bench[r["task_id"].split("/")[0]].append(r)

tot_w = tot_n = 0
print(f"\n{'bench':>10} {'EM':>10} {'截断':>4} {'void':>5} {'code轮/ep':>9} {'用码ep':>7} {'码错率':>7} {'轮数中位':>8}")
for b in sorted(by_bench):
    rs = by_bench[b]
    att = [r for r in rs if r["outcome"] != "ENV_ERROR"]
    won = sum(1 for r in att if r["won"])
    tot_w += won; tot_n += len(att)
    trunc = sum(1 for r in att if r.get("truncated"))
    voids = sum(r.get("invalid_format") or 0 for r in att)
    code_runs, code_eps, code_fail = [], 0, [0, 0]
    turns = []
    for r in att:
        steps = (r.get("trajectory") or {}).get("steps") or []
        cr = sum(1 for s in steps if '"run_python"' in (s.get("action_raw") or ""))
        for s in steps:
            if '"run_python"' in (s.get("action_raw") or ""):
                code_fail[1] += 1
                if not s.get("valid", True):
                    code_fail[0] += 1
        code_runs.append(cr)
        code_eps += 1 if cr else 0
        turns.append(r.get("assistant_turns") or 0)
    code_runs.sort(); turns.sort()
    n = max(len(att), 1)
    err_rate = f"{100*code_fail[0]/max(code_fail[1],1):.0f}%" if code_fail[1] else "—"
    print(f"{b:>10} {won:>3}/{len(att):<3}={100*won/n:>4.0f}% {trunc:>4} {voids:>5} "
          f"{sum(code_runs)/n:>9.1f} {code_eps:>4}/{len(att):<3} {err_rate:>7} {turns[len(turns)//2] if turns else 0:>8}")

envE = sum(1 for r in rows if r["outcome"] == "ENV_ERROR")
print(f"\n总计 EM = {tot_w}/{tot_n} = {100*tot_w/max(tot_n,1):.1f}%   ENV_ERROR={envE}")

# 输局的死法分布
lost = [r for r in rows if r["outcome"] != "ENV_ERROR" and not r["won"]]
death = collections.Counter()
for r in lost:
    if r.get("truncated"):
        death["truncated"] += 1
    else:
        steps = (r.get("trajectory") or {}).get("steps") or []
        subs = [s for s in steps if '"submit"' in (s.get("action_raw") or "")]
        death["答案错" if subs else "未交答案(耗尽轮次)"] += 1
print("输局死法:", dict(death))
