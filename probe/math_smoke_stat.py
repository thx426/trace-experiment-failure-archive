#!/usr/bin/env python3
"""Math 冒烟 rollout 分析:val 分桶 EM / TIR 行为 / 干预效果(训练步)。
用法: math_smoke_stat.py <rollouts/step_XXXXX 目录> [--val]
只统计非 0 字节文件;val 模式按 task_id 前缀分桶。"""
import json, glob, sys, os, collections

d = sys.argv[1]
is_val = "--val" in sys.argv
recs = []
for f in glob.glob(os.path.join(d, "*.json")):
    if os.path.getsize(f) == 0:
        continue
    try:
        recs.append(json.load(open(f)))
    except Exception:
        pass
print(f"读入 {len(recs)} 行(已排 0 字节)  目录={d}")
if not recs:
    sys.exit(0)

def code_stats(r):
    """(码轮数, 是否有 boxed 终答, void 轮数, 总 assistant 轮)"""
    steps = (r.get("trajectory") or {}).get("steps") or []
    code = sum(1 for s in steps if '"run_python"' in (s.get("action_raw") or ""))
    sub = sum(1 for s in steps if '"submit"' in (s.get("action_raw") or ""))
    return code, sub, r.get("invalid_format") or 0, r.get("assistant_turns") or 0

if is_val:
    by = collections.defaultdict(list)
    for r in recs:
        by[str(r.get("task_id", "?")).split("/")[0]].append(r)
    print(f"\n{'bench':>10} {'EM':>12} {'截断':>4} {'void':>5} {'码轮/ep':>8} {'用码率':>7} {'交答率':>7}")
    tw = tn = 0
    for b in sorted(by):
        rs = by[b]
        won = sum(1 for r in rs if r.get("won")); tw += won; tn += len(rs)
        tr = sum(1 for r in rs if r.get("truncated"))
        cs = [code_stats(r) for r in rs]
        voids = sum(c[2] for c in cs)
        n = len(rs)
        print(f"{b:>10} {won:>4}/{n:<4}={100*won/n:>4.0f}% {tr:>4} {voids:>5} "
              f"{sum(c[0] for c in cs)/n:>8.2f} {100*sum(1 for c in cs if c[0]>0)/n:>6.0f}% "
              f"{100*sum(1 for c in cs if c[1]>0)/n:>6.0f}%")
    print(f"\n总 EM = {tw}/{tn} = {100*tw/max(tn,1):.1f}%")
else:
    # 训练步:整体行为 + steering 配对(若有)
    won = sum(1 for r in recs if r.get("won"))
    tr = sum(1 for r in recs if r.get("truncated"))
    cs = [code_stats(r) for r in recs]
    n = len(recs)
    print(f"won={won}/{n}={100*won/n:.1f}%  截断={tr}  void轮总数={sum(c[2] for c in cs)}")
    print(f"码轮/ep={sum(c[0] for c in cs)/n:.2f}  用码 ep={100*sum(1 for c in cs if c[0]>0)/n:.0f}%  "
          f"交答 ep={100*sum(1 for c in cs if c[1]>0)/n:.0f}%  轮数中位={sorted(c[3] for c in cs)[n//2]}")
    # 死法
    death = collections.Counter()
    for r in recs:
        if r.get("won"):
            continue
        if r.get("truncated"):
            death["truncated"] += 1
        else:
            c = code_stats(r)
            death["答案错" if c[1] else "未交答案"] += 1
    print(f"输局死法: {dict(death)}")
    # steering 配对(steer_effect.py 的精简版,凑齐 8 副本的 task)
    by_task = collections.defaultdict(list)
    for r in recs:
        by_task[r.get("task_id")].append(r)
    full = {t: v for t, v in by_task.items() if len(v) == 8}
    st_w = st_n = ba_w = ba_n = 0
    paired = []
    for t, rs in full.items():
        s = [r for r in rs if (r.get("steering") or {}).get("steer_arm") == 1]
        b = [r for r in rs if (r.get("steering") or {}).get("steer_arm") == 0]
        if not s or not b:
            continue
        sw = sum(1 for r in s if r.get("won")); bw = sum(1 for r in b if r.get("won"))
        st_w += sw; st_n += len(s); ba_w += bw; ba_n += len(b)
        paired.append((sw/len(s)) - (bw/len(b)))
    if st_n:
        pos = sum(1 for x in paired if x > 0); neg = sum(1 for x in paired if x < 0)
        print(f"\nsteering: steered {st_w}/{st_n}={100*st_w/st_n:.1f}%  bare {ba_w}/{ba_n}={100*ba_w/ba_n:.1f}%  "
              f"gap={100*(st_w/st_n-ba_w/ba_n):+.1f}pp  配对 {pos}胜/{len(paired)-pos-neg}平/{neg}负")
        pos_lens = []
        for t, rs in full.items():
            for r in rs:
                if (r.get("steering") or {}).get("steer_arm") == 1:
                    pos_lens.append(len((r.get("steering") or {}).get("steer_positions") or []))
        if pos_lens:
            pos_lens.sort()
            print(f"干预 token:中位={pos_lens[len(pos_lens)//2]}  p90={pos_lens[int(len(pos_lens)*.9)]}  "
                  f"为0占比={100*sum(1 for x in pos_lens if x==0)/len(pos_lens):.0f}%")
    dw = [(r.get("research") or {}).get("research_doc_words") for r in recs]
    dw = sorted(x for x in dw if isinstance(x, (int, float)))
    if dw:
        print(f"doc 词数:中位={dw[len(dw)//2]:.0f}  n={len(dw)}")
