"""step-16/17 思考膨胀的真实轨迹标本(4B 纯 OPD 双臂;供周报直观例)。"""
import glob, json, os, re
from collections import Counter

BASE = "/data3/wushang/workspace/TRACE-OPD/trace_opd/runs"

for run in ("trace_opd_4b_rkl", "trace_opd_4b_sp2"):
    rd = os.path.join(BASE, run, "rollouts")
    if not os.path.isdir(rd):
        print(run, "无 rollouts;有:", os.listdir(os.path.join(BASE, run))[:8]
              if os.path.isdir(os.path.join(BASE, run)) else "目录不存在")
        continue
    steps = sorted(os.listdir(rd))
    print(f"\n===== {run}: {len(steps)} dirs, 最后: {steps[-4:]}")
    for sd in steps[-2:]:
        files = glob.glob(os.path.join(rd, sd, "*.json"))
        rows = []
        for f in files:
            try:
                d = json.load(open(f))
            except Exception:
                continue
            st = (d.get("trajectory") or {}).get("steps") or []
            th = sum(len(s.get("thought") or "") for s in st)
            ac = sum(len(s.get("action") or "") for s in st)
            rows.append((th + ac, th, ac, len(st), bool(d.get("won")), f,
                         str(d.get("task_id", ""))))
        rows.sort(reverse=True)
        if not rows:
            print(f"  {sd}: 空")
            continue
        tot = [r[0] for r in rows]
        print(f"  {sd}: n={len(rows)} 思考+动作字符 p50={tot[len(tot)//2]} "
              f"p90={tot[int(len(tot)*0.1)]} max={tot[0]}")
        top = rows[0]
        d = json.load(open(top[5]))
        st = (d.get("trajectory") or {}).get("steps") or []
        lens = [len(s.get("thought") or "") for s in st]
        print(f"  ★ 最长轨迹 {os.path.basename(top[5])} task={top[6][:60]} won={top[4]}")
        print(f"    步数={len(st)} 思考总字符={top[1]} 动作字符={top[2]}")
        print(f"    每步思考长度={lens}")
        if not st:
            continue
        big_i = max(range(len(st)), key=lambda i: lens[i])
        big = st[big_i].get("thought") or ""
        sents = [x.strip() for x in re.split(r"[。.!?\n]", big) if len(x.strip()) > 12]
        cc = Counter(sents)
        print(f"    最长单步思考: 第{big_i+1}步 {lens[big_i]} 字符;高频句 top3:")
        for s, c in cc.most_common(3):
            print(f"      ×{c}  {s[:120]}")
        seen = set(); rep = 0
        for s in st:
            t = (s.get("thought") or "").strip()
            if t in seen:
                rep += 1
            if t:
                seen.add(t)
        print(f"    步间逐字重复思考步数={rep}/{len(st)}")
        print("    ---- 摘录:最长单步思考开头 300 字 ----")
        print("    " + big[:300].replace("\n", " ⏎ "))
        print("    ---- 同一步中段 200 字 ----")
        mid = len(big) // 2
        print("    " + big[mid:mid + 200].replace("\n", " ⏎ "))
