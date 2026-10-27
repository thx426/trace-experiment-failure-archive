#!/usr/bin/env python3
"""对比两轮 val 的代码执行超时率与响应长度分布(诊断 temp>0 是否致长尾)。"""
import json, glob, os

def scan(d, label, limit=1600):
    tot = code = timeout = err = 0
    lens = []
    for f in glob.glob(os.path.join(d, "*.json"))[:limit]:
        if os.path.getsize(f) == 0:
            continue
        try:
            r = json.load(open(f))
        except Exception:
            continue
        tot += 1
        lens.append(r.get("response_token_len") or 0)
        for s in (r.get("trajectory") or {}).get("steps") or []:
            if "run_python" in (s.get("action_raw") or ""):
                code += 1
                obs = s.get("observation") or ""
                if "timeout after" in obs:
                    timeout += 1
                elif "[exit" in obs:
                    err += 1
    lens.sort()
    n = max(len(lens), 1)
    print(f"{label}")
    print(f"  n={tot}  代码调用={code}  超时={timeout} ({100*timeout/max(code,1):.1f}%)  "
          f"报错={err} ({100*err/max(code,1):.1f}%)")
    print(f"  响应 token: 中位={lens[n//2]} p90={lens[int(n*.9)]} p99={lens[int(n*.99)]} max={max(lens)}")
    print(f"  超长(>8000 tok)占比: {100*sum(1 for x in lens if x > 8000)/n:.1f}%")

B = "/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM/runs"
scan(f"{B}/classic_math7b_pgs/rollouts/step_00000_val", "【现在】avg@3 temp0.6(进行中,有幸存者偏差)")
print()
scan(f"{B}/classic_math7b_pgs_smoke/rollouts/step_00000_val", "【冒烟】greedy@1(已完成,无偏)")
