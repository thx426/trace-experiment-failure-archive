#!/usr/bin/env python3
"""渲染若干条 math 轨迹为可读文本(人工观察用)。
用法: math_traj_dump.py <rollouts/step_XXXXX> <N> [won|lost|steered|code]"""
import json, glob, sys, os

d, N = sys.argv[1], int(sys.argv[2])
want = sys.argv[3] if len(sys.argv) > 3 else ""
recs = []
for f in sorted(glob.glob(os.path.join(d, "*.json"))):
    if os.path.getsize(f) == 0:
        continue
    try:
        r = json.load(open(f)); r["_file"] = os.path.basename(f)
        recs.append(r)
    except Exception:
        pass

def has_code(r):
    return any('"run_python"' in (s.get("action_raw") or "")
               for s in (r.get("trajectory") or {}).get("steps") or [])

def keep(r):
    if want == "won":     return bool(r.get("won"))
    if want == "lost":    return not r.get("won")
    if want == "steered": return (r.get("steering") or {}).get("steer_arm") == 1
    if want == "code":    return has_code(r)
    return True

picked = [r for r in recs if keep(r)][:N]
for r in picked:
    st = r.get("steering") or {}
    print("═" * 78)
    print(f"file={r['_file']}  task={r.get('task_id')}  won={r.get('won')}  "
          f"outcome={r.get('outcome')}  turns={r.get('assistant_turns')}  "
          f"steer_arm={st.get('steer_arm')}  干预tok={len(st.get('steer_positions') or [])}")
    doc = (r.get("research") or {}).get("research_doc") or ""
    if doc and st.get("steer_arm") == 1:
        print(f"--- doc({len(doc.split())}词,前400字)---")
        print(doc[:400])
    steps = (r.get("trajectory") or {}).get("steps") or []
    for i, s in enumerate(steps):
        th = (s.get("thought") or "").strip()
        ar = s.get("action_raw") or ""
        fb = (s.get("feedback") or s.get("observation") or "").strip()
        print(f"--- turn {i+1} ---")
        if th: print(f"[思考] {th[:500]}")
        if ar:
            try:
                c = json.loads(ar)
                if c.get("name") == "run_python":
                    print(f"[代码]\n{c['arguments']['code'][:600]}")
                else:
                    print(f"[动作] {c.get('name')}({json.dumps(c.get('arguments'), ensure_ascii=False)[:200]})")
            except Exception:
                print(f"[动作raw] {ar[:200]}")
        if fb: print(f"[反馈] {fb[:400]}")
    print()
