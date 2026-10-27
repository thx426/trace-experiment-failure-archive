"""同任务 steered vs bare 轨迹并排看 —— doc 到底改变了什么行为。"""
from __future__ import annotations

import glob
import json
import statistics
import sys
from collections import defaultdict


def score_of(r):
    return float((r.get("trajectory") or {}).get("score") or 0.0)


def arm_of(r):
    return int((r.get("steering") or {}).get("steer_arm", -1))


def show(r, tag, maxstep):
    tj = r.get("trajectory") or {}
    print(f"  --- {tag} score={score_of(r):.3f} won={r.get('won')} "
          f"env_steps={r.get('env_steps')} invalid={r.get('invalid_format')} ---")
    for i, s in enumerate((tj.get("steps") or [])[:maxstep]):
        a = s.get("action_norm") or s.get("action_raw") or ""
        ob = (s.get("observation") or "").replace("\n", " ")[:100]
        th = (s.get("thought") or "").replace("\n", " ")[:110]
        print(f"    [{i:02d}] valid={str(s.get('valid'))[:5]:<5} A| {a[:60]}")
        if th:
            print(f"         R| {th}")
        print(f"         O| {ob}")


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "classic_ws3b_pgs"
    step = sys.argv[2] if len(sys.argv) > 2 else "step_00001"
    maxstep = int(sys.argv[3]) if len(sys.argv) > 3 else 6

    recs = []
    for f in sorted(glob.glob(f"runs/{run}/rollouts/{step}/*.json")):
        try:
            recs.append(json.loads(open(f, encoding="utf-8").read()))
        except Exception:  # noqa: BLE001
            pass
    by_task = defaultdict(lambda: defaultdict(list))
    for r in recs:
        if r.get("outcome") != "ENV_ERROR":
            by_task[r["task_id"]][arm_of(r)].append(r)

    paired = {t: a for t, a in by_task.items() if a.get(0) and a.get(1)}
    # 挑 ΔScore 最大与最小的各一个任务
    deltas = []
    for t, a in paired.items():
        d = (statistics.mean(score_of(r) for r in a[1])
             - statistics.mean(score_of(r) for r in a[0]))
        deltas.append((d, t))
    deltas.sort()
    picks = [("★ steered 最占优", deltas[-1]), ("✗ steered 最吃亏", deltas[0])]

    for label, (d, t) in picks:
        a = paired[t]
        res = (a[1][0].get("research") or {})
        print(f"\n{'=' * 74}\n{label}  task={t}  ΔScore={d*100:+.1f}")
        print(f"  researcher: 实验 {res.get('research_experiments')} 次 / "
              f"episodes {res.get('research_episodes')} / "
              f"自评成功 {res.get('research_final_success')} / "
              f"doc {res.get('research_doc_words')} 词")
        best1 = max(a[1], key=score_of)
        best0 = max(a[0], key=score_of)
        show(best1, "steered(带 doc)", maxstep)
        show(best0, "bare(无 doc)", maxstep)


if __name__ == "__main__":
    main()
