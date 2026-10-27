"""看某个任务在训练时的全部 rollout —— doc 里给了照抄序列,执行者照做了吗?"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter


def main():
    run = sys.argv[1]
    task = sys.argv[2]
    step = sys.argv[3] if len(sys.argv) > 3 else "step_00001"
    show = int(sys.argv[4]) if len(sys.argv) > 4 else 3

    recs = []
    for f in glob.glob(f"runs/{run}/rollouts/{step}/*{task}*.json"):
        try:
            recs.append(json.loads(open(f, encoding="utf-8").read()))
        except Exception:  # noqa: BLE001
            pass
    if not recs:
        print(f"未找到 {task} 的 rollout")
        return
    arm = lambda r: int((r.get("steering") or {}).get("steer_arm", -1))  # noqa: E731
    sc = lambda r: float((r.get("trajectory") or {}).get("score") or 0.0)  # noqa: E731
    print(f"{task} @ {step}: {len(recs)} 行")
    for a in (1, 0):
        rows = [r for r in recs if arm(r) == a]
        if not rows:
            continue
        name = "steered(带 doc)" if a == 1 else "bare(无 doc)"
        print(f"\n  {name}: n={len(rows)} "
              f"Score={sum(sc(r) for r in rows)/len(rows)*100:.1f} "
              f"Succ={sum(1 for r in rows if r.get('won'))}/{len(rows)}")
        # 动作序列的分布
        seqs = Counter()
        for r in rows:
            acts = [s.get("action_norm", "") for s in
                    (r.get("trajectory") or {}).get("steps", [])]
            seqs[" → ".join(acts[:6])] += 1
        for s, c in seqs.most_common(show):
            print(f"    ×{c} {s[:150]}")


if __name__ == "__main__":
    main()
