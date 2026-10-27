"""step 133 阶跃前后:只看 bare 行的训练侧曲线(口径可比)。

停产前 512 行 = 256 bare + 256 steered,停产后 512 全 bare。混着看会被
steered 的高分带偏,所以一律**只统计 arm==0 的行**。
"""
from __future__ import annotations
import glob, json, re, statistics
from collections import defaultdict
from pathlib import Path

def series(run, lo, hi, every=1):
    rows = []
    for d in sorted((p for p in glob.glob(f"runs/{run}/rollouts/step_*")
                     if not p.endswith("_val")),
                    key=lambda p: int(re.search(r"step_0*(\d+)$", p).group(1))):
        s = int(re.search(r"step_0*(\d+)$", d).group(1))
        if not (lo <= s <= hi) or s % every:
            continue
        recs = []
        for f in glob.glob(f"{d}/*.json"):
            try: recs.append(json.loads(Path(f).read_text(encoding="utf-8")))
            except Exception: pass
        bare = [r for r in recs
                if int((r.get("steering") or {}).get("steer_arm", 0)) == 0
                and r.get("outcome") != "ENV_ERROR"]
        if len(bare) < 8: continue
        sc = [float((r.get("trajectory") or {}).get("score") or 0) for r in bare]
        won = [1.0 if r.get("won") else 0.0 for r in bare]
        ln = [len((r.get("trajectory") or {}).get("steps", [])) for r in bare]
        # 组内 score 方差(LOO 基线的噪声源)
        g = defaultdict(list)
        for r, v in zip(bare, sc):
            g[str(r.get("task_id"))].append(v)
        within = [statistics.pstdev(v) for v in g.values() if len(v) >= 2]
        rows.append((s, len(bare), statistics.mean(sc)*100,
                     statistics.mean(won)*100, statistics.mean(ln),
                     statistics.mean(within)*100 if within else 0.0,
                     len(g)))
    return rows

for run in ("classic_ws3b_pgs", "classic_ws3b_grpo"):
    print(f"\n=== {run} —— 只统计 bare 行 ===")
    print(f"{'step':>5} {'bare行':>6} {'Score':>7} {'Succ%':>7} {'轮数':>6} "
          f"{'组内σ':>7} {'组数':>5}")
    for r in series(run, 120, 200, 8):
        print(f"{r[0]:>5} {r[1]:>6} {r[2]:>7.1f} {r[3]:>7.2f} {r[4]:>6.2f} "
              f"{r[5]:>7.1f} {r[6]:>5}")

print("\n=== PGS 阶跃邻域逐步(130-140)===")
print(f"{'step':>5} {'bare行':>6} {'Score':>7} {'Succ%':>7} {'轮数':>6} {'组内σ':>7}")
for r in series("classic_ws3b_pgs", 130, 140, 1):
    print(f"{r[0]:>5} {r[1]:>6} {r[2]:>7.1f} {r[3]:>7.2f} {r[4]:>6.2f} {r[5]:>7.1f}")
