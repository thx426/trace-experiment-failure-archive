"""GRPO 臂收官盘点:val 序列是否齐全、进程是否干净退出、产物是否完整。"""
import glob, json, os, re, subprocess
from pathlib import Path

R = "classic_ws3b_grpo"
print("=== 1. launch.log 尾部(看退出是否干净)===")
print(subprocess.run(f"tail -6 runs/{R}.launch.log", shell=True,
                     capture_output=True, text=True).stdout)

print("=== 2. val 序列(avg@3)===")
rows = []
for d in sorted(glob.glob(f"runs/{R}/rollouts/step_*_val")):
    recs = []
    for f in glob.glob(f"{d}/*.json"):
        try:
            recs.append(json.loads(Path(f).read_text(encoding="utf-8")))
        except Exception:
            pass
    att = [x for x in recs if x.get("outcome") != "ENV_ERROR"]
    if not att:
        continue
    sc = [float((x.get("trajectory") or {}).get("score") or 0) for x in att]
    won = sum(1 for x in att if x.get("won"))
    rows.append((int(re.search(r"step_0*(\d+)_val", d).group(1)),
                 len(recs), len(recs) - len(att),
                 sum(sc) / len(sc) * 100, won / len(att) * 100))
print(f"{'step':>5} {'n':>5} {'err':>4} {'Score':>7} {'Succ.%':>7}")
for s, n, e, sc, w in rows:
    print(f"{s:>5} {n:>5} {e:>4} {sc:>7.2f} {w:>7.2f}")

print("\n=== 3. 产物完整性 ===")
print("训练步目录:", len([p for p in glob.glob(f"runs/{R}/rollouts/step_*")
                          if not p.endswith("_val")]))
print("val 目录:", len(glob.glob(f"runs/{R}/rollouts/step_*_val")))
for c in sorted(glob.glob(f"runs/{R}/checkpoints/global_step_*")):
    sz = subprocess.run(f"du -sh {c} 2>/dev/null | cut -f1", shell=True,
                        capture_output=True, text=True).stdout.strip()
    print(f"  ckpt {Path(c).name}: {sz}")
print("总占用:", subprocess.run(f"du -sh runs/{R} 2>/dev/null | cut -f1",
      shell=True, capture_output=True, text=True).stdout.strip())
