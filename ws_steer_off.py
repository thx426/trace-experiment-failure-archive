"""α=0 之后采样端干预也停了 —— 是设计(短路省算力)还是干预死了?"""
import glob, json, re, subprocess
from collections import Counter
from pathlib import Path

print("=== 1. 各步的臂构成(steer_arm 分布)===")
steps = sorted((p for p in glob.glob("runs/classic_ws3b_pgs/rollouts/step_*")
                if not p.endswith("_val")),
               key=lambda p: int(re.search(r"step_0*(\d+)$", p).group(1)))
pick = [s for s in steps if int(re.search(r"step_0*(\d+)$", s).group(1))
        in (100, 125, 130, 132, 133, 134, 135, 140, 160)]
print(f"{'step':>5} {'n':>5} {'arm=0(bare)':>12} {'arm=1(steer)':>13} {'其他':>6}")
for d in pick:
    recs = []
    for f in glob.glob(f"{d}/*.json"):
        try: recs.append(json.loads(Path(f).read_text(encoding="utf-8")))
        except Exception: pass
    if not recs: continue
    c = Counter(int((r.get("steering") or {}).get("steer_arm", -9)) for r in recs)
    s = int(re.search(r"step_0*(\d+)$", d).group(1))
    print(f"{s:>5} {len(recs):>5} {c.get(0,0):>12} {c.get(1,0):>13} "
          f"{sum(v for k,v in c.items() if k not in (0,1)):>6}")

print("\n=== 2. 代码里是否有 alpha==0 短路 ===")
for pat in ("alpha", "steered_alpha", "adv_decay"):
    out = subprocess.run(
        f"grep -rn --include=*.py -E '(alpha *(<=|==|<) *0|alpha *(and|if)|skip.*steer|steer.*skip)' "
        f"src/trace_opd/ train/ 2>/dev/null | head -20",
        shell=True, capture_output=True, text=True).stdout
    if out: print(out); break

print("\n=== 3. alpha 计算与消费点 ===")
print(subprocess.run(
    "grep -rn --include=*.py 'steered_alpha\\|steered_adv_decay' src/ train/ 2>/dev/null | head -20",
    shell=True, capture_output=True, text=True).stdout)

print("=== 4. 训练日志里 alpha 归零前后有无异常 ===")
print(subprocess.run(
    "grep -niE 'steer.*(disabl|off|skip|zero)|alpha.*(0\\.0|zero)' "
    "runs/classic_ws3b_pgs.launch.log | tail -10",
    shell=True, capture_output=True, text=True).stdout or "  (无相关日志)")
