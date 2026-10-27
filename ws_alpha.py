"""alpha=0 是字段名不对,还是 advantage 权重真的归零?"""
import glob, json, math
from pathlib import Path

d = sorted(p for p in glob.glob("runs/classic_ws3b_pgs/rollouts/step_*")
           if not p.endswith("_val"))[-3]
f = sorted(glob.glob(f"{d}/*.json"))
rec = None
for x in f:
    r = json.loads(Path(x).read_text(encoding="utf-8"))
    if int((r.get("steering") or {}).get("steer_arm", -1)) == 1:
        rec = r; break
print(f"=== {Path(d).name} 的一条 steered record ===")
print("顶层键:", sorted(rec.keys()))
print("\nsteering 全貌:")
print(json.dumps(rec.get("steering"), ensure_ascii=False, indent=2)[:1500])

print("\n=== 配置里的 advantage 衰减参数 ===")
import yaml
cfg = yaml.safe_load(Path("configs/classic/webshop_3b_pgs.yaml").read_text())
for k, v in sorted(cfg.items()):
    if isinstance(v, dict):
        for k2, v2 in sorted(v.items()):
            if "steer" in k2 or "alpha" in k2 or "adv" in k2:
                print(f"  {k}.{k2} = {v2}")
    elif "steer" in k or "alpha" in k or "adv" in k:
        print(f"  {k} = {v}")

print("\n=== launch 实际传给 trainer 的 steered_* ===")
import re, subprocess
log = subprocess.run("grep -oE '\\+algorithm\\.steered_[a-z_]+=[^ ]+' "
                     "runs/classic_ws3b_pgs.launch.log | tail -5",
                     shell=True, capture_output=True, text=True).stdout
print(log or "  (未找到)")

D = 133
for s in (1, 20, 82, 122, 133, 150):
    a = 0.0 if s >= D else 0.5 * (1 + math.cos(math.pi * s / D))
    print(f"  理论 alpha(cosine,decay_steps={D}) step {s:>3} = {a:.4f}")
