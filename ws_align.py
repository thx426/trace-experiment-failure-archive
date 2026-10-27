"""双臂 val 对齐 + PGS 采样干预是否真的在动。"""
import glob, json, re, statistics
from collections import defaultdict
from pathlib import Path

def val_series(run):
    out = {}
    for d in sorted(glob.glob(f"runs/{run}/rollouts/step_*_val")):
        recs = []
        for f in glob.glob(f"{d}/*.json"):
            try: recs.append(json.loads(Path(f).read_text(encoding="utf-8")))
            except Exception: pass
        att = [x for x in recs if x.get("outcome") != "ENV_ERROR"]
        if not att: continue
        sc = [float((x.get("trajectory") or {}).get("score") or 0) for x in att]
        s = int(re.search(r"step_0*(\d+)_val", d).group(1))
        out[s] = (sum(sc)/len(sc)*100,
                  sum(1 for x in att if x.get("won"))/len(att)*100, len(recs))
    return out

P, G = val_series("classic_ws3b_pgs"), val_series("classic_ws3b_grpo")
print("=== val 对齐(avg@3)===")
print(f"{'step':>5} | {'PGS Sc':>7} {'PGS SR%':>8} | {'GRPO Sc':>8} {'GRPO SR%':>9} | {'ΔSc':>6} {'ΔSR':>6}")
for s in sorted(set(P) & set(G)):
    p, g = P[s], G[s]
    print(f"{s:>5} | {p[0]:>7.2f} {p[1]:>8.2f} | {g[0]:>8.2f} {g[1]:>9.2f} | "
          f"{p[0]-g[0]:>+6.2f} {p[1]-g[1]:>+6.2f}")
only = sorted(set(P) - set(G))
if only:
    print("PGS 独有:", {s: f"{P[s][0]:.1f}/{P[s][1]:.1f}%" for s in only})

# ---- 采样干预是否在动 -------------------------------------------------
print("\n=== PGS 采样干预(按训练步)===")
steps = sorted(p for p in glob.glob("runs/classic_ws3b_pgs/rollouts/step_*")
               if not p.endswith("_val"))
print(f"{'step':>5} {'n':>4} {'bare':>5} {'steer':>6} {'有干预%':>8} "
      f"{'steer_tok中位':>12} {'alpha':>7} {'bareSc':>7} {'stSc':>7} {'Δ':>6}")
for d in steps[-1:] + steps[-6:-5] + steps[-21:-20] + steps[-41:-40]:
    recs = []
    for f in glob.glob(f"{d}/*.json"):
        try: recs.append(json.loads(Path(f).read_text(encoding="utf-8")))
        except Exception: pass
    if not recs: continue
    arm = lambda r: int((r.get("steering") or {}).get("steer_arm", -1))
    sc = lambda r: float((r.get("trajectory") or {}).get("score") or 0)*100
    bare = [r for r in recs if arm(r) == 0]
    st = [r for r in recs if arm(r) == 1]
    cnt = [int((r.get("steering") or {}).get("steer_count") or 0) for r in st]
    pos = [c for c in cnt if c > 0]
    al = [float((r.get("steering") or {}).get("alpha") or 0) for r in st]
    s = int(re.search(r"step_0*(\d+)$", d).group(1))
    bs = statistics.mean([sc(r) for r in bare]) if bare else 0
    ss = statistics.mean([sc(r) for r in st]) if st else 0
    print(f"{s:>5} {len(recs):>4} {len(bare):>5} {len(st):>6} "
          f"{len(pos)/max(1,len(cnt)):>8.1%} "
          f"{statistics.median(pos) if pos else 0:>12.0f} "
          f"{statistics.mean(al) if al else 0:>7.3f} {bs:>7.1f} {ss:>7.1f} {ss-bs:>+6.1f}")
