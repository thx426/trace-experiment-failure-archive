"""双臂 val 的**配对**检验:同一批 500 测试任务,逐任务配对比 avg@3。

配对比独立两样本功效高得多(消掉任务难度这个最大方差源)。
"""
from __future__ import annotations
import glob, json, math, re, statistics
from collections import defaultdict
from pathlib import Path

def per_task(run, step):
    d = f"runs/{run}/rollouts/step_{step:05d}_val"
    if not Path(d).exists():
        cand = glob.glob(f"runs/{run}/rollouts/step_*{step}_val")
        if not cand: return {}
        d = cand[0]
    agg = defaultdict(lambda: {"sc": [], "won": []})
    for f in glob.glob(f"{d}/*.json"):
        try: r = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception: continue
        if r.get("outcome") == "ENV_ERROR": continue
        t = str(r.get("task_id"))
        agg[t]["sc"].append(float((r.get("trajectory") or {}).get("score") or 0))
        agg[t]["won"].append(1.0 if r.get("won") else 0.0)
    return {t: (statistics.mean(v["sc"]), statistics.mean(v["won"]))
            for t, v in agg.items() if v["sc"]}

def paired(a, b, idx, label):
    common = sorted(set(a) & set(b))
    d = [a[t][idx] - b[t][idx] for t in common]
    n = len(d)
    m = statistics.mean(d)
    sd = statistics.stdev(d) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 and sd > 0 else 0.0
    t = m / se if se > 0 else 0.0
    # 正态近似的双侧 p
    p = math.erfc(abs(t) / math.sqrt(2))
    win = sum(1 for x in d if x > 1e-9)
    los = sum(1 for x in d if x < -1e-9)
    print(f"  {label:9} n={n} Δ={m*100:+6.2f} (95%CI {(m-1.96*se)*100:+6.2f}"
          f"~{(m+1.96*se)*100:+6.2f})  t={t:+5.2f} p={p:.4f}"
          f"  {'PGS赢' if m>0 else 'GRPO赢'} {win}/{los}/{n-win-los}")
    return p

print("=== 逐任务配对检验(PGS − GRPO;正=PGS 好)===")
print("   任务数 / Δ 均值(×100) / 95%CI / t / p / 赢-输-平\n")
for step in (20, 40, 100, 160, 180, 200):
    A, B = per_task("classic_ws3b_pgs", step), per_task("classic_ws3b_grpo", step)
    if not A or not B:
        print(f"step {step:>3}: 缺数据"); continue
    print(f"step {step:>3}:")
    paired(A, B, 0, "Score")
    paired(A, B, 1, "Succ.")
    print()
