"""干预臂 vs 裸臂逐步训练分数 —— "分布干预采样出更有效轨迹"的直接证据。

预期形态:早期 steered 显著 > bare(文档信息有增量);随训练 bare 追平
(内化);干预量(steer_count)随之下降(自动退场)。
"""
import glob, json, re, statistics
from pathlib import Path

CFG = {
    "classic_ws3b_pgs":  ("score", 8),
    "classic_alf3b_pgs": ("won", 8),
    "classic_alf7b_pgs": ("won", 8),
}
allout = {}
for run, (metric, every) in CFG.items():
    dirs = sorted((p for p in glob.glob(f"runs/{run}/rollouts/step_*")
                   if not p.endswith("_val")),
                  key=lambda p: int(re.search(r"step_0*(\d+)$", p).group(1)))
    rows = []
    for d in dirs:
        s = int(re.search(r"step_0*(\d+)$", d).group(1))
        if s > 8 and s % every:
            continue
        bare, st, cnts = [], [], []
        for f in glob.glob(f"{d}/*.json"):
            try:
                r = json.loads(Path(f).read_text(encoding="utf-8"))
            except Exception:
                continue
            if r.get("outcome") == "ENV_ERROR":
                continue
            arm = int((r.get("steering") or {}).get("steer_arm", -1))
            v = (float((r.get("trajectory") or {}).get("score") or 0)
                 if metric == "score" else float(bool(r.get("won"))))
            if arm == 0:
                bare.append(v)
            elif arm == 1:
                st.append(v)
                cnts.append(int((r.get("steering") or {}).get("steer_count") or 0))
        if not bare:
            continue
        rows.append({
            "step": s, "n_bare": len(bare), "n_st": len(st),
            "bare": statistics.mean(bare), 
            "st": statistics.mean(st) if st else None,
            "steer_cnt": statistics.mean(cnts) if cnts else None,
        })
    allout[run] = {"metric": metric, "rows": rows}
    print(f"\n=== {run}({metric}×100)===")
    print(f"{'step':>5} {'bare':>7} {'steer':>7} {'gap':>7} {'干预tok/条':>10}")
    for r in rows:
        stv = f"{r['st']*100:7.1f}" if r["st"] is not None else "   停产"
        gap = f"{(r['st']-r['bare'])*100:+7.1f}" if r["st"] is not None else "      -"
        c = f"{r['steer_cnt']:10.1f}" if r["steer_cnt"] is not None else "         -"
        print(f"{r['step']:>5} {r['bare']*100:7.1f} {stv} {gap} {c}")
    # 阶段汇总
    eb = [r for r in rows if r["st"] is not None]
    if eb:
        early = [r for r in eb if r["step"] <= 24]
        late = [r for r in eb if r["step"] >= max(eb[-1]["step"] - 24, 25)]
        for tag, seg in (("早期(≤24)", early), ("干预末期", late)):
            if seg:
                g = statistics.mean((r["st"] - r["bare"]) * 100 for r in seg)
                print(f"  {tag}: 平均 gap {g:+.1f}")
Path("arm_series_all.json").write_text(json.dumps(allout))
print("\nsaved arm_series_all.json")
