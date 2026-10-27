"""ALFWorld 机制臂 val 曲线 + 末点六族 macro(与 SEED 同型对比用)。"""
import glob, json, re, statistics
from collections import defaultdict
from pathlib import Path

FAM = ["pick_and_place_simple", "pick_two_obj_and_place", "look_at_obj_in_light",
       "pick_heat_then_place", "pick_cool_then_place", "pick_clean_then_place"]

def fam_of(tid):
    for f in FAM:
        if f in tid:
            return f
    return "other"

out = {}
for run in ("classic_alf3b_pgs", "classic_alf7b_pgs"):
    dirs = sorted(glob.glob(f"runs/{run}/rollouts/step_*_val"),
                  key=lambda p: int(re.search(r"step_0*(\d+)_val", p).group(1)))
    if not dirs:
        continue
    probe = json.loads(Path(glob.glob(f"{dirs[0]}/*.json")[0]).read_text())
    print(f"{run}: task_id 样例 = {probe.get('task_id')}")
    vals, last_f = [], {}
    for d in dirs:
        s = int(re.search(r"step_0*(\d+)_val", d).group(1))
        by_split = defaultdict(list)
        fam_acc = defaultdict(list)
        for f in glob.glob(f"{d}/*.json"):
            try:
                r = json.loads(Path(f).read_text())
            except Exception:
                continue
            if r.get("outcome") == "ENV_ERROR":
                continue
            tid = str(r.get("task_id", ""))
            blob = tid + json.dumps(r.get("extra") or {}, ensure_ascii=False)
            sp = "unseen" if "unseen" in blob else ("seen" if "seen" in blob else "?")
            w = float(bool(r.get("won")))
            by_split[sp].append(w)
            fam_acc[(sp, fam_of(tid))].append(w)
        rec = {"step": s}
        for sp, v in by_split.items():
            rec[sp] = round(statistics.mean(v), 4)
            rec[f"n_{sp}"] = len(v)
        vals.append(rec)
        last_f = {f"{sp}:{fm}": (round(statistics.mean(v), 4), len(v))
                  for (sp, fm), v in sorted(fam_acc.items())}
    out[run] = {"vals": vals, "final_family": last_f}
    for r in vals:
        print("  ", r)
    print("  末点分族:")
    for k, (m, n) in last_f.items():
        print(f"    {k:42s} {m*100:6.1f}% (n={n})")
    for sp in ("seen", "unseen"):
        fams = [v[0] for k, v in last_f.items() if k.startswith(sp + ":") and "other" not in k]
        if fams:
            print(f"  ★ {sp} 六族 macro = {statistics.mean(fams)*100:.1f}"
                  f"(micro 见上表末行)")
Path("alf_val_series.json").write_text(json.dumps(out))
print("saved alf_val_series.json")
