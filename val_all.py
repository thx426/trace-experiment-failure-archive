import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("scan", "scan_arm.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for run in ("classic_alf3b_pgs", "classic_alf3b_grpo", "classic_alf3b_seed"):
    p = Path("runs") / run / "rollouts"
    if not p.is_dir(): continue
    for d in sorted(p.glob("*_val")):
        recs = m.load_dir(d)
        if len(recs) < 800: continue
        line = f"{run:22s} {d.name:16s} "
        for split in ("valid_seen", "valid_unseen"):
            rows = [r for r in recs if r["split"] == split and r["outcome"] != "ENV_ERROR"]
            if rows:
                line += f"{split.split('_')[1]}={sum(1 for r in rows if r['won'])/len(rows):6.2%}  "
        print(line)
