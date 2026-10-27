"""Extract engine walkthroughs for ALFWorld tasks and render them as
privileged docs (gt.py render_gt_as_doc, the G0-ablation posture).

Writes <out-dir>/<task_id_safe>.md per task with a playable walkthrough;
tasks whose engine yields no walkthrough are listed in missing.json and a
filtered parquet (only covered tasks) is written for paired eval.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from trace_opd.envs.alfworld.adapter import _Worker  # noqa: PLC2701
from trace_opd.envs.alfworld.gt import render_gt_as_doc, render_walkthrough


def safe_name(task_id: str) -> str:
    return task_id.replace("/", "__").replace(" ", "_")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--filtered-parquet", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--worker-python", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.parquet)
    covered_idx, missing = [], []
    for idx, row in df.iterrows():
        item = dict(row["extra_info"])
        tid = item["task_id"]
        dest = out / f"{safe_name(tid)}.md"
        if dest.exists():
            covered_idx.append(idx)
            continue
        gamefile = str(Path(args.data_root) / str(item["gamefile"]))
        w = _Worker(
            python_exe=args.worker_python, gamefile=gamefile, max_steps=10,
            data_root=args.data_root, start_timeout_s=300.0,
            step_timeout_s=120.0, want_gt=True,
        )
        try:
            plan = w.request("__GT__").get("plan")
        finally:
            w.close()
        doc = render_gt_as_doc(render_walkthrough(plan))
        if doc:
            dest.write_text(doc, encoding="utf-8")
            covered_idx.append(idx)
        else:
            missing.append(tid)
        print(f"[gt] {len(covered_idx)}+{len(missing)}/{len(df)}", flush=True)

    df.loc[covered_idx].to_parquet(args.filtered_parquet, index=False)
    (out / "missing.json").write_text(json.dumps(missing, indent=2))
    print(f"DONE covered={len(covered_idx)} missing={len(missing)} "
          f"filtered={args.filtered_parquet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
