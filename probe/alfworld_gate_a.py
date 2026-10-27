"""ALFWorld Gate-A probe: scoring positive/negative controls via the engine.

Positive control: request the engine walkthrough (__GT__, policy_commands)
and replay it — the episode must end won=True. Also activates the playable
low-level GT channel (privileged-doc material upgrade).
Negative control: a do-nothing loop (look/inventory) — must end won=False.

Run on the machine that hosts the engine venv + json_2.1.1 data:
  python alfworld_gate_a.py --parquet <p1> [<p2> ...] --n 5 \
      --data-root <dir containing json_2.1.1> --worker-python <alfworld venv python>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# import the engine host from the bench tree (installed or via PYTHONPATH)
from trace_opd.envs.alfworld.adapter import _Worker  # noqa: PLC2701


def probe_task(item: dict, args) -> dict:
    gamefile = str(Path(args.data_root) / str(item["gamefile"]))
    rec: dict = {"task_id": item["task_id"], "split": item["split"]}

    # --- positive control: replay engine walkthrough
    w = _Worker(
        python_exe=args.worker_python,
        gamefile=gamefile,
        max_steps=120,
        data_root=args.data_root,
        start_timeout_s=300.0,
        step_timeout_s=120.0,
        want_gt=True,
    )
    try:
        plan = w.request("__GT__").get("plan")
        if not plan:
            rec["walkthrough"] = "missing"
        else:
            rec["walkthrough"] = f"{len(plan)} cmds"
            won = False
            for cmd in plan:
                ev = w.request(cmd)
                won = ev.get("won", False)
                if ev.get("done"):
                    break
            rec["positive_won"] = bool(won)
    finally:
        w.close()

    # --- negative control: do-nothing loop
    w = _Worker(
        python_exe=args.worker_python,
        gamefile=gamefile,
        max_steps=30,
        data_root=args.data_root,
        start_timeout_s=300.0,
        step_timeout_s=120.0,
    )
    try:
        won = False
        for i in range(20):
            ev = w.request("look" if i % 2 == 0 else "inventory")
            won = ev.get("won", False)
            if ev.get("done"):
                break
        rec["negative_won"] = bool(won)
    finally:
        w.close()
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=5, help="tasks per parquet file")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--worker-python", required=True)
    args = ap.parse_args()

    records = []
    for path in args.parquet:
        df = pd.read_parquet(path)
        for _, row in df.head(args.n).iterrows():
            item = dict(row["extra_info"])
            print(f"[probe] {item['task_id']}", flush=True)
            try:
                records.append(probe_task(item, args))
            except Exception as e:  # noqa: BLE001 — record, keep probing
                records.append(
                    {"task_id": item["task_id"], "split": item["split"],
                     "error": str(e)[:500]}
                )

    pos = [r for r in records if "positive_won" in r]
    neg = [r for r in records if "negative_won" in r]
    missing = [r for r in records if r.get("walkthrough") == "missing"]
    errors = [r for r in records if "error" in r]
    summary = {
        "tasks": len(records),
        "positive_pass": f"{sum(r['positive_won'] for r in pos)}/{len(pos)}",
        "negative_all_lost": all(not r["negative_won"] for r in neg) if neg else None,
        "walkthrough_missing": len(missing),
        "errors": len(errors),
    }
    print(json.dumps({"summary": summary, "records": records}, indent=2))
    ok = (
        pos
        and all(r["positive_won"] for r in pos)
        and neg
        and all(not r["negative_won"] for r in neg)
        and not errors
    )
    print("GATE-A-ALFWORLD:", "PASS" if ok else "CHECK-RECORDS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
