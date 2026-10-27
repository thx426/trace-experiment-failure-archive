#!/usr/bin/env python3
"""双臂同期对比:PGS vs GRPO 的训练 rollout 质量。只读。"""
from __future__ import annotations

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("scan", "scan_arm.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

ROOT = Path("runs")
HDR = (f"{'step':>5} {'n':>4} {'won':>7} {'trunc%':>7} {'invfmt/traj':>12} "
       f"{'envsteps':>9} {'toklen':>7} {'rep>0':>6} {'repmean':>8} {'uniqT':>6}")


def row(sp: Path) -> str | None:
    recs = m.load_dir(sp)
    if not recs:
        return None
    att = [r for r in recs if r["outcome"] != "ENV_ERROR"]
    if not att:
        return None
    won = sum(1 for r in att if r["won"])
    reps = [m.repeat_rate(m.thoughts_of(r.get("trajectory", {}))) for r in att]
    trunc = sum(1 for r in att if r.get("truncated"))
    invf = sum(r.get("invalid_format", 0) for r in att)
    return (f"{sp.name[-5:]:>5} {len(recs):>4} {won / len(att):>6.1%} "
            f"{trunc / len(att):>6.1%} {invf / len(att):>12.2f} "
            f"{sum(r['env_steps'] for r in att) / len(att):>9.1f} "
            f"{sum(r.get('response_token_len', 0) for r in att) / len(att):>7.0f} "
            f"{sum(1 for x in reps if x > 0) / len(att):>5.0%} "
            f"{sum(reps) / len(reps):>8.1%} "
            f"{len({r['task_id'] for r in recs}):>6}")


def dump(run_name: str, which: list[int] | None = None) -> None:
    run = ROOT / run_name
    steps = sorted(p for p in (run / "rollouts").iterdir()
                   if p.is_dir() and not p.name.endswith("_val"))
    if which is not None:
        steps = [p for p in steps if int(p.name.split("_")[-1]) in which]
    print(f"\n=== {run_name} ===")
    print(HDR)
    for sp in steps:
        r = row(sp)
        if r:
            print(r)


if __name__ == "__main__":
    early = [1, 2, 3, 4, 5]
    dump("classic_alf3b_grpo", early)
    dump("classic_alf3b_pgs", early + [20, 40, 60, 65, 66, 67])
