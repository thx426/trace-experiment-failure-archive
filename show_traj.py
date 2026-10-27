#!/usr/bin/env python3
"""渲染一条 rollout 轨迹用于人工质检。只读。

用法: python3 show_traj.py <step_dir> [--won|--lost] [--n 1] [--maxstep 12]
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path


def render(d: dict, maxstep: int) -> None:
    traj = d.get("trajectory", {})
    print("=" * 100)
    print(f"task   : {d['task_id']}")
    print(f"outcome: {d['outcome']}  won={d['won']}  env_steps={d['env_steps']} "
          f"turns={d['assistant_turns']} invalid_format={d.get('invalid_format')} "
          f"truncated={d.get('truncated')} resp_tokens={d.get('response_token_len')}")
    for k in ("score", "error"):
        if traj.get(k) is not None or d.get(k) is not None:
            print(f"{k:7s}: {traj.get(k, d.get(k))}")
    goal = traj.get("goal") or traj.get("task") or ""
    if goal:
        print(f"goal   : {goal}")
    print("-" * 100)
    steps = traj.get("steps", []) or []
    for i, s in enumerate(steps):
        if i >= maxstep:
            print(f"... (还有 {len(steps) - maxstep} 步,省略)")
            break
        th = (s.get("thought") or "").strip()
        ac = s.get("action_norm") or s.get("action_raw") or ""
        ob = (s.get("observation") or "").strip()
        print(f"[{i:02d}] acted={s.get('acted')} valid={s.get('valid')}")
        if th:
            print(textwrap.indent(textwrap.fill(th, 92)[:600], "     R| "))
        print(f"     A| {ac}")
        print(f"     O| {ob[:220]}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step_dir")
    ap.add_argument("--won", action="store_true")
    ap.add_argument("--lost", action="store_true")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--maxstep", type=int, default=12)
    a = ap.parse_args()
    shown = 0
    for f in sorted(Path(a.step_dir).glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if a.won and not d["won"]:
            continue
        if a.lost and d["won"]:
            continue
        render(d, a.maxstep)
        shown += 1
        if shown >= a.n:
            break
    if not shown:
        print("no matching trajectory")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
