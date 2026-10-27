#!/usr/bin/env python3
"""GRPO 健康度核心:有效组率(组内既有赢又有输 = 有学习信号)。

纯 GRPO 在 norm_adv_by_std=false + 组内中心化下,全赢或全输的组 advantage 恒 0,
该组对梯度零贡献。有效组率是"这一步到底学到多少"的直接量。

用法: python3 eff_groups.py <run_dir> [<run_dir> ...]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def scan(run: Path) -> None:
    steps = sorted(p for p in (run / "rollouts").iterdir()
                   if p.is_dir() and not p.name.endswith("_val"))
    print(f"\n=== {run.name} ===")
    print(f"{'step':>5} {'grp':>4} {'eff':>5} {'eff%':>6} {'allwin':>7} {'alllose':>8} "
          f"{'won%':>6}")
    for sp in steps:
        by_task: dict[str, list[bool]] = defaultdict(list)
        for f in sp.glob("*.json"):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            if d.get("outcome") == "ENV_ERROR":
                continue
            by_task[d["task_id"]].append(bool(d["won"]))
        if not by_task:
            continue
        eff = allw = alll = 0
        tot_won = tot_n = 0
        for _, ws in by_task.items():
            tot_won += sum(ws)
            tot_n += len(ws)
            if all(ws):
                allw += 1
            elif not any(ws):
                alll += 1
            else:
                eff += 1
        g = len(by_task)
        print(f"{sp.name[-5:]:>5} {g:>4} {eff:>5} {eff / g:>5.0%} {allw:>7} {alll:>8} "
              f"{tot_won / max(1, tot_n):>5.1%}")


if __name__ == "__main__":
    for a in sys.argv[1:]:
        scan(Path(a))
