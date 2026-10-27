#!/usr/bin/env python3
"""只读扫描一个 run 的 rollout 产物:val 起点 + 训练步质量。

用法: python3 scan_arm.py <run_dir> [<run_dir2> ...]
不写任何文件,不碰 GPU,只 json.load。
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_dir(d: Path) -> list[dict]:
    out = []
    for f in sorted(d.glob("*.json")):
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception as e:  # 半写文件
            print(f"  !! 读失败 {f.name}: {e}")
    return out


def family(task_id: str) -> str:
    # task_id 形如 split/pick_heat_then_place_in_recep-Mug-.../trial_...
    parts = task_id.split("/")
    if len(parts) >= 2:
        return parts[1].split("-")[0]
    return "?"


_WS = re.compile(r"\s+")


def thoughts_of(traj: dict) -> list[str]:
    """从 trajectory 抽 assistant 的推理块内容。"""
    ths = []
    for s in traj.get("steps", []) or []:
        t = s.get("thought")
        if t:
            ths.append(_WS.sub(" ", t.strip()))
    return ths


def repeat_rate(ths: list[str]) -> float:
    """逐字重复率:重复出现的 thought 占比。"""
    if len(ths) < 2:
        return 0.0
    c = Counter(ths)
    dup = sum(n - 1 for n in c.values() if n > 1)
    return dup / len(ths)


def summarize_val(recs: list[dict], label: str) -> None:
    by_split = defaultdict(list)
    for r in recs:
        by_split[r["split"]].append(r)
    print(f"\n### {label}  (n={len(recs)})")
    for split in sorted(by_split):
        rows = by_split[split]
        errs = [r for r in rows if r["outcome"] == "ENV_ERROR"]
        att = [r for r in rows if r["outcome"] != "ENV_ERROR"]
        won = sum(1 for r in att if r["won"])
        ntask = len({r["task_id"] for r in rows})
        print(f"  {split:12s} {won:4d}/{len(att):4d} = {won/max(1,len(att)):7.2%}"
              f"  tasks={ntask} k={len(rows)/max(1,ntask):.1f}"
              f"  env_err={len(errs)}"
              f"  trunc={sum(1 for r in att if r.get('truncated')):3d}"
              f"  invfmt={sum(r.get('invalid_format',0) for r in att):4d}"
              f"  mean_steps={sum(r['env_steps'] for r in att)/max(1,len(att)):5.1f}")
        # family 分解
        fam = defaultdict(lambda: [0, 0])
        for r in att:
            f = family(r["task_id"])
            fam[f][1] += 1
            fam[f][0] += int(r["won"])
        line = "    ".join(f"{k}:{v[0]}/{v[1]}" for k, v in sorted(fam.items()))
        print(f"      {line}")


def summarize_train(run: Path, label: str) -> None:
    steps = sorted(p for p in (run / "rollouts").iterdir()
                   if p.is_dir() and not p.name.endswith("_val"))
    print(f"\n### {label} 训练步 (n_steps={len(steps)})")
    hdr = (f"  {'step':>5} {'n':>4} {'won':>7} {'trunc':>6} {'invfmt':>7} "
           f"{'steps':>6} {'toklen':>7} {'rep>0':>6} {'repmean':>8} "
           f"{'uniq_task':>9} {'outcomes'}")
    print(hdr)
    for sp in steps:
        recs = load_dir(sp)
        if not recs:
            continue
        att = [r for r in recs if r["outcome"] != "ENV_ERROR"]
        won = sum(1 for r in att if r["won"])
        reps = [repeat_rate(thoughts_of(r.get("trajectory", {}))) for r in att]
        nrep = sum(1 for x in reps if x > 0)
        oc = Counter(r["outcome"] for r in recs)
        print(f"  {sp.name[-5:]:>5} {len(recs):>4} {won/max(1,len(att)):>6.1%} "
              f"{sum(1 for r in att if r.get('truncated')):>6} "
              f"{sum(r.get('invalid_format',0) for r in att):>7} "
              f"{sum(r['env_steps'] for r in att)/max(1,len(att)):>6.1f} "
              f"{sum(r.get('response_token_len',0) for r in att)/max(1,len(att)):>7.0f} "
              f"{nrep/max(1,len(att)):>5.0%} {sum(reps)/max(1,len(reps)):>8.1%} "
              f"{len({r['task_id'] for r in recs}):>9} "
              f"{dict(oc)}")


def main() -> int:
    for arg in sys.argv[1:]:
        run = Path(arg)
        label = run.name
        vd = run / "rollouts" / "step_00000_val"
        if vd.is_dir():
            summarize_val(load_dir(vd), f"{label} val@step0")
        summarize_train(run, label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
