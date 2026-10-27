#!/usr/bin/env python3
"""格式失效模式统计:thought 里残留标签 = 抽取器没匹配上。

关注 <reasoning> 的拼写变体(模型实测会写成 <reasonning>)。
用法: python3 fmt_modes.py <step_dir> [<step_dir> ...]
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

TAGISH = re.compile(r"<\s*/?\s*([A-Za-z_][A-Za-z0-9_]{0,24})\s*>")


def scan(sp: Path) -> None:
    tags: Counter[str] = Counter()
    n_step = n_dirty = 0
    n_traj = n_traj_dirty = 0
    invfmt = 0
    for f in sp.glob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("outcome") == "ENV_ERROR":
            continue
        n_traj += 1
        invfmt += d.get("invalid_format", 0)
        dirty_here = False
        for s in d.get("trajectory", {}).get("steps", []) or []:
            n_step += 1
            th = s.get("thought") or ""
            found = TAGISH.findall(th)
            if found:
                n_dirty += 1
                dirty_here = True
                for t in found:
                    tags[t.lower()] += 1
        if dirty_here:
            n_traj_dirty += 1
    print(f"{sp.parent.parent.name}/{sp.name}: "
          f"traj={n_traj} 含残留标签的轨迹={n_traj_dirty} ({n_traj_dirty/max(1,n_traj):.0%})  "
          f"step={n_step} 残留step={n_dirty} ({n_dirty/max(1,n_step):.1%})  "
          f"invalid_format合计={invfmt}")
    if tags:
        print("   残留标签 TOP:", "  ".join(f"{k}×{v}" for k, v in tags.most_common(8)))


if __name__ == "__main__":
    for a in sys.argv[1:]:
        scan(Path(a))
