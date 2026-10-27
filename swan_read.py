#!/usr/bin/env python3
"""只读拉取 swanlab 云端指标。禁止 swanlab sync(会终结在跑的 run)。

用法: python3 swan_read.py <experiment_name> [key_substr ...]
无 key 参数时先列出全部 summary key。
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("SWANLAB_API_KEY", "58KT7T652vKUKobNhHb50")
import swanlab  # noqa: E402


def main() -> int:
    exp = sys.argv[1]
    pats = [p.lower() for p in sys.argv[2:]]
    api = swanlab.Api()
    runs = list(api.runs("Wushang/trace_opd"))
    try:
        run = next(r for r in runs if r.name == exp)
    except StopIteration:
        print("未找到 experiment;现有(最近 15):")
        for r in runs[:15]:
            print("  ", r.name)
        return 1
    summ = run.summary
    if callable(summ):
        summ = summ()
    keys = sorted(summ)
    if not pats:
        print(f"# {exp}: {len(keys)} keys")
        for k in keys:
            v = summ[k]
            step = v.get("step") if isinstance(v, dict) else None
            val = v.get("value") if isinstance(v, dict) else v
            print(f"  {k:60s} step={step} val={val}")
        return 0
    sel = [k for k in keys if any(p in k.lower() for p in pats)]
    if not sel:
        print("no key matched")
        return 1
    seqs = run.metrics(sel)
    # 版本差异:可能是 list[dict] / DataFrame / dict[key]->rows
    if hasattr(seqs, "columns"):  # pandas DataFrame
        import pandas as pd  # noqa: F401
        cols = [c for c in seqs.columns if c in sel or any(p in str(c).lower() for p in pats)]
        idx = seqs.index
        for c in cols:
            s = seqs[c].dropna()
            head = "  ".join(f"{int(i)}:{float(v):.4g}" for i, v in s.items())
            print(f"\n## {c}  (n={len(s)})\n   {head}")
        if not cols:
            print("DataFrame columns:", list(seqs.columns)[:40])
            print("index sample:", list(idx)[:10])
        return 0
    if isinstance(seqs, dict) and "list" in seqs:
        # 现行返回:{"keys":[...],"metricType":...,"list":[{"key":..,"metrics":[..]}]}
        items = [(x.get("key"), x) for x in seqs["list"]]
    elif isinstance(seqs, dict):
        items = list(seqs.items())
    else:
        items = [(getattr(x, "key", None) or (x.get("key") if isinstance(x, dict) else None), x)
                 for x in seqs]
    for k, item in items:
        if isinstance(item, dict):
            rows = item.get("metrics") or item.get("data") or []
        else:
            rows = getattr(item, "metrics", None) or []
        pts = []
        for r in rows:
            if isinstance(r, dict):
                i, d = r.get("index"), r.get("data")
            else:
                i, d = getattr(r, "index", None), getattr(r, "data", None)
            if i is not None and d is not None:
                try:
                    pts.append((int(i), float(d)))
                except (TypeError, ValueError):
                    pass
        pts.sort()
        head = "  ".join(f"{i}:{d:.4g}" for i, d in pts)
        print(f"\n## {k}  (n={len(pts)})\n   {head}")
        if not pts:
            print("   raw:", repr(item)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
