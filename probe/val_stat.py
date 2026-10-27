#!/usr/bin/env python3
"""统计 step0 val 读数(seen/unseen × 任务族)。用法: val_stat.py <val_dir>"""
import json, glob, sys, collections, os

d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, "*.json")))
if not files:
    print("no files in", d); sys.exit(1)

# 先探一个 schema
probe = json.load(open(files[0]))
print("schema keys:", sorted(probe.keys())[:20])
for k in ("won", "score", "reward", "metrics", "success"):
    if k in probe:
        print("  %-10s = %r" % (k, probe[k] if not isinstance(probe[k], (list, dict)) else type(probe[k])))
print()


def won_of(rec):
    for k in ("won", "success"):
        if isinstance(rec.get(k), bool):
            return int(rec[k])
    m = rec.get("metrics") or {}
    for k in ("won", "success"):
        if k in m:
            return int(bool(m[k]))
    for k in ("score", "reward"):
        v = rec.get(k, m.get(k))
        if isinstance(v, (int, float)):
            return int(v >= 1.0)
    return 0


split = {"seen": collections.Counter(), "unseen": collections.Counter()}
fam = {"seen": collections.defaultdict(collections.Counter),
       "unseen": collections.defaultdict(collections.Counter)}
trunc = collections.Counter()

for f in files:
    rec = json.load(open(f))
    base = os.path.basename(f)
    sp = "seen" if base.startswith("valid_seen__") else "unseen"
    w = won_of(rec)
    split[sp]["n"] += 1
    split[sp]["won"] += w
    family = base.split("__", 1)[1].split("-")[0] if "__" in base else "?"
    fam[sp][family]["n"] += 1
    fam[sp][family]["won"] += w
    m = rec.get("metrics") or {}
    if rec.get("truncated") or m.get("truncated"):
        trunc[sp] += 1

for sp in ("seen", "unseen"):
    c = split[sp]
    if not c["n"]:
        continue
    print("valid_%s: %d/%d = %.1f%%   (truncated=%d)"
          % (sp, c["won"], c["n"], 100.0 * c["won"] / c["n"], trunc[sp]))
    for k in sorted(fam[sp]):
        cc = fam[sp][k]
        print("    %-28s %2d/%2d = %5.1f%%" % (k, cc["won"], cc["n"], 100.0 * cc["won"] / cc["n"]))
    print()
