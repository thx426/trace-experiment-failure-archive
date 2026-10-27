#!/usr/bin/env python3
"""对一棵 src/trace_opd 打逐文件 sha256,供跨机对拍(记忆 sync-overwrites-prod-patches:
先分清远端是纯落后还是有独有改动,再决定同步方向)。

用法: python3 sha_tree.py <src_root> > out.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    out = {}
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    json.dump(out, sys.stdout, indent=0, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
