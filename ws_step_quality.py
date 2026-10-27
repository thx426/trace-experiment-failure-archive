"""WebShop 训练步的 rollout 质量速查(Score + Succ. 双报口径)。"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter


def one(run: str, step: str) -> None:
    fs = sorted(glob.glob(f"runs/{run}/rollouts/{step}/*.json"))
    if not fs:
        print(f"{run}/{step}: (无产出)")
        return
    recs = []
    for f in fs:
        try:
            recs.append(json.loads(open(f, encoding="utf-8").read()))
        except Exception:  # noqa: BLE001
            pass
    att = [r for r in recs if r.get("outcome") != "ENV_ERROR"]
    if not att:
        print(f"{run}/{step}: 全是 ENV_ERROR({len(recs)})")
        return
    won = sum(1 for r in att if r.get("won"))
    sc = [float((r.get("trajectory") or {}).get("score") or 0.0) for r in att]
    invf = sum(int(r.get("invalid_format") or 0) for r in att)
    trunc = sum(1 for r in att if r.get("truncated"))
    steps = sum(int(r.get("env_steps") or 0) for r in att) / len(att)
    steer = sum(1 for r in att if r.get("steering"))
    res = sum(1 for r in att if r.get("research"))
    print(f"{run}/{step}: n={len(recs)} 有效={len(att)} "
          f"env_err={len(recs) - len(att)}")
    print(f"   Score {sum(sc) / len(sc) * 100:5.1f} | Succ. {won / len(att):6.2%} "
          f"({won}/{len(att)})")
    print(f"   invalid_format={invf}  truncated={trunc}  env_steps均={steps:.1f}")
    print(f"   带 steering 字段={steer}  带 research 字段={res}")
    print(f"   outcome={dict(Counter(r.get('outcome') for r in recs))}")


if __name__ == "__main__":
    for run in ("classic_ws3b_pgs", "classic_ws3b_grpo"):
        for step in sys.argv[1:] or ["step_00001"]:
            one(run, step)
