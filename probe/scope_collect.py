#!/usr/bin/env python3
"""双臂实验显微镜:扫描 rollouts 产物,汇总逐步指标 + 抽取代表性轨迹。
输出单个 JSON 供本地渲染。用法: scope_collect.py <out.json>

观察口径遵循既有纪律:
  - 赢/输两通道分开(极化是崩塌前置阶段,合并看会被均值掩盖)
  - thought 逐字重复率(最强判别指标:赢轨迹恒 0、输轨迹 26-70%)
  - 截断率 + 未截断子集胜率(截断会把能力上升伪装成下跌)
  - 机制臂额外分 steered/bare 两臂(steered_fraction=0.5),差值=机制直接读数
"""
import json, glob, os, sys, re, collections

W = "/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM"
# 任务对由环境变量指定(默认 ALFWorld),便于同一套看板服务多个域:
#   RUN_PREFIX=classic_sqa7b python scope_collect.py out.json
PREFIX = os.environ.get("RUN_PREFIX", "classic_alf7b")
ARMS = {"pgs": f"{PREFIX}_pgs", "grpo": f"{PREFIX}_grpo"}


def thought_repeat_rate(steps):
    """thought 逐字重复率:出现过 ≥2 次的 thought 占总轮数的比例。"""
    th = [(s.get("thought") or "").strip() for s in steps]
    th = [t for t in th if t]
    if len(th) < 2:
        return 0.0
    c = collections.Counter(th)
    dup = sum(n for t, n in c.items() if n >= 2)
    return dup / len(th)


def load_step(d):
    recs = []
    for f in glob.glob(os.path.join(d, "*.json")):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        r["_f"] = os.path.basename(f)
        recs.append(r)
    return recs


def summarize(recs):
    if not recs:
        return None
    n = len(recs)
    won = [r for r in recs if r.get("won")]
    lost = [r for r in recs if not r.get("won")]
    trunc = [r for r in recs if r.get("truncated")]
    nt = [r for r in recs if not r.get("truncated")]

    def med(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2] if xs else 0

    def steps_of(r):
        return (r.get("trajectory") or {}).get("steps") or []

    tot_turns = sum(len(steps_of(r)) for r in recs)
    tot_valid = sum(sum(1 for s in steps_of(r) if s.get("valid")) for r in recs)
    tot_acted = sum(sum(1 for s in steps_of(r) if s.get("acted")) for r in recs)

    out = {
        "n": n,
        "won": len(won),
        "won_rate": len(won) / n,
        "trunc_rate": len(trunc) / n,
        "won_rate_untrunc": (sum(1 for r in nt if r.get("won")) / len(nt)) if nt else None,
        "won_and_trunc": sum(1 for r in recs if r.get("won") and r.get("truncated")),
        "turns_med": med([r.get("assistant_turns", 0) for r in recs]),
        "turns_med_won": med([r.get("assistant_turns", 0) for r in won]),
        "env_steps_med": med([r.get("env_steps", 0) for r in recs]),
        "resp_tok_med": med([r.get("response_token_len", 0) for r in recs]),
        "acted_rate": tot_acted / max(tot_turns, 1),      # 格式服从
        "valid_rate": tot_valid / max(tot_turns, 1),      # 动作被引擎接受
        "invalid_format_mean": sum(r.get("invalid_format", 0) for r in recs) / n,
        "degenerate": sum(1 for r in recs if str(r.get("outcome", "")).startswith("DEGENERATE")),
        # 赢/输两通道的 thought 重复率(极化判别)
        "repeat_won": round(sum(thought_repeat_rate(steps_of(r)) for r in won) / max(len(won), 1), 4),
        "repeat_lost": round(sum(thought_repeat_rate(steps_of(r)) for r in lost) / max(len(lost), 1), 4),
    }
    # 任务族分解
    fam = collections.defaultdict(lambda: [0, 0])
    for r in recs:
        b = r["_f"]
        m = re.search(r"__([a-z_]+)-", b)      # ALFWorld: <split>__<族>-<Obj>-…
        if m:
            k = m.group(1)
        else:                                   # SearchQA: <source>_<split>_<n>
            k = str(r.get("task_id", b)).split("_")[0] or "?"
        fam[k][1] += 1
        if r.get("won"):
            fam[k][0] += 1
    out["family"] = {k: {"won": v[0], "n": v[1]} for k, v in sorted(fam.items())}
    return out


def collect_arm(arm, run):
    root = f"{W}/runs/{run}/rollouts"
    res = {"run": run, "train": [], "val": []}
    if not os.path.isdir(root):
        return res
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if not os.path.isdir(p):
            continue
        recs = load_step(p)
        if not recs:
            continue
        s = summarize(recs)
        if s is None:
            continue
        m = re.search(r"step_(\d+)", d)
        s["step"] = int(m.group(1)) if m else -1
        s["dir"] = d
        if d.endswith("_val"):
            # val 再按 seen/unseen 拆
            for sp in ("seen", "unseen"):
                sub = [r for r in recs if r["_f"].startswith(f"valid_{sp}__")]
                if sub:
                    ss = summarize(sub)
                    s[f"val_{sp}"] = {"n": ss["n"], "won_rate": ss["won_rate"],
                                      "trunc_rate": ss["trunc_rate"],
                                      "family": ss["family"]}
            res["val"].append(s)
        else:
            # 机制臂:分 steered / bare 两臂(机制是否起作用的直接读数)
            st = [r for r in recs if (r.get("steering") or {}).get("steer_arm") == 1]
            ba = [r for r in recs if (r.get("steering") or {}).get("steer_arm") == 0]
            if st and ba:
                a, b = summarize(st), summarize(ba)
                s["steered"] = {"n": a["n"], "won_rate": a["won_rate"], "valid_rate": a["valid_rate"],
                                "turns_med": a["turns_med"], "repeat_lost": a["repeat_lost"]}
                s["bare"] = {"n": b["n"], "won_rate": b["won_rate"], "valid_rate": b["valid_rate"],
                             "turns_med": b["turns_med"], "repeat_lost": b["repeat_lost"]}
                s["steer_gap"] = a["won_rate"] - b["won_rate"]
                pos = [len((r.get("steering") or {}).get("steer_positions") or []) for r in st]
                s["steer_pos_med"] = sorted(pos)[len(pos) // 2] if pos else 0
            # researcher 观察面
            rs = [r.get("research") or {} for r in recs if r.get("research")]
            if rs:
                def avg(k):
                    v = [x.get(k) for x in rs if isinstance(x.get(k), (int, float))]
                    return round(sum(v) / len(v), 3) if v else None
                s["research"] = {
                    "exp": avg("research_experiments"), "eps": avg("research_episodes"),
                    "turns": avg("research_turns"), "final_success": avg("research_final_success"),
                    "doc_words": avg("research_doc_words"),
                    "doc_missing": sum(int(x.get("research_doc_missing") or 0) for x in rs),
                    "errors": sum(int(x.get("research_error") or 0) for x in rs),
                }
            res["train"].append(s)
    return res


def pick_specimens(run, limit=3):
    """显微镜标本:最新训练步里各抽赢/输轨迹,带逐轮明细。"""
    root = f"{W}/runs/{run}/rollouts"
    dirs = [d for d in sorted(os.listdir(root)) if not d.endswith("_val")] if os.path.isdir(root) else []
    if not dirs:
        return []
    recs = load_step(os.path.join(root, dirs[-1]))
    out = []
    for tag, pool in (("won", [r for r in recs if r.get("won")]),
                      ("lost", [r for r in recs if not r.get("won")])):
        for r in pool[:limit]:
            st = (r.get("trajectory") or {}).get("steps") or []
            out.append({
                "tag": tag, "step_dir": dirs[-1], "task_id": r.get("task_id"),
                "turns": r.get("assistant_turns"), "env_steps": r.get("env_steps"),
                "outcome": r.get("outcome"), "invalid_format": r.get("invalid_format"),
                "steer_arm": (r.get("steering") or {}).get("steer_arm"),
                "n_steer_pos": len((r.get("steering") or {}).get("steer_positions") or []),
                "repeat": round(thought_repeat_rate(st), 3),
                "steps": [{"t": s.get("t"), "valid": bool(s.get("valid")), "acted": bool(s.get("acted")),
                           "action": s.get("action_norm") or "",
                           "thought": (s.get("thought") or "")[:260],
                           "obs": (s.get("observation") or "")[:200]} for s in st],
            })
    return out


if __name__ == "__main__":
    out = {"generated_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"), "arms": {}}
    for arm, run in ARMS.items():
        out["arms"][arm] = collect_arm(arm, run)
        out["arms"][arm]["specimens"] = pick_specimens(run)
    # watchdog 状态
    try:
        out["watchdog"] = open(f"{W}/runs/watchdog.log").read().strip().split("\n")[-6:]
    except Exception:
        out["watchdog"] = []
    json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False)
    print("written:", sys.argv[1], os.path.getsize(sys.argv[1]), "bytes")
