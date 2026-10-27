"""PGS 组内配对对照:特权 doc + 采样干预到底有没有带来提升?

设计上每个任务出 `rollout_n` 行,由 `steered_fraction` 切成两半:
  * steer_arm=1(steered):system prompt 注入 researcher 产的 per-task doc,
    且采样端有保尾核倾斜干预;
  * steer_arm=0(bare)  :同一任务、同一 step、同一权重,**无 doc 无干预**。
⇒ 同组内两臂的差值就是机制的效应,不受任务难度/训练进度混淆 —— 这是比
"跨 run 比总体均值"干净得多的对照(后者要靠 family 标准化才勉强可比)。

同时报 researcher 侧的过程量(实验次数/自评成功率/doc 字数),用来判断
"doc 没带来提升"到底是 doc 没用,还是 researcher 自己就没做成。

用法(A800,仓库根):
  python3 ws_arm_contrast.py classic_ws3b_pgs step_00001 [step_00002 ...]
"""
from __future__ import annotations

import glob
import json
import statistics
import sys
from collections import defaultdict


def load(run: str, step: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(f"runs/{run}/rollouts/{step}/*.json")):
        try:
            out.append(json.loads(open(f, encoding="utf-8").read()))
        except Exception:  # noqa: BLE001
            pass
    return out


def arm_of(r: dict) -> int:
    return int((r.get("steering") or {}).get("steer_arm", -1))


def score_of(r: dict) -> float:
    return float((r.get("trajectory") or {}).get("score") or 0.0)


def report(run: str, step: str) -> None:
    recs = [r for r in load(run, step) if r.get("outcome") != "ENV_ERROR"]
    if not recs:
        print(f"{run}/{step}: 无有效数据")
        return
    print(f"\n{'=' * 66}\n{run} / {step}   n={len(recs)}")

    # ---- 总体两臂 ----
    by_arm: dict[int, list[dict]] = defaultdict(list)
    for r in recs:
        by_arm[arm_of(r)].append(r)
    print(f"\n【总体】{'臂':<10}{'n':>5}{'Score':>8}{'Succ.':>9}"
          f"{'env_steps':>11}{'干预token':>10}")
    for arm in sorted(by_arm):
        rows = by_arm[arm]
        name = {1: "steered", 0: "bare"}.get(arm, f"未知({arm})")
        sc = [score_of(r) for r in rows]
        won = sum(1 for r in rows if r.get("won"))
        st = [int((r.get("steering") or {}).get("steer_count", 0)) for r in rows]
        print(f"      {name:<10}{len(rows):>5}{sum(sc)/len(sc)*100:>8.1f}"
              f"{won/len(rows):>9.2%}"
              f"{sum(r['env_steps'] for r in rows)/len(rows):>11.1f}"
              f"{sum(st)/len(rows):>10.1f}")

    # ---- 组内配对(同 task_id 才比)----
    by_task: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in recs:
        by_task[r["task_id"]][arm_of(r)].append(r)
    paired = {t: a for t, a in by_task.items() if a.get(0) and a.get(1)}
    print(f"\n【组内配对】{len(paired)} 个任务两臂齐全")
    if paired:
        d_score, d_won = [], []
        for _t, a in paired.items():
            s1 = statistics.mean(score_of(r) for r in a[1])
            s0 = statistics.mean(score_of(r) for r in a[0])
            w1 = statistics.mean(float(bool(r.get("won"))) for r in a[1])
            w0 = statistics.mean(float(bool(r.get("won"))) for r in a[0])
            d_score.append(s1 - s0)
            d_won.append(w1 - w0)
        pos = sum(1 for d in d_score if d > 1e-9)
        neg = sum(1 for d in d_score if d < -1e-9)
        print(f"      ΔScore(steered − bare): 均值 {statistics.mean(d_score)*100:+.2f} "
              f"中位 {statistics.median(d_score)*100:+.2f}")
        print(f"        任务级方向: {pos} 正 / {neg} 负 / "
              f"{len(d_score)-pos-neg} 持平")
        print(f"      ΔSucc.(steered − bare): 均值 {statistics.mean(d_won)*100:+.2f}pp")
        # 符号检验(粗略):正负各半为零假设
        n_eff = pos + neg
        if n_eff >= 8:
            import math
            z = (pos - n_eff / 2) / math.sqrt(n_eff / 4)
            print(f"        符号检验 z≈{z:+.2f}"
                  f"{'(|z|>1.96 显著)' if abs(z) > 1.96 else '(不显著)'}")

    # ---- researcher 过程量 ----
    res = [r.get("research") or {} for r in recs]
    res = [x for x in res if x]
    if res:
        def m(k):
            v = [float(x.get(k, 0) or 0) for x in res]
            return sum(v) / len(v) if v else 0.0
        succ = [float(x.get("research_final_success", 0) or 0) for x in res]
        words = [float(x.get("research_doc_words", 0) or 0) for x in res]
        print(f"\n【researcher 过程】n={len(res)}")
        print(f"      实验次数均 {m('research_experiments'):.2f} / "
              f"episodes 均 {m('research_episodes'):.2f} / "
              f"会话轮次均 {m('research_turns'):.2f}")
        print(f"      自评成功率 mean={statistics.mean(succ):.2%} "
              f"(=1.0 的占 {sum(1 for s in succ if s >= 1.0)/len(succ):.1%})")
        print(f"      doc 字数 p50={statistics.median(words):.0f} "
              f"min={min(words):.0f} max={max(words):.0f}")
        print(f"      doc 缺失 {sum(1 for x in res if x.get('research_doc_missing'))} / "
              f"error {sum(1 for x in res if x.get('research_error'))}")
        # ★ doc 有效性的上游判据:researcher 自己都没做成的任务,doc 多半是
        #   "我试了很多都失败"的记录 —— 对 executor 帮助有限。
        if paired:
            good = [t for t, a in paired.items()
                    if statistics.mean(float((r.get("research") or {})
                                             .get("research_final_success", 0) or 0)
                                       for r in a[1]) >= 1.0]
            bad = [t for t in paired if t not in good]
            for tag, ts in (("researcher 成功", good), ("researcher 失败", bad)):
                if not ts:
                    continue
                ds = []
                for t in ts:
                    a = paired[t]
                    ds.append(statistics.mean(score_of(r) for r in a[1])
                              - statistics.mean(score_of(r) for r in a[0]))
                print(f"      · {tag} 的 {len(ts)} 个任务: "
                      f"ΔScore 均值 {statistics.mean(ds)*100:+.2f}")


if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else "classic_ws3b_pgs"
    for st in (sys.argv[2:] or ["step_00001"]):
        report(run, st)
