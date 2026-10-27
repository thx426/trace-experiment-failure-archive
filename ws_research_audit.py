"""researcher 迭代过程与 doc 质量的批量审计。

回答两个问题:
  ① **迭代有没有效**:researcher 每轮实验会把当前 doc 注入 executor 再跑,
     所以"首轮(bare,无 doc)成功率"→"末轮(带最新 doc)成功率"的变化,就是
     doc 在**researcher 自己的实验里**是否奏效的直接证据。这比事后看
     executor 训练读数更早、更干净。
  ② **doc 写了什么、对不对**:抽取每份 doc 的强断言句("Confirmed" /
     "always" / "never" / "regardless of" 等),供人工核验事实正确性 ——
     已知 researcher 会把自己实验设计的缺陷写成环境性质(2026-08-17 实测:
     "搜索不区分 query" 被 Jaccard=0.002 证伪)。

用法(A800,仓库根):
  python3 ws_research_audit.py [run_name] [--docs N] [--full-doc TASK]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

STRONG = re.compile(
    r"(confirmed|always|never|every|all (?:observed|episodes|searches)|"
    r"regardless of|does not exist|impossible|cannot)", re.I)


def load_task(d: Path) -> dict | None:
    try:
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    eps = []
    f = d / "episodes.jsonl"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    eps.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    pass
    doc = ""
    if (d / "doc.md").exists():
        doc = (d / "doc.md").read_text(encoding="utf-8")
    return {"meta": meta, "eps": eps, "doc": doc, "dir": d}


def won_of(ep: dict) -> bool:
    return bool((ep.get("trajectory") or {}).get("won"))


def score_of(ep: dict) -> float:
    return float((ep.get("trajectory") or {}).get("score") or 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="classic_ws3b_pgs")
    ap.add_argument("--docs", type=int, default=4)
    ap.add_argument("--full-doc", default="")
    a = ap.parse_args()

    dirs = [Path(p) for p in sorted(glob.glob(f"runs/{a.run}/research/*/*/"))
            if Path(p, "meta.json").exists()]
    tasks = [t for t in (load_task(d) for d in dirs) if t]
    print(f"[audit] {len(tasks)} 个 researcher 会话")
    if not tasks:
        return 1

    # ---- ① 首轮(bare) vs 末轮(带 doc) ---------------------------------
    print("\n=== ① 迭代过程:首轮(bare)vs 末轮(带 doc)===")
    f_sr, l_sr, f_sc, l_sc = [], [], [], []
    improved = worse = same = 0
    for t in tasks:
        by_g: dict[str, list[dict]] = defaultdict(list)
        order: list[str] = []
        for ep in t["eps"]:
            g = str((ep.get("trajectory") or {}).get("guidance_hash", "?"))
            if g not in by_g:
                order.append(g)
            by_g[g].append(ep)
        if len(order) < 2:
            continue
        fe, le = by_g[order[0]], by_g[order[-1]]
        f_sr.append(statistics.mean(float(won_of(e)) for e in fe))
        l_sr.append(statistics.mean(float(won_of(e)) for e in le))
        fs = statistics.mean(score_of(e) for e in fe)
        ls = statistics.mean(score_of(e) for e in le)
        f_sc.append(fs); l_sc.append(ls)
        if ls - fs > 1e-9:
            improved += 1
        elif ls - fs < -1e-9:
            worse += 1
        else:
            same += 1
    if f_sr:
        print(f"  可比会话 {len(f_sr)} 个(至少两个 guidance 版本)")
        print(f"  Succ.  首轮 {statistics.mean(f_sr):6.2%} → "
              f"末轮 {statistics.mean(l_sr):6.2%}  "
              f"(Δ {statistics.mean(l_sr)-statistics.mean(f_sr):+.2%})")
        print(f"  Score  首轮 {statistics.mean(f_sc)*100:6.1f} → "
              f"末轮 {statistics.mean(l_sc)*100:6.1f}  "
              f"(Δ {(statistics.mean(l_sc)-statistics.mean(f_sc))*100:+.1f})")
        print(f"  会话级方向: {improved} 改善 / {worse} 变差 / {same} 持平")

    # ---- 各轮次曲线 -----------------------------------------------------
    print("\n=== 各实验轮次的平均成绩(跨全部会话)===")
    per_r: dict[int, list[float]] = defaultdict(list)
    per_w: dict[int, list[float]] = defaultdict(list)
    for t in tasks:
        seen: list[str] = []
        for ep in t["eps"]:
            g = str((ep.get("trajectory") or {}).get("guidance_hash", "?"))
            if g not in seen:
                seen.append(g)
            per_r[len(seen)].append(score_of(ep))
            per_w[len(seen)].append(float(won_of(ep)))
    print(f"  {'轮次':>4} {'n':>5} {'Score':>8} {'Succ.':>8}")
    for r in sorted(per_r):
        print(f"  {r:>4} {len(per_r[r]):>5} {statistics.mean(per_r[r])*100:>8.1f} "
              f"{statistics.mean(per_w[r]):>8.2%}")

    # ---- 会话统计 -------------------------------------------------------
    print("\n=== 会话统计 ===")
    ex = [t["meta"].get("experiments_used", 0) for t in tasks]
    epc = [t["meta"].get("episodes_consumed", 0) for t in tasks]
    dw = [t["meta"].get("doc_words", 0) for t in tasks]
    fin = sum(1 for t in tasks if t["meta"].get("finalized"))
    nerr = sum(1 for t in tasks if t["meta"].get("session_error"))
    ls = [(t["meta"].get("last_exp_successes", 0),
           t["meta"].get("last_exp_attempts", 0)) for t in tasks]
    ts_, ta_ = sum(s for s, _ in ls), sum(x for _, x in ls)
    print(f"  finalized {fin}/{len(tasks)};session_error {nerr}")
    print(f"  实验次数 p50={statistics.median(ex):.0f} "
          f"(耗尽 6 次占 {sum(1 for x in ex if x >= 6)/len(ex):.0%})")
    print(f"  episodes p50={statistics.median(epc):.0f} / "
          f"doc 词数 p50={statistics.median(dw):.0f}")
    print(f"  ★ 末轮实验合计 {ts_}/{ta_} = {ts_/max(1,ta_):.1%} "
          f"—— researcher 收官时的真实命中率")

    # ---- ② 强断言抽查 ---------------------------------------------------
    print("\n=== ② doc 强断言抽查(需人工核验事实正确性)===")
    cc: Counter[str] = Counter()
    n_with = 0
    for t in tasks:
        hit = False
        for line in t["doc"].splitlines():
            s = line.strip()
            if len(s) > 40 and STRONG.search(s):
                cc[s[:160]] += 1
                hit = True
        n_with += int(hit)
    print(f"  {n_with}/{len(tasks)} 份 doc 含强断言;句子 "
          f"{sum(cc.values())} 条(去重 {len(cc)})")
    for s, c in cc.most_common(6):
        print(f"   ×{c}  {s}")

    print(f"\n=== doc 结构抽样(前 {a.docs} 份)===")
    for t in tasks[:a.docs]:
        m = t["meta"]
        heads = [l.strip() for l in t["doc"].splitlines() if l.strip().startswith("#")]
        print(f"  {m['task_id']}: {m.get('doc_words')} 词 / 实验 "
              f"{m.get('experiments_used')} / 末轮 "
              f"{m.get('last_exp_successes')}/{m.get('last_exp_attempts')}")
        print(f"     {heads}")

    if a.full_doc:
        for t in tasks:
            if a.full_doc in t["meta"]["task_id"]:
                print(f"\n{'=' * 70}\n完整 doc: {t['meta']['task_id']}\n")
                print(t["doc"])
                print(f"\n--- self_assessment ---\n{t['meta'].get('self_assessment')}")
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
