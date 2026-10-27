"""bamboogle≈0 根因定位:数据侧 / 评分侧 / 行为侧 三选一。

线索:其余六个数据集全部 ≥ Search-R1 锚(musique .194≈其 .196),唯独
bamboogle(手工构建 CSV 来源,schema 与其他六源不同)≈0 —— 先验指向
数据构建/评分口径,而非能力。曲线有 0.083/0.165 两次凸起 ⇒ 不是恒零的
硬 schema bug,更像"只有部分行可判对"的天花板结构。
"""
from __future__ import annotations
import glob, json, re
from collections import Counter
from pathlib import Path

# ---- 1. 测试集里 bamboogle 行的构造 vs 健康数据集 ----------------------
log = Path("runs/classic_sqa7b_pgs.launch.log").read_text(encoding="utf-8",
                                                          errors="ignore")
m = re.search(r"data\.val_files=\['([^']+)'", log)
print("val 文件:", m.group(1) if m else "未找到")
import pandas as pd
df = pd.read_parquet(m.group(1))
print("列:", list(df.columns))
sc = next((c for c in ("data_source", "dataset", "source") if c in df.columns), None)
print("\n各数据集行数:")
print(df[sc].value_counts().to_string())

def gt_of(row):
    rm = row.get("reward_model")
    return rm

bb = df[df[sc].astype(str).str.contains("bamboogle", case=False)]
ok = df[~df[sc].astype(str).str.contains("bamboogle", case=False)]
print(f"\n--- bamboogle 前 3 行的 ground_truth 结构 ---")
for i in range(min(3, len(bb))):
    r = bb.iloc[i]
    print(f"[{i}] reward_model={str(r.get('reward_model'))[:260]}")
    q = str(r.get("prompt"))[:0]  # prompt 太长不打
print(f"\n--- 对照(非 bamboogle)首行 ---")
r = ok.iloc[0]
print(f"    src={r[sc]} reward_model={str(r.get('reward_model'))[:260]}")

# gold 形态统计:类型/元素数/是否含逗号等
def gold_list(rm):
    try:
        gt = rm.get("ground_truth") if isinstance(rm, dict) else None
        if isinstance(gt, dict):
            gt = gt.get("target") or gt.get("golden_answers") or gt.get("answer")
        if isinstance(gt, str):
            return [gt]
        return list(gt) if gt is not None else []
    except Exception as e:
        return [f"<ERR {e}>"]
shapes = Counter()
for _, r in bb.iterrows():
    g = gold_list(r.get("reward_model"))
    shapes[f"n={len(g)} 类型={type(r.get('reward_model')).__name__}"] += 1
print("\nbamboogle gold 形态分布:", dict(shapes))
samp = [gold_list(bb.iloc[i].get("reward_model")) for i in range(min(6, len(bb)))]
print("gold 样例:", samp)

# ---- 2. val dump:模型到底答了什么、评分怎么判 --------------------------
def diag_step(run, step, show=0):
    d = f"runs/{run}/rollouts/step_{step:05d}_val"
    if not Path(d).exists():
        return None
    tot = n = won = err = noans = 0
    ex = []
    for f in glob.glob(d + "/*.json"):
        raw = Path(f).read_text(encoding="utf-8", errors="ignore")
        tot += 1
        if "bamboogle" not in raw.lower():
            continue
        try:
            r = json.loads(raw)
        except Exception:
            continue
        n += 1
        if r.get("outcome") == "ENV_ERROR":
            err += 1; continue
        w = bool(r.get("won"))
        won += int(w)
        # 找模型的最终 answer 与 gold
        traj = r.get("trajectory") or {}
        steps = traj.get("steps") or []
        last = ""
        for s in reversed(steps):
            t = str(s.get("model_text") or s.get("action") or s.get("text") or "")
            if t.strip():
                last = t; break
        ans = re.findall(r"<answer>(.*?)</answer>", last, re.S)
        if not ans:
            noans += 1
        if len(ex) < show and not w:
            gold = (r.get("extra") or {}).get("ground_truth") or \
                   (r.get("reward_model") or {}) or traj.get("gold")
            ex.append((str(r.get("task_id"))[:40],
                       (ans[-1].strip()[:90] if ans else "<无 answer 标签>"),
                       str(gold)[:120],
                       last[-160:].replace("\n", "⏎")))
    return dict(step=step, files=tot, bb=n, won=won, err=err, noans=noans, ex=ex)

print("\n=== PGS 臂各 val 步的 bamboogle ===")
dirs = sorted(glob.glob("runs/classic_sqa7b_pgs/rollouts/step_*_val"))
print("val 目录:", [d.split("/")[-1] for d in dirs])
for d in dirs:
    st = int(re.search(r"step_0*(\d+)_val", d).group(1))
    r = diag_step("classic_sqa7b_pgs", st, show=(4 if st in (100, 240) else 0))
    if r and r["bb"]:
        print(f"  step {st:>3}: bb行={r['bb']} won={r['won']} "
              f"({r['won']/max(1,r['bb']-r['err']):.1%}) ENV_ERR={r['err']} "
              f"无answer标签={r['noans']}")
        for tid, a, g, tail in r["ex"]:
            print(f"     ✗ {tid}\n       模型答: {a}\n       gold : {g}")

print("\n=== GRPO 臂最后一个 val 的 bamboogle ===")
dirs = sorted(glob.glob("runs/classic_sqa7b_grpo/rollouts/step_*_val"))
if dirs:
    st = int(re.search(r"step_0*(\d+)_val", dirs[-1]).group(1))
    r = diag_step("classic_sqa7b_grpo", st, show=3)
    print(f"  step {st}: bb行={r['bb']} won={r['won']} ENV_ERR={r['err']} "
          f"无answer标签={r['noans']}")
    for tid, a, g, tail in r["ex"]:
        print(f"     ✗ {tid}\n       模型答: {a}\n       gold : {g}")
