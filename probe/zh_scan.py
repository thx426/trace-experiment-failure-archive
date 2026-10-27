#!/usr/bin/env python3
"""扫描 rollout/研究产物里的中文,按字段归因(区分"我们的提示词"与"模型自发")。
用法: zh_scan.py <run_dir> [--since-min 60]"""
import json, glob, os, re, sys, collections

CJK = re.compile(r"[一-鿿]")
run = sys.argv[1].rstrip("/")
since_min = 60
if "--since-min" in sys.argv:
    since_min = int(sys.argv[sys.argv.index("--since-min") + 1])
import time
cutoff = time.time() - since_min * 60


def zh_ratio(s):
    s = str(s or "")
    if not s:
        return 0.0, 0
    n = len(CJK.findall(s))
    return n / max(len(s), 1), n


# ---- 1) rollout 轨迹:按字段统计 ----
field_hits = collections.Counter()
field_total = collections.Counter()
samples = collections.defaultdict(list)

steps = sorted(glob.glob(os.path.join(run, "rollouts", "step_*")))
steps = [d for d in steps if "val" not in d][-3:]
for d in steps:
    for f in glob.glob(os.path.join(d, "*.json")):
        if os.path.getsize(f) == 0 or os.path.getmtime(f) < cutoff:
            continue
        try:
            r = json.load(open(f))
        except Exception:
            continue
        for s in (r.get("trajectory") or {}).get("steps") or []:
            for key in ("thought", "action_raw", "observation"):
                v = s.get(key) or ""
                field_total[key] += 1
                ratio, n = zh_ratio(v)
                if n >= 3:
                    field_hits[key] += 1
                    if len(samples[key]) < 3:
                        m = CJK.search(v)
                        samples[key].append(v[max(0, m.start()-60):m.start()+120])
        # research doc 字段
        doc = (r.get("research") or {}).get("research_doc") or ""
        if doc:
            field_total["research_doc"] += 1
            ratio, n = zh_ratio(doc)
            if n >= 3:
                field_hits["research_doc"] += 1
                if len(samples["research_doc"]) < 2:
                    m = CJK.search(doc)
                    samples["research_doc"].append(doc[max(0, m.start()-60):m.start()+120])

print(f"=== rollout 字段中文命中(近 {since_min} 分钟,最后 3 个训练步)===")
for k in ("thought", "action_raw", "observation", "research_doc"):
    t, h = field_total[k], field_hits[k]
    if t:
        print(f"  {k:>14}: {h}/{t} = {100*h/t:.1f}% 含中文(≥3 字)")

# ---- 2) researcher 产物(doc.md / session)----
docs = glob.glob(os.path.join(run, "research", "*", "*", "doc.md"))[-200:]
dh = 0
for f in docs:
    try:
        s = open(f, encoding="utf-8", errors="replace").read()
    except Exception:
        continue
    _, n = zh_ratio(s)
    if n >= 3:
        dh += 1
        if len(samples["doc.md"]) < 2:
            m = CJK.search(s)
            samples["doc.md"].append(s[max(0, m.start()-60):m.start()+120])
if docs:
    print(f"  {'doc.md 文件':>14}: {dh}/{len(docs)} = {100*dh/len(docs):.1f}% 含中文")

# ---- 3) narration(tape)----
nh = nt = 0
for f in glob.glob(os.path.join(run, "tape", "narrator.*.jsonl"))[:6]:
    try:
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            nt += 1
            out = json.dumps(r.get("response") or r.get("text") or "", ensure_ascii=False)
            _, n = zh_ratio(out)
            if n >= 3:
                nh += 1
                if len(samples["narration"]) < 2:
                    m = CJK.search(out)
                    samples["narration"].append(out[max(0, m.start()-60):m.start()+120])
            if nt > 3000:
                break
    except Exception:
        pass
if nt:
    print(f"  {'narration':>14}: {nh}/{nt} = {100*nh/nt:.1f}% 含中文")

print("\n=== 样本(定位来源)===")
for k, v in samples.items():
    for s in v:
        print(f"  [{k}] …{s.strip()[:170]}…")
