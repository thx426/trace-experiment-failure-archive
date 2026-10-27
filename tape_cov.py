"""microscope tape 的覆盖面:哪些训练步的轨迹能查到逐词 KL。"""
import glob, json, os, re

W = "/data3/wushang/workspace/TRACE-OPD/trace_opd/runs/trace_opd_4b_rkl"
tape_hashes = set()
pat = re.compile(rb'"resp_hash":\s*(-?\d+)')
for f in sorted(glob.glob(W + "/microscope/rank*.jsonl")):
    sz = os.path.getsize(f) / 1e6
    n = 0
    with open(f, "rb") as fh:
        for line in fh:
            m = pat.search(line[:200])
            if m:
                tape_hashes.add(int(m.group(1)))
                n += 1
    print(f"{os.path.basename(f)}: {sz:.0f}MB, {n} 行")
print(f"tape 覆盖的独立轨迹数: {len(tape_hashes)}")

print(f"\n{'step':>6} {'dump轨迹':>8} {'tape命中':>8}")
for d in sorted(glob.glob(W + "/rollouts/step_*"), 
                key=lambda p: (len(p), p)):
    name = os.path.basename(d)
    hs = []
    for f in glob.glob(d + "/*.json"):
        try:
            hs.append(int(json.load(open(f)).get("response_token_hash") or 0))
        except Exception:
            pass
    hit = sum(1 for h in hs if h in tape_hashes)
    print(f"{name:>6} {len(hs):>8} {hit:>8}")
