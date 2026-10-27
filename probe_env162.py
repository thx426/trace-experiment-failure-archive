"""建 OPD 显微镜前的三项探查:tokenizer 路径/可用 python 环境/tape 语义。"""
import glob, json, os, subprocess, sys

W = "/data3/wushang/workspace/TRACE-OPD"
R = W + "/trace_opd/runs/trace_opd_4b_rkl"

print("=== 1. effective_config 里的模型路径 ===")
for line in open(R + "/effective_config.yaml"):
    if any(k in line for k in ("model", "path", "tokenizer")) and "/" in line:
        print("  ", line.rstrip()[:120])

print("\n=== 2. 工作区可用 python 环境 ===")
for p in glob.glob(W + "/env*/bin/python") + glob.glob(W + "/*/bin/python"):
    print("  ", p)

print("\n=== 3. tape 一行的语义 ===")
f = glob.glob(R + "/microscope/rank0.jsonl")[0]
row = json.loads(open(f).readline())
ids, kl, mask = row["ids"], row["kl"], row["mask"]
print(f"  resp_len={row['resp_len']} len(ids)={len(ids)} len(kl)={len(kl)} len(mask)={len(mask)}")
print(f"  mask=0 占比 {1 - sum(mask)/len(mask):.1%}(灰=环境反馈?)")
nz = [x for x, m in zip(kl, mask) if m]
print(f"  mask=1 的 kl: 均值 {sum(nz)/len(nz):.4f} / >1 占比 {sum(1 for x in nz if x>1)/len(nz):.1%}")

# 与 dump 的 join 验证
tape_h = {json.loads(l)["resp_hash"] for l in open(f)}
hit = None
for fp in glob.glob(R + "/rollouts/step_00001/*.json"):
    d = json.load(open(fp))
    if int(d.get("response_token_hash") or 0) in tape_h:
        hit = d; break
if hit:
    print(f"\n=== 4. join 成功样例 ===")
    print(f"  task={hit['task_id'][:60]} resp_token_len={hit.get('response_token_len')}")
    print(f"  turn_token_spans[:3]={hit.get('turn_token_spans', [])[:3]}")
