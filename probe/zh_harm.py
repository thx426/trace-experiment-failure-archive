#!/usr/bin/env python3
"""量化"模型自发中文"的实际危害:代码执行失败率 与 任务成功率 的中英对照。
用法: zh_harm.py <run_dir> [--since-min 120]"""
import json, glob, os, re, sys, time

CJK = re.compile(r"[一-鿿]")
run = sys.argv[1].rstrip("/")
since_min = int(sys.argv[sys.argv.index("--since-min") + 1]) if "--since-min" in sys.argv else 120
cutoff = time.time() - since_min * 60

# 代码层:含中文 vs 纯英文的执行失败率
code_zh = [0, 0]      # [失败, 总数]
code_en = [0, 0]
# 轨迹层:thought 含中文 vs 不含的胜率
traj_zh = [0, 0]
traj_en = [0, 0]
syntax_zh = 0         # 明确因中文字符导致的 SyntaxError

steps = [d for d in sorted(glob.glob(os.path.join(run, "rollouts", "step_*"))) if "val" not in d][-3:]
for d in steps:
    for f in glob.glob(os.path.join(d, "*.json")):
        if os.path.getsize(f) == 0 or os.path.getmtime(f) < cutoff:
            continue
        try:
            r = json.load(open(f))
        except Exception:
            continue
        steps_ = (r.get("trajectory") or {}).get("steps") or []
        th_all = " ".join(str(s.get("thought") or "") for s in steps_)
        bucket = traj_zh if len(CJK.findall(th_all)) >= 3 else traj_en
        bucket[1] += 1
        if r.get("won"):
            bucket[0] += 1
        for s in steps_:
            ar = s.get("action_raw") or ""
            if "run_python" not in ar:
                continue
            try:
                code = json.loads(ar).get("arguments", {}).get("code", "")
            except Exception:
                code = ar
            obs = s.get("observation") or ""
            failed = "[exit" in obs or "timeout after" in obs
            b = code_zh if CJK.search(code) else code_en
            b[1] += 1
            if failed:
                b[0] += 1
            if failed and ("invalid character" in obs or "SyntaxError" in obs) and CJK.search(code):
                syntax_zh += 1

print(f"=== 代码执行失败率(近 {since_min} 分钟,最后 3 训练步)===")
for name, b in (("代码含中文", code_zh), ("代码纯英文", code_en)):
    if b[1]:
        print(f"  {name}: {b[0]}/{b[1]} = {100*b[0]/b[1]:.1f}% 失败")
if code_zh[1] and code_en[1]:
    d = 100*code_zh[0]/code_zh[1] - 100*code_en[0]/code_en[1]
    print(f"  ⇒ 中文代码的失败率高 {d:+.1f}pp")
print(f"  其中明确因中文字符触发 SyntaxError/invalid character: {syntax_zh} 次")

print(f"\n=== 任务成功率(按 thought 语言分组)===")
for name, b in (("推理含中文", traj_zh), ("推理纯英文", traj_en)):
    if b[1]:
        print(f"  {name}: {b[0]}/{b[1]} = {100*b[0]/b[1]:.1f}% 胜")
if traj_zh[1] and traj_en[1]:
    d = 100*traj_zh[0]/traj_zh[1] - 100*traj_en[0]/traj_en[1]
    print(f"  ⇒ 中文推理的胜率{'高' if d>0 else '低'} {abs(d):.1f}pp")
    print("  ⚠ 这是观察性对照,非因果:模型可能在'觉得难'的题上更易切中文(混杂因素)")
