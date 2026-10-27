#!/usr/bin/env python3
"""验证 agent_num_workers 从 128 降到 16 后,机制没坏。

推理说"同 task 的 n 个副本仍落同一 worker",但推理不算证据。本脚本用
rollout 产物直接检验三件事:
  ① 每个 task 恰好 8 个副本(切块没把组切碎)
  ② 同 task 的副本共享同一份 doc(guidance_hash 一致)—— 若副本被分到不同
     worker,各 worker 会独立跑 researcher,hash 必然不同
  ③ steered/bare 分臂是 4+4(steered_fraction=0.5 的组内配对)
用法: verify_grouping.py <rollouts/step_XXXXX 目录>
"""
import json, glob, sys, os, collections

d = sys.argv[1]
recs = []
for f in glob.glob(os.path.join(d, "*.json")):
    r = json.load(open(f))
    recs.append(r)
print(f"n={len(recs)}  目录={os.path.basename(d)}")
if not recs:
    sys.exit(1)

by_task = collections.defaultdict(list)
for r in recs:
    by_task[r.get("task_id")].append(r)

# ① 每 task 副本数
sizes = collections.Counter(len(v) for v in by_task.values())
print(f"\n① 每 task 副本数分布: {dict(sizes)}  "
      f"{'PASS(全 8)' if set(sizes) == {8} else 'FAIL —— 组被切碎'}")

# ② 同 task 是否共享同一 doc
bad_doc = []
for t, rs in by_task.items():
    hs = {(r.get("trajectory") or {}).get("guidance_hash") for r in rs}
    if len(hs) > 1:
        bad_doc.append((t, hs))
print(f"② 同 task 共享同一 doc: {len(by_task)-len(bad_doc)}/{len(by_task)} "
      f"{'PASS' if not bad_doc else 'FAIL'}")
for t, hs in bad_doc[:3]:
    print(f"     ✗ {t}: {hs}")

# ③ steered/bare 分臂
arm_dist = collections.Counter()
for t, rs in by_task.items():
    st = sum(1 for r in rs if (r.get("steering") or {}).get("steer_arm") == 1)
    ba = sum(1 for r in rs if (r.get("steering") or {}).get("steer_arm") == 0)
    arm_dist[(st, ba)] += 1
if arm_dist:
    print(f"③ (steered, bare) 分布: {dict(arm_dist)}  "
          f"{'PASS(全 4+4)' if set(arm_dist) == {(4, 4)} else '注意:非 4+4'}")
else:
    print("③ 无 steering 字段(基线臂,符合预期)")

# ④ researcher 会话是否每 task 一次(机制臂)
res = [r for r in recs if (r.get("research") or {}).get("research_experiments") is not None]
if res:
    per_task = collections.Counter(r.get("task_id") for r in res)
    print(f"④ 带 research 记账的行数={len(res)};每 task 行数分布="
          f"{dict(collections.Counter(per_task.values()))}")
