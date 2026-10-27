#!/usr/bin/env python3
"""SearchQA parquet 全面排查:逐项对照 ALFWorld(已跑通 444 步的活样本)。
目的是找出所有会让 verl trainer 走错路径的字段差异,不只已知那两个。"""
import pandas as pd, json, collections

ALF = "/fs/fast/u2021000902/wushang/data/classic/classic_alfworld_data/"
SQA = "/fs/fast/u2021000902/wushang/data/classic/classic_searchqa_data/"

a_tr = pd.read_parquet(ALF + "train_3553.parquet")
a_va = pd.read_parquet(ALF + "valid_seen.parquet")
s_tr = pd.read_parquet(SQA + "train.parquet")
s_va = pd.read_parquet(SQA + "test.parquet")

def show(tag, d):
    print(f"--- {tag}  n={len(d)}")
    print(f"    columns      : {list(d.columns)}")
    r0 = d.iloc[0]
    print(f"    data_source  : {r0['data_source']!r}")
    if "agent_name" in d.columns:
        print(f"    agent_name   : {r0['agent_name']!r}")
    else:
        print(f"    agent_name   : ** 缺失 **")
    print(f"    reward_model : {r0['reward_model']!r}"[:150])
    ei = r0["extra_info"]
    print(f"    extra_info   : keys={sorted(ei.keys()) if hasattr(ei,'keys') else type(ei)}")
    p = r0["prompt"]
    try:
        n = len(p); roles = [m.get("role") for m in p]
        print(f"    prompt       : list[{n}] roles={roles}")
        print(f"                   [0] {str(p[0].get('content'))[:90]}…")
    except Exception:
        print(f"    prompt       : {type(p)} {str(p)[:90]}")

print("=========== ALFWorld(对照:已跑通)")
show("train", a_tr); show("valid_seen", a_va)
print()
print("=========== SearchQA(待修)")
show("train", s_tr); show("test", s_va)

print()
print("=========== 差异清单")
issues = []
for nm, d in (("sqa_train", s_tr), ("sqa_test", s_va)):
    if "agent_name" not in d.columns:
        issues.append(f"{nm}: 缺 agent_name 列(ALFWorld='trace_opd';verl 据此选 agent loop)")
    st = d.iloc[0]["reward_model"].get("style")
    if st != "env":
        issues.append(f"{nm}: reward_model.style={st!r} 应为 'env'(否则 verl 走 rule 分派 → data_source 未注册 → NotImplementedError)")
    cols_a = set(a_tr.columns); cols_s = set(d.columns)
    miss = cols_a - cols_s
    extra = cols_s - cols_a
    if miss: issues.append(f"{nm}: 相比 ALFWorld 缺列 {sorted(miss)}")
    if extra: issues.append(f"{nm}: 相比 ALFWorld 多列 {sorted(extra)}")
    # extra_info 必需键(adapter 消费)
    need = {"task_id", "task_type", "split", "question", "golden_answers", "initial_obs"}
    have = set(d.iloc[0]["extra_info"].keys())
    if need - have:
        issues.append(f"{nm}: extra_info 缺 {sorted(need-have)}")
for i in issues:
    print("  ✗", i)
if not issues:
    print("  ✓ 无差异")
