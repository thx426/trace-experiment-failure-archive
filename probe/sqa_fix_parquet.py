#!/usr/bin/env python3
"""SearchQA parquet 修复(2026-08-13,冒烟拦下):补 verl trainer 需要的两个字段。

根因:SearchQA 数据构建期只服务过探针(走独立 evalm 路径,不经 verl trainer),
故缺两项 trainer 侧约定,ALFWorld 有:
  1) agent_name='trace_opd' —— verl 据此选 agent loop(train/agent_loop_config.yaml
     注册名);缺失会落到默认 loop。
  2) reward_model.style='env' —— 告诉 verl「reward 由环境/AgentLoop 提供」,绕过
     rule 分派;原值 'rule' 会去查 verl reward_score 注册表,而我们的
     data_source('searchqa' / 'searchqa/nq' …)不在其中 ⇒ NotImplementedError。
     判分本身不受影响:EM 判分内嵌 adapter,读的是 extra_info.golden_answers,
     reward_model.ground_truth 只是冗余副本(保留不动)。

原始文件不改(保留可追溯);产出带 _trainer 后缀的训练用文件。
"""
import sys, pandas as pd, collections

BASE = "/fs/fast/u2021000902/wushang/data/classic/classic_searchqa_data/"
VAL_N = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
SEED = 42


def fix(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["agent_name"] = "trace_opd"
    d["reward_model"] = d["reward_model"].map(
        lambda rm: {**dict(rm), "style": "env"}
    )
    # 列序与 ALFWorld 对齐(纯整洁,verl 不依赖列序)
    return d[["prompt", "data_source", "agent_name", "reward_model", "extra_info"]]


# ---- 训练集 ----
tr = pd.read_parquet(BASE + "train.parquet")
tr_fixed = fix(tr)
tr_fixed.to_parquet(BASE + "train_trainer.parquet", index=False)
print(f"train_trainer.parquet  n={len(tr_fixed)}  "
      f"style={tr_fixed['reward_model'].iloc[0]['style']!r}  "
      f"agent_name={tr_fixed['agent_name'].iloc[0]!r}")

# ---- val:test 集 shuffle 固定取 N(用户裁定 2026-08-13:固定则取大,2048)----
te = pd.read_parquet(BASE + "test.parquet")
va = te.sample(n=VAL_N, random_state=SEED).reset_index(drop=True)
va_fixed = fix(va)
out = BASE + f"val_shuf{VAL_N}.parquet"
va_fixed.to_parquet(out, index=False)
print(f"{out.split('/')[-1]}  n={len(va_fixed)}  seed={SEED}")
c = collections.Counter(va_fixed["data_source"])
tot = collections.Counter(te["data_source"])
print("  来源分布(实际 / 按比例期望):")
for k in sorted(tot):
    print(f"    {k:<28} {c.get(k,0):>5} / {VAL_N*tot[k]/len(te):7.1f}")

# ---- 自检 ----
print("\n自检:")
for nm, d in (("train_trainer", tr_fixed), (f"val_shuf{VAL_N}", va_fixed)):
    ok = (
        "agent_name" in d.columns
        and set(d["agent_name"]) == {"trace_opd"}
        and all(rm["style"] == "env" for rm in d["reward_model"].head(1000))
        and {"task_id", "question", "golden_answers", "initial_obs"}
           <= set(d["extra_info"].iloc[0].keys())
    )
    print(f"  {nm}: {'PASS' if ok else 'FAIL'}")
