#!/usr/bin/env python3
"""对比各 parquet 的 data_source(定位 reward 分派失败根因)。"""
import pandas as pd, collections
base = "/fs/fast/u2021000902/wushang/data/classic/classic_searchqa_data/"
for f in ["train", "valid", "test", "valid_strat63", "val_shuf500"]:
    try:
        d = pd.read_parquet(base + f + ".parquet")
    except Exception as e:
        print(f"{f:16s} 读取失败: {e}")
        continue
    c = collections.Counter(d["data_source"].tolist())
    print(f"{f:16s} n={len(d):>6}  data_source={dict(list(c.items())[:8])}")

# 顺带看 ALFWorld 的,作为"能跑通"的对照
alf = "/fs/fast/u2021000902/wushang/data/classic/classic_alfworld_data/"
for f in ["train_3553", "valid_seen"]:
    try:
        d = pd.read_parquet(alf + f + ".parquet")
        c = collections.Counter(d["data_source"].tolist())
        print(f"[ALF] {f:10s} n={len(d):>6}  data_source={dict(list(c.items())[:4])}")
    except Exception as e:
        print(f"[ALF] {f}: {e}")

# 看一行 extra_info 结构
d = pd.read_parquet(base + "test.parquet").head(1)
print("\ntest 首行 extra_info keys:", list(d["extra_info"].iloc[0].keys()))
print("test 首行 reward_model:", d["reward_model"].iloc[0])
