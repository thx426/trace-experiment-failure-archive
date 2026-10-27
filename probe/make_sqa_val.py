#!/usr/bin/env python3
"""SearchQA val 子集:test 集 shuffle 后固定取 N 条(用户裁定 2026-08-13,
与 SEED 的 val_batch_size=512 对齐,取 500)。

固定 seed shuffle 一次后落盘为固定集合 —— 若每次 val 重新采样,跨 step
的读数就不是同一批任务,曲线不可比。
"""
import sys, pandas as pd, collections

SRC = "/fs/fast/u2021000902/wushang/data/classic/classic_searchqa_data/test.parquet"
OUT = "/fs/fast/u2021000902/wushang/data/classic/classic_searchqa_data/val_shuf500.parquet"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
SEED = 42

d = pd.read_parquet(SRC)
print(f"test 全量: {len(d)}")
src_all = collections.Counter(d["data_source"])
print("  来源分布:", dict(src_all))

sub = d.sample(n=N, random_state=SEED).reset_index(drop=True)
sub.to_parquet(OUT, index=False)
print(f"\n已写 {OUT}  n={len(sub)}  (seed={SEED})")
src_sub = collections.Counter(sub["data_source"])
print("  子集来源分布:")
for k in sorted(src_all):
    exp = N * src_all[k] / len(d)
    print(f"    {k:<20} {src_sub.get(k,0):>4}  (按比例期望 {exp:.1f})")
