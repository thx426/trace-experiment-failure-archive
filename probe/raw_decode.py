#!/usr/bin/env python3
"""解码 rollout 的 response_token_ids,看模型真实输出的原始文本。"""
import json, glob, sys, os

d = sys.argv[1]
model = sys.argv[2] if len(sys.argv) > 2 else "/fs/fast/u2021000902/wushang/model/Qwen/Qwen2.5-7B-Instruct"
files = sorted(glob.glob(os.path.join(d, "*.json")))
rec = json.load(open(files[0]))
ids = rec.get("response_token_ids") or []
print("task:", rec.get("task_id"))
print("n_response_tokens:", len(ids))
print()

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

print("=" * 72)
print("原始解码(skip_special_tokens=False):")
print("=" * 72)
print(repr(tok.decode(ids, skip_special_tokens=False)))
print()
print("=" * 72)
print("逐 token(前 60 个):")
print("=" * 72)
for i, t in enumerate(ids[:60]):
    print(f"  {i:3d} {t:8d} {tok.convert_ids_to_tokens([t])[0]!r}")
print()
print("=" * 72)
print("turn_token_spans:", json.dumps(rec.get("turn_token_spans"))[:300])
print("=" * 72)
print("initial_obs:", repr((rec.get("trajectory") or {}).get("initial_obs", ""))[:600])
