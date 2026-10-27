#!/usr/bin/env python3
"""渲染 ALFWorld text 协议的首轮 prompt,检查 '<' 是否幸存 + gen_prefix 内容。"""
import sys
sys.path.insert(0, "/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM/src")

from transformers import AutoTokenizer
M = "/fs/fast/u2021000902/wushang/model/Qwen/Qwen2.5-7B-Instruct"
tok = AutoTokenizer.from_pretrained(M)

from trace_opd.envs.alfworld import adapter as A

sysp = A.SYSTEM_PROMPT_BARE
print("=" * 72)
print("SYSTEM_PROMPT_BARE 尾段(输出契约部分):")
print("=" * 72)
print(repr(sysp[-420:]))
print()
print("  '<reasoning>' 在源串中:", "<reasoning>" in sysp)
print("  '<action>'    在源串中:", "<action>" in sysp)
print()

msgs = [{"role": "system", "content": sysp},
        {"role": "user", "content": "You are in the middle of a room. Your task is to: look at alarmclock under the desklamp."}]

rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print("=" * 72)
print("apply_chat_template 渲染后的尾段 300 字符:")
print("=" * 72)
print(repr(rendered[-300:]))
print()
print("  渲染后含 '<reasoning>':", "<reasoning>" in rendered)
print("  渲染后含 '<action>':   ", "<action>" in rendered)
print("  渲染以什么结尾:        ", repr(rendered[-40:]))
print()

ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
print("  末 8 个 token:", [(i, tok.convert_ids_to_tokens([i])[0]) for i in ids[-8:]])
print()

# TemplateDelta 的 gen_prefix
try:
    from trace_opd.core.template_delta import TemplateDelta
    d = TemplateDelta(tok)
    print("=" * 72)
    print("TemplateDelta.gen_prefix:", [(i, tok.convert_ids_to_tokens([i])[0]) for i in d.gen_prefix])
    print("=" * 72)
except Exception as e:
    print("TemplateDelta 构造失败:", type(e).__name__, e)
