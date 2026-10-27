#!/usr/bin/env python3
"""text 协议格式服从性 A/B:Qwen2.5-7B greedy 下各 scaffold 变体的首轮输出。
在空闲 GPU 上用 transformers 直接复现(与 SGLang 无关,排除引擎因素)。"""
import sys, re
sys.path.insert(0, "/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM/src")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

M = "/fs/fast/u2021000902/wushang/model/Qwen/Qwen2.5-7B-Instruct"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16, device_map="cuda:0")
model.eval()

from trace_opd.envs.alfworld import adapter as A
BARE = A.SYSTEM_PROMPT_BARE

OBS = ("-= Welcome to TextWorld, ALFRED! =-\n\nYou are in the middle of a room. "
       "Looking quickly around you, you see a bed 1, a desk 1, a drawer 3, a drawer 2, "
       "a drawer 1, a garbagecan 1, a laundryhamper 1, a safe 1, a shelf 2, a shelf 1, "
       "a sidetable 2, and a sidetable 1.\n\nYour task is to: look at alarmclock under the desklamp.")

ONE_SHOT_DEMO = (
    "Example of the required output format (for a different task):\n"
    "<reasoning>I need to find a mug. The countertop is a likely place, so I will go there first.</reasoning>\n"
    "<action>go to countertop 1</action>"
)

variants = {
    # A: 现行 scaffold 原样
    "A_current": [
        {"role": "system", "content": BARE},
        {"role": "user", "content": OBS},
    ],
    # B: system prompt 尾部附一条完整示例(one-shot in system)
    "B_sys_oneshot": [
        {"role": "system", "content": BARE + "\n\n" + ONE_SHOT_DEMO},
        {"role": "user", "content": OBS},
    ],
    # C: 首轮 user 消息尾部重申格式模板(相关工作 verl-agent 形态:格式要求随每轮 obs)
    "C_obs_reminder": [
        {"role": "system", "content": BARE},
        {"role": "user", "content": OBS + "\n\nRespond in exactly this format:\n<reasoning>...</reasoning>\n<action>...</action>"},
    ],
    # D: 伪首轮 assistant 示例(one-shot as a fake first exchange)
    "D_fake_turn": [
        {"role": "system", "content": BARE},
        {"role": "user", "content": "You are in the middle of a room. You see a countertop 1 and a fridge 1.\n\nYour task is to: put a mug on the countertop."},
        {"role": "assistant", "content": "<reasoning>I need to find a mug first. The fridge is a likely place for a mug, so I will check it.</reasoning>\n<action>go to fridge 1</action>"},
        {"role": "user", "content": "You arrive at fridge 1. The fridge 1 is closed.\n\n(NEW EPISODE — previous conversation is a format demonstration only.)\n\n" + OBS},
    ],
}

_ACT = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
_RSN = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL | re.IGNORECASE)

for name, msgs in variants.items():
    rendered_p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(rendered_p, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda:0")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=160, do_sample=False,
                             eos_token_id=[tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")],
                             pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    ok_r = bool(_RSN.search(text)); ok_a = bool(_ACT.search(text))
    act = _ACT.findall(text)
    print("=" * 70)
    print(f"[{name}]  reasoning_tag={'✓' if ok_r else '✗'}  action_tag={'✓' if ok_a else '✗'}  action={act[-1].strip() if act else None!r}")
    print(repr(text[:400]))
    print()
