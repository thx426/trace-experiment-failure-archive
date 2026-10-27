#!/usr/bin/env python3
"""math 文本 TIR 协议单元测试(本地,不依赖 math_verify/GPU)。"""
import sys, threading
sys.path.insert(0, "/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM/src")

from trace_opd.envs.math import adapter as M

fails = []
def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  {detail}"))
    if not cond:
        fails.append(name)

# ---- extract_boxed:嵌套花括号 ----
check("boxed 简单", M.extract_boxed(r"answer \boxed{42}.") == "42")
check("boxed 嵌套", M.extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}")
check("boxed 取最后", M.extract_boxed(r"\boxed{1} then \boxed{2}") == "2")
check("boxed 不闭合→None", M.extract_boxed(r"\boxed{\frac{1}{2}") is None)
check("boxed 无→None", M.extract_boxed("no answer") is None)
check("boxed 深嵌套", M.extract_boxed(r"\boxed{a_{n}^{2}+\sqrt{b_{k}}}") == r"a_{n}^{2}+\sqrt{b_{k}}")

# ---- parse_text_action:优先级与取块 ----
code_turn = "reasoning...\n```python\nprint(1+1)\n```\n"
a = M.parse_text_action(code_turn)
check("代码块解析", a == {"name": "run_python", "arguments": {"code": "print(1+1)"}}, str(a))

both = "```python\nprint(9)\n```\nso the answer should be \\boxed{9}"
a = M.parse_text_action(both)
check("代码>boxed(防幻觉输出)", a["name"] == "run_python", str(a))

multi = "```python\nprint(1)\n```\ntry again\n```python\nprint(2)\n```"
a = M.parse_text_action(multi)
check("多块取最后", a["arguments"]["code"] == "print(2)", str(a))
check("extra_note 计数", "1 earlier code block" in M.text_extra_note(multi))

ans_turn = "The final answer is \\boxed{2278}."
a = M.parse_text_action(ans_turn)
check("终答解析", a == {"name": "submit", "arguments": {"answer": "2278"}}, str(a))

check("void turn→None", M.parse_text_action("hmm let me think") is None)
check("裸围栏不收(须 ```python 标注)", M.parse_text_action("```\nprint(1)\n```") is None)
check("py 标注也收", M.parse_text_action("```py\nprint(3)\n```")["arguments"]["code"] == "print(3)")

# ---- thought 记账 ----
th = M.text_thought(code_turn, {"name": "run_python"})
check("代码轮 thought=围栏前正文", th == "reasoning...", repr(th))
check("终答轮 thought=全文", M.text_thought(ans_turn, {"name": "submit"}) == ans_turn.strip())

# ---- 观察包裹 ----
check("obs 围栏", M.text_obs_wrap("42") == "```output\n42\n```")

# ---- 端到端 react(真 subprocess;text 模式)----
class Cfg:  # 最小 MathCfg 替身
    protocol = "text"
    python_timeout_s = 10
    obs_max_chars = 3500
    exec_concurrency = 4

item = {"task_id": "t1", "task_type": "math", "split": "valid",
        "problem": "What is 7^4 - 3*41?", "ground_truth": "2278",
        "initial_obs": "What is 7^4 - 3*41?"}
ad = M.MathTaskAdapter(item, Cfg(), threading.Semaphore(2))
env = ad.open_episode()
try:
    o1 = ad.react(env, "verify:\n```python\nprint(7**4 - 3*41)\n```")
    check("react 代码轮 feedback", o1.feedback.startswith("```output\n2278"), repr(o1.feedback[:60]))
    check("react 代码轮语义", o1.acted and o1.valid and not o1.done and o1.consumed_env_step)
    o2 = ad.react(env, "The final answer is \\boxed{2278}.")
    check("react 终答判分", o2.done and o2.score == 1.0, f"done={o2.done} score={o2.score}")
    o3 = ad.react(env, "let me think more")
    check("react void turn", (not o3.acted) and (not o3.consumed_env_step)
          and "```python" in o3.feedback)
    o4 = ad.react(env, "```python\nraise ValueError('boom')\n```")
    check("react 错误回喂", o4.feedback.startswith("```output\n[exit 1]")
          and "ValueError: boom" in o4.feedback and not o4.valid, repr(o4.feedback[:80]))
    # scaffold / tools / reminder 的协议切换
    check("scaffold=TEXT", ad.executor_scaffold() is M.SYSTEM_PROMPT_TEXT)
    check("tools()=None", ad.tools() is None)
    check("format_reminder=text", "```python" in ad.format_reminder())
    check("task_brief 文本措辞", "```python" in ad.task_brief() and "run_python(code)" not in ad.task_brief())
    # replay 等价
    fb, done = ad.replay_action(env, o1.action_norm)
    check("replay 同型", fb.startswith("```output\n2278") and not done, repr(fb[:40]))
finally:
    env.close()

# ---- FC 档回归(默认协议不变)----
class CfgFC(Cfg):
    protocol = "fncall"
ad_fc = M.MathTaskAdapter(item, CfgFC(), threading.Semaphore(2))
check("FC scaffold 不变", ad_fc.executor_scaffold() is M.SYSTEM_PROMPT)
check("FC tools 保留", ad_fc.tools() is M.TOOLS)

print(f"\n{'ALL PASS' if not fails else 'FAILURES: ' + str(fails)} ({'OK' if not fails else 'BAD'})")
sys.exit(1 if fails else 0)
