#!/usr/bin/env python3
"""SearchQA 文本协议单测(真实打 162 检索)。
覆盖:标签解析 / 判分复用 / <information> 包裹 / 格式提醒 / 与 FC 档判分等价。"""
import sys
sys.path.insert(0, "/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM/src")
from trace_opd.common.config import SearchQACfg
from trace_opd.envs.searchqa import adapter as A

ok_all = True


def chk(name, cond, extra=""):
    global ok_all
    ok_all &= bool(cond)
    print(f"  {'✓' if cond else '✗'} {name}{('  ' + extra) if extra else ''}")


print("=== ① 标签解析 ===")
chk("<search> 提取", A.parse_text_action("<think>t</think>\n<search> paris capital </search>")
    == {"name": "search", "arguments": {"query": "paris capital"}})
chk("<answer> 提取", A.parse_text_action("<think>t</think><answer> Beijing </answer>")
    == {"name": "submit", "arguments": {"answer": "Beijing"}})
chk("answer 优先于 search", (A.parse_text_action("<search>q</search><answer>A</answer>") or {}).get("name") == "submit")
chk("取最后一个(模型先复述格式)",
    (A.parse_text_action("format is <search> your query </search>\n<search> real q </search>") or {})
    .get("arguments", {}).get("query") == "real q")
chk("无标签返回 None", A.parse_text_action("I think the answer is Beijing.") is None)
chk("thought 取 <think>", A.text_thought("<think>my reasoning</think><search>q</search>") == "my reasoning")
chk("thought 回落到正文", A.text_thought("plain reasoning\n<search>q</search>") == "plain reasoning")

print("\n=== ② 端到端:真实检索 + 判分 ===")
cfg = SearchQACfg(protocol="text", retriever_url="http://HOST_IP:8000",
                  topk=3, preflight=True)
fac = A.SearchQAEnvFactory(cfg)
item = {"task_id": "t1", "task_type": "nq", "split": "test",
        "question": "What is the capital of France?",
        "golden_answers": ["Paris"], "initial_obs": "What is the capital of France?"}
ad = fac.adapter_from_item(item)
env = ad.open_episode()

chk("scaffold 用文本模板", "<search>" in ad.executor_scaffold() and "<answer> Beijing </answer>" in ad.executor_scaffold())
chk("tools() 为 None(不渲染 FC 语法)", ad.tools() is None)

out = ad.react(env, "<think>I should look this up.</think>\n<search> capital of France </search>")
chk("search 被执行", out.acted and out.valid and not out.done)
chk("观察用 <information> 包裹", "<information>" in out.feedback and "</information>" in out.feedback,
    repr(out.feedback[:70]))
chk("检索到 Paris", "Paris" in out.feedback)

out2 = ad.react(env, "<think>Now I know.</think>\n<answer> Paris </answer>")
chk("answer 判对 (score=1)", out2.done and out2.score == 1.0)
out3 = ad.react(env, "<think>guess</think>\n<answer> Lyon </answer>")
chk("错答判 0", out3.done and out3.score == 0.0)
chk("判分不反馈(信息状态红线)", "Submitted." in out3.feedback and "correct" not in out3.feedback.lower())

out4 = ad.react(env, "I believe it is Paris but forgot the tags.")
chk("无标签 ⇒ 不消耗 env step", (not out4.acted) and (not out4.consumed_env_step))
chk("提醒含两种标签写法", "<search>" in out4.feedback and "<answer>" in out4.feedback)

print("\n=== ③ replay(前缀续写依赖)===")
fb, done = ad.replay_action(env, out.action_norm)
chk("replay 走 <information> 包裹", "<information>" in fb and not done)

print("\n" + ("全部通过 ✓" if ok_all else "有失败项 ✗"))
sys.exit(0 if ok_all else 1)
