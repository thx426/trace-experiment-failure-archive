#!/usr/bin/env python3
"""Phase3 门C-R:35B 外部研究员批量产特权文档——**走方案机制代码**
(research/session.py + tools.py + prompts.py;实验 episode 走同一
core/episode.py runner,executor = 7B SGLang /v1/completions token-ids)。

与 sftgen_researcher.py(终端域 SFT 管线)同装配,三点不同:
  * 特权注入:build_researcher_system(gt_text=按域特权素材)——强特权探针
    口径(信息状态红线只约束训练态,此处是验证探针;doc 下游仅用于 doc 臂
    评测,不做 SFT);
  * 产物面向 doc 臂:finalize 的 doc 落 <out>/docs/<task_id_safe>.md
    (run_eval.py --doc-dir 布局);会话全量 payload 落 <out>/sessions/
    (messages 逐字 + guidance_log + 实验成功率序列 = 用户观察面);
  * executor 多端点 round-robin(6×7B SGLang 分流)。

用法(A800,TERM 树 probe/ 下;PYTHONPATH 由 parents[1]/src 自解):
  python research_gen_docs.py --config configs/classic/searchqa.yaml \
    --tasks-parquet .../valid_strat63.parquet --n-tasks 14 \
    --model-path <7B> --executor-api-base http://127.0.0.1:8253/v1,... \
    --executor-model qwen25-7b \
    --researcher-api-base http://HOST_IP:8880/v1,http://HOST_IP:8881/v1 \
    --researcher-model qwen3.6-35b-a3b --out-dir runs/classic_research/sqa14
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import httpx  # noqa: E402
import pandas as pd  # noqa: E402

from trace_opd.common.config import load_config  # noqa: E402
from trace_opd.common.llm import make_client  # noqa: E402
from trace_opd.core.template_delta import TemplateDelta, render_ids  # noqa: E402
from trace_opd.evalm.run_eval import external_generate, load_tasks  # noqa: E402
from trace_opd.research.prompts import build_researcher_system  # noqa: E402
from trace_opd.research.session import ResearchSession  # noqa: E402
from trace_opd.research.toolproto import make_tool_protocol  # noqa: E402
from trace_opd.research.tools import ToolRouter  # noqa: E402


def _safe(task_id: str) -> str:
    return task_id.replace("/", "__").replace(" ", "_")


# ---------------------------------------------------------------------------
# 特权素材(gt_text)——按域构造;素材缺失的任务显式跳过并记 missing
# ---------------------------------------------------------------------------

def build_gt_text(domain: str, ei: dict, args) -> str | None:
    if domain == "math":
        return (
            f"The correct final answer to this problem is: {ei['ground_truth']}\n"
            "A successful executor run must end with submit(answer) whose value "
            "is equivalent to this reference (equivalent forms accepted by the "
            "grader)."
        )
    if domain == "searchqa":
        golden = ei.get("golden_answers")
        golden = list(golden) if golden is not None else []
        return (
            f"Accepted gold answers (submitting any one of them, in exact short "
            f"form, counts as success): {golden}"
        )
    if domain == "webshop":
        gold = ei.get("_gold") or {}
        opts = gold.get("goal_options") or {}
        return (
            "The gold target product for this instruction:\n"
            f"- a search query that surfaces it: {gold.get('query')!r}\n"
            f"- product id (asin): {gold.get('asin')}\n"
            f"- required attributes: {gold.get('attributes')}\n"
            f"- required options to select on the product page: {opts}\n"
            f"- price must be <= {gold.get('price_upper')}\n"
            "Full score requires buying this product with the required options "
            "selected within the step budget."
        )
    if domain == "alfworld":
        if not args.alf_gtwalk_dir:
            raise SystemExit("FATAL: alfworld 域必须传 --alf-gtwalk-dir(引擎 "
                             "walkthrough 素材;annotation 高层素材已证无 lift)")
        p = Path(args.alf_gtwalk_dir) / f"{_safe(ei['task_id'])}.md"
        if not p.exists():
            return None  # 引擎无 walkthrough 的任务:跳过并记 missing
        return (
            "An engine-verified playable walkthrough for this exact task "
            "(command sequence known to succeed):\n" + p.read_text(encoding="utf-8")
        )
    raise SystemExit(f"FATAL: unknown domain {domain!r}")


# ---------------------------------------------------------------------------
# 会话(装配与 sftgen 同构;narrate 用 executor server 的离线等价)
# ---------------------------------------------------------------------------

def _offline_narrate_fn(http: httpx.AsyncClient, pick_base, model: str, tokenizer):
    async def narrate(messages, temperature: float, max_tokens: int) -> str:
        prompt_ids = render_ids(tokenizer, messages, True)
        resp = await http.post(
            f"{pick_base()}/completions",
            json={"model": model, "prompt": prompt_ids,
                  "temperature": temperature, "max_tokens": max_tokens},
            timeout=600.0,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["text"]

    return narrate


async def _one_session(*, cfg, item, researcher_client, tokenizer, delta,
                       http, args, factory, out_dir: Path, pick_base) -> dict:
    adapter = factory.adapter_from_item(item)
    safe = _safe(adapter.task_id)
    sess_path = out_dir / "sessions" / f"{safe}.json"
    doc_path = out_dir / "docs" / f"{safe}.md"
    if sess_path.exists():
        return {"task_id": adapter.task_id, "skipped": True}

    gt_text = None
    if not args.no_privileged:
        gt_text = build_gt_text(cfg.environment.name, item, args)
        if gt_text is None:
            return {"task_id": adapter.task_id, "missing_gt": True}

    async def researcher_chat(messages, tools):
        return await researcher_client.chat(
            messages, tools=tools,
            max_tokens=cfg.research.researcher_max_tokens,
            route_key=adapter.task_id,
        )

    router = ToolRouter(
        adapter=adapter,
        research=cfg.research,
        narration=cfg.narration,
        generate_factory=lambda: external_generate(
            http, pick_base(), args.executor_model, tokenizer
        ),
        tokenizer=tokenizer,
        delta=delta,
        episode_response_budget=cfg.eval.response_budget,
        narrate_fn=_offline_narrate_fn(http, pick_base, args.executor_model,
                                       tokenizer),
    )
    session = ResearchSession(
        researcher_chat=researcher_chat,
        toolproto=make_tool_protocol(cfg.research.tool_protocol),
        router=router,
        system_prompt=build_researcher_system(
            task_brief=adapter.task_brief(),
            executor_model=Path(args.model_path).name,
            n_exp=cfg.research.n_exp,
            doc_max_words=cfg.research.doc_max_words,
            gt_text=gt_text,
        ),
        max_turns=cfg.research.max_turns,
        wrapup_buffer=cfg.research.wrapup_buffer,
    )
    t0 = time.time()
    try:
        result = await asyncio.wait_for(
            session.run(), timeout=cfg.research.session_timeout_s
        )
    except asyncio.TimeoutError:
        result = session.snapshot(error="session timeout")
    state = router.state
    payload = {
        "task_id": adapter.task_id,
        "family": adapter.task_type,
        "privileged": gt_text is not None,
        "messages": result.messages,
        "doc": result.doc,
        "finalized": result.finalized,
        "no_finalize": result.no_finalize,
        "error": result.error,
        "turns_used": result.turns_used,
        "experiments_used": result.experiments_used,
        "episodes_consumed": result.episodes_consumed,
        # 用户观察面:逐实验 guidance 与成败序列(成功率提升过程)
        "guidance_log": state.guidance_log,
        "exp_successes_total": state.exp_successes_total,
        "exp_attempts_total": state.exp_attempts_total,
        "wall_s": round(time.time() - t0, 1),
        "meta": {
            "researcher_model": args.researcher_model,
            "n_exp": cfg.research.n_exp,
            "max_turns": cfg.research.max_turns,
            "config": str(args.config),
            "ts": time.time(),
        },
    }
    sess_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sess_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.rename(sess_path)
    if result.finalized and (result.doc or "").strip():
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.write_text(result.doc, encoding="utf-8")
    return {"task_id": adapter.task_id, "finalized": result.finalized,
            "error": result.error,
            "exp": f"{state.exp_successes_total}/{state.exp_attempts_total}"}


async def amain(args) -> int:
    from transformers import AutoTokenizer

    from trace_opd.core import protocol
    from trace_opd.envs.registry import make_env_factory

    cfg = load_config(args.config)
    protocol.configure_episode_caps(
        cfg.episode.max_env_steps,
        cfg.episode.max_assistant_turns,
        cfg.episode.max_tokens_per_turn,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    delta = TemplateDelta(tokenizer)
    # fncall:researcher 侧 server-parsed(35B 端点必须带 tool-call-parser),
    # executor 侧方言随 7B 模板(与训练同构)
    protocol.configure_fc_dialect(delta.fc_dialect)
    factory = make_env_factory(cfg)

    items = load_tasks([args.tasks_parquet])
    if cfg.environment.name == "webshop":
        if not args.webshop_goals:
            raise SystemExit("FATAL: webshop 域必须传 --webshop-goals")
        gold_by_idx = {}
        with open(args.webshop_goals, encoding="utf-8") as f:
            for line in f:
                g = json.loads(line)
                gold_by_idx[int(g["goal_idx"])] = g
        for ei in items:
            ei["_gold"] = gold_by_idx[int(ei["goal_idx"])]
    if args.n_tasks and args.n_tasks < len(items):
        # 与 bare 成对:保序取前 N 的分层版(探针 parquet 已按源/族分层构造,
        # 按 family round-robin 取满保证族覆盖)
        buckets: dict[str, list[dict]] = {}
        for it in items:
            fam = factory.adapter_from_item(it).task_type
            buckets.setdefault(fam, []).append(it)
        order = sorted(buckets)
        picked, idx = [], {f: 0 for f in order}
        while len(picked) < args.n_tasks:
            moved = False
            for f in order:
                if len(picked) >= args.n_tasks:
                    break
                if idx[f] < len(buckets[f]):
                    picked.append(buckets[f][idx[f]]); idx[f] += 1; moved = True
            if not moved:
                break
        items = picked
    fams: dict[str, int] = {}
    for it in items:
        fam = factory.adapter_from_item(it).task_type
        fams[fam] = fams.get(fam, 0) + 1
    print(f"[research] {len(items)} tasks: {fams}", flush=True)

    researcher_role = replace(
        cfg.role("researcher"),
        endpoint=args.researcher_api_base,
        model=args.researcher_model,
        max_concurrency=args.researcher_concurrency,
        temperature=cfg.research_rl.researcher_temperature,
        max_tokens=cfg.research.researcher_max_tokens,
        timeout_s=600.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    if not researcher_role.api_key:
        raise SystemExit(
            f"FATAL: researcher API key env {researcher_role.api_key_env!r} "
            "unset — 会话会全量 401")
    researcher_client = make_client("researcher-classic", researcher_role)

    bases = [b.strip() for b in args.executor_api_base.split(",") if b.strip()]
    rr = itertools.cycle(bases)
    pick_base = lambda: next(rr)  # noqa: E731

    out_dir = Path(args.out_dir)
    (out_dir / "sessions").mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient() as http:
        async def guarded(item):
            async with sem:
                try:
                    return await _one_session(
                        cfg=cfg, item=item, researcher_client=researcher_client,
                        tokenizer=tokenizer, delta=delta, http=http, args=args,
                        factory=factory, out_dir=out_dir, pick_base=pick_base)
                except Exception as exc:  # noqa: BLE001 — 单会话失败不炸批
                    return {"task_id": item.get("task_id"), "error": repr(exc)}

        results = await asyncio.gather(*[guarded(it) for it in items])
    new = [r for r in results if not r.get("skipped")]
    fin = sum(1 for r in new if r.get("finalized"))
    missing = sum(1 for r in new if r.get("missing_gt"))
    errs = [r for r in new if r.get("error")]
    print(f"[research] total={len(results)} skipped={len(results)-len(new)} "
          f"finalized={fin}/{len(new)} missing_gt={missing} errors={len(errs)}",
          flush=True)
    for r in errs[:5]:
        print(f"  ERR {r['task_id']}: {str(r['error'])[:160]}", flush=True)
    docs = len(list((out_dir / "docs").glob("*.md"))) if (out_dir / "docs").exists() else 0
    print(f"[research] docs written: {docs} -> {out_dir}/docs", flush=True)
    return 0 if not errs else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tasks-parquet", required=True)
    ap.add_argument("--n-tasks", type=int, default=0, help="0=全量;否则按族轮转取前N")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--executor-api-base", required=True, help="逗号分隔多端点轮询")
    ap.add_argument("--executor-model", required=True)
    ap.add_argument("--researcher-api-base", required=True, help="逗号分隔(PooledClient)")
    ap.add_argument("--researcher-model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--researcher-concurrency", type=int, default=64)
    ap.add_argument("--webshop-goals", default="")
    ap.add_argument("--alf-gtwalk-dir", default="")
    ap.add_argument("--no-privileged", action="store_true",
                    help="消融:researcher 无特权素材(纯实验探索)")
    return asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
