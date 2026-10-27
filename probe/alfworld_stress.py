#!/usr/bin/env python3
"""ALFWorld 环境侧高并发压测(RL rollout 支撑能力标定,2026-08-13)。

与生产完全同构:make_env_factory → adapter.open_episode()(过 spawn gate)
→ AlfworldEnvHandle.step → close;每 episode 一个 TextWorld 子进程
(引擎非线程安全,进程隔离是唯一并发模型——worker.py 注释)。

测什么:
  * spawn 风暴:C 个 episode 并发经 gate=G 分批 boot 的时延分布与失败率;
  * 稳态 step:真实引擎负载(混合 valid/invalid 命令脚本)的时延分布;
  * 资源:load1/内存 每 5s 采样;
  * 共存安全:压测期间对全部 SGLang 服务做健康探针,并可选 --coload
    对 8250 持续打真实生成请求,对比生成吞吐降幅(cpu-starvation 前科)。

用法(A800,TERM/probe 下):
  python alfworld_stress.py --config configs/classic/alfworld.yaml \
    --parquet .../train_3553.parquet --episodes 512 --concurrency 512 \
    --spawn-gate 128 --steps 10 --out /tmp/alf_stress_512.json [--coload]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

# 混合命令脚本:含移动/开容器/取放(常 invalid)/观察——负载覆盖引擎的
# 解析+状态更新+格式渲染路径;invalid 也走完整 parser(真实 rollout 常态)。
STEP_SCRIPT = [
    "look", "inventory", "go to drawer 1", "open drawer 1",
    "take book 1 from drawer 1", "close drawer 1", "go to shelf 1",
    "examine shelf 1", "go to desk 1", "look",
]


def one_episode(adapter, steps: int) -> dict:
    rec = {"spawn_s": None, "step_s": [], "ok": False, "err": ""}
    t0 = time.monotonic()
    try:
        h = adapter.open_episode()
        rec["spawn_s"] = round(time.monotonic() - t0, 3)
        try:
            for i in range(steps):
                cmd = STEP_SCRIPT[i % len(STEP_SCRIPT)]
                t1 = time.monotonic()
                h.step(cmd)
                rec["step_s"].append(round(time.monotonic() - t1, 4))
            rec["ok"] = True
        finally:
            h.close()
    except Exception as exc:  # noqa: BLE001
        rec["err"] = repr(exc)[:300]
    return rec


def sampler(stop: threading.Event, out: list, ports: list[int]) -> None:
    while not stop.is_set():
        load1 = os.getloadavg()[0]
        with open("/proc/meminfo") as f:
            mem = {l.split(":")[0]: int(l.split()[1]) for l in f if ":" in l}
        avail_gb = mem.get("MemAvailable", 0) / 1048576
        health = 0
        for p in ports:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{p}/v1/models", timeout=3)
                health += 1
            except Exception:  # noqa: BLE001
                pass
        out.append({"t": round(time.time(), 1), "load1": round(load1, 1),
                    "avail_gb": round(avail_gb, 1), "sglang_up": health})
        stop.wait(5.0)


def coload(stop: threading.Event, out: dict, api: str) -> None:
    """压测共存负载:对 SGLang 持续打小批生成,统计吞吐(tokens/s)。"""
    body = json.dumps({"model": "qwen25-7b",
                       "prompt": "Write a detailed story about a robot:",
                       "max_tokens": 256, "temperature": 1.0}).encode()

    def one():
        req = urllib.request.Request(api + "/completions", body,
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)["usage"]["completion_tokens"]

    tok, n, t0 = 0, 0, time.monotonic()
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(one) for _ in range(16)}
        while not stop.is_set():
            done = {f for f in futs if f.done()}
            for f in done:
                futs.discard(f)
                try:
                    tok += f.result(); n += 1
                except Exception:  # noqa: BLE001
                    pass
                if not stop.is_set():
                    futs.add(ex.submit(one))
            time.sleep(0.2)
        for f in futs:
            try:
                tok += f.result(timeout=120); n += 1
            except Exception:  # noqa: BLE001
                pass
    dt = time.monotonic() - t0
    out.update({"requests": n, "tokens": tok, "wall_s": round(dt, 1),
                "tok_per_s": round(tok / dt, 1)})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--spawn-gate", type=int, default=0, help="覆盖 spawn_concurrency;0=用 config")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--tasks", type=int, default=128, help="不同任务数(episodes 在其上轮转=n 副本形态)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--coload", default="", help="SGLang api base,启用共存生成负载")
    ap.add_argument("--sglang-ports", default="8250,8251,8252,8253,8254,8255,8256,8257")
    args = ap.parse_args()

    from trace_opd.common.config import load_config
    from trace_opd.core import protocol
    from trace_opd.envs.registry import make_env_factory

    cfg = load_config(args.config)
    protocol.configure_episode_caps(cfg.episode.max_env_steps,
                                    cfg.episode.max_assistant_turns,
                                    cfg.episode.max_tokens_per_turn)
    if args.spawn_gate:
        cfg.alfworld.spawn_concurrency = args.spawn_gate
    factory = make_env_factory(cfg)

    df = pd.read_parquet(args.parquet)
    items = [dict(r["extra_info"]) for _, r in df.iterrows()]
    rng = random.Random(args.seed)
    rng.shuffle(items)
    tasks = items[: args.tasks]
    episodes = [tasks[i % len(tasks)] for i in range(args.episodes)]

    ports = [int(p) for p in args.sglang_ports.split(",") if p]
    samples: list = []
    stop = threading.Event()
    smp = threading.Thread(target=sampler, args=(stop, samples, ports), daemon=True)
    smp.start()
    co_out: dict = {}
    co_thread = None
    if args.coload:
        co_stop = threading.Event()
        co_thread = threading.Thread(target=coload, args=(co_stop, co_out, args.coload), daemon=True)
        co_thread.start()

    print(f"[stress] episodes={args.episodes} concurrency={args.concurrency} "
          f"gate={cfg.alfworld.spawn_concurrency} steps={args.steps} "
          f"tasks={len(tasks)}", flush=True)
    t0 = time.monotonic()
    recs: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(one_episode, factory.adapter_from_item(it), args.steps)
                for it in episodes]
        for i, f in enumerate(as_completed(futs), 1):
            recs.append(f.result())
            if i % 64 == 0:
                print(f"[stress] {i}/{len(futs)} done, load1={os.getloadavg()[0]:.0f}",
                      flush=True)
    wall = time.monotonic() - t0
    if args.coload:
        co_stop.set(); co_thread.join(timeout=150)
    stop.set(); smp.join(timeout=10)

    ok = [r for r in recs if r["ok"]]
    fails = [r for r in recs if not r["ok"]]
    spawns = sorted(r["spawn_s"] for r in ok)
    steps_flat = sorted(s for r in ok for s in r["step_s"])

    def pct(a, q):
        return round(a[min(len(a) - 1, int(q * len(a)))], 3) if a else None

    summary = {
        "params": {"episodes": args.episodes, "concurrency": args.concurrency,
                   "gate": cfg.alfworld.spawn_concurrency, "steps": args.steps,
                   "tasks": len(tasks), "coload": bool(args.coload)},
        "wall_s": round(wall, 1),
        "eps_per_min": round(60 * len(ok) / wall, 1),
        "ok": len(ok), "fail": len(fails),
        "fail_samples": [f["err"] for f in fails[:3]],
        "spawn_s": {"p50": pct(spawns, .5), "p90": pct(spawns, .9),
                    "p99": pct(spawns, .99), "max": spawns[-1] if spawns else None},
        "step_s": {"p50": pct(steps_flat, .5), "p90": pct(steps_flat, .9),
                   "p99": pct(steps_flat, .99),
                   "max": steps_flat[-1] if steps_flat else None},
        "load1_max": max((s["load1"] for s in samples), default=None),
        "avail_gb_min": min((s["avail_gb"] for s in samples), default=None),
        "sglang_up_min": min((s["sglang_up"] for s in samples), default=None),
        "coload": co_out,
        "samples": samples,
    }
    Path(args.out).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    slim = {k: v for k, v in summary.items() if k != "samples"}
    print("[stress] " + json.dumps(slim, ensure_ascii=False), flush=True)
    return 0 if not fails else 2


if __name__ == "__main__":
    sys.exit(main())
