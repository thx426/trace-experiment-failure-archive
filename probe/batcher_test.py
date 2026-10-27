#!/usr/bin/env python3
"""_SearchBatcher 正确性 + 吞吐验证(真实打 162 retriever)。
必须先证明:①结果与逐条串行一致(不串号);②失败仍 fail loud;③吞吐提升。"""
import sys, time, threading
sys.path.insert(0, "/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM/src")
from concurrent.futures import ThreadPoolExecutor

from trace_opd.common.config import SearchQACfg
from trace_opd.envs.searchqa.adapter import SearchQAEnvFactory, _SearchBatcher

cfg = SearchQACfg(retriever_url="http://HOST_IP:8000", topk=3, preflight=True)
print(f"cfg: max_batch={cfg.search_max_batch} wait_ms={cfg.search_batch_wait_ms} "
      f"concurrency={cfg.search_concurrency}")
fac = SearchQAEnvFactory(cfg)
print("preflight ✓")

QUERIES = [
    "who wrote the novel Dune",
    "atomic number of hydrogen",
    "capital city of Australia",
    "director of the film Inception",
    "when was the Eiffel Tower completed",
]

# ---- ① 正确性:并发走 batcher 的结果,必须与单条直连结果逐字一致 ----
import httpx, json
direct = {}
with httpx.Client() as c:
    for q in QUERIES:
        r = c.post("http://HOST_IP:8000/retrieve",
                   json={"queries": [q], "topk": 3}, timeout=30)
        direct[q] = r.json()["result"][0]

b = fac._batcher
got = {}
with ThreadPoolExecutor(len(QUERIES)) as ex:
    futs = {ex.submit(b.search, q): q for q in QUERIES}
    for f, q in futs.items():
        got[q] = f.result()

ok = True
for q in QUERIES:
    a = json.dumps(direct[q], sort_keys=True)
    c_ = json.dumps(got[q], sort_keys=True)
    same = a == c_
    ok &= same
    print(f"  {'✓' if same else '✗ 串号!'} {q[:40]:<42} 直连=={('batcher' if same else '不一致')}")
print(f"① 正确性(无串号): {'PASS' if ok else 'FAIL'}")
print()

# ---- ② 吞吐:模拟 RL rollout 的高并发检索 ----
N, PAR = 512, 128
qs = [f"{QUERIES[i % len(QUERIES)]} variant {i}" for i in range(N)]
t = time.time()
with ThreadPoolExecutor(PAR) as ex:
    list(ex.map(b.search, qs))
dt = time.time() - t
print(f"② 吞吐: {N} queries / {PAR} 并发调用者 = {dt:.2f}s → {N/dt:.1f} q/s")
print(f"   (对照:单 query 单请求实测天花板 ~15 q/s ⇒ 提升 {N/dt/15:.1f}×)")
print()

# ---- ③ fail loud:指向坏地址必须 raise,不能静默返回空 ----
bad_cfg = SearchQACfg(retriever_url="http://HOST_IP:9", topk=3,
                      preflight=False, search_timeout_s=3.0)
bad = _SearchBatcher(httpx.Client(), bad_cfg)
try:
    bad.search("this must raise")
    print("③ fail loud: FAIL(坏地址竟然没 raise)")
except Exception as e:
    print(f"③ fail loud: PASS — {type(e).__name__}: {str(e)[:80]}")
