#!/usr/bin/env python3
"""_SearchBatcher 修复后验证 —— 重点:必须真正走到"多 query 合批"路径。

上一版测试的教训:只发 5 个 query,batcher 把它们分散到不同批(每批 1 个),
恰好走单 query 路径,躲过了服务端多 query 转置 bug ⇒ 测试 PASS 但生产全灭。
本版通过 (a) 高并发同时提交 (b) 直接埋点统计实际批大小 来确保覆盖。
"""
import sys, time, json, threading, collections
sys.path.insert(0, "/fs/fast/u2021000902/wushang/workspace/TRACE-OPD-TERM/src")
from concurrent.futures import ThreadPoolExecutor
import httpx

from trace_opd.common.config import SearchQACfg
from trace_opd.envs.searchqa import adapter as A

cfg = SearchQACfg(retriever_url="http://HOST_IP:8000", topk=3, preflight=True)
fac = A.SearchQAEnvFactory(cfg)
b = fac._batcher
print(f"batcher: max_batch={cfg.search_max_batch} wait_ms={cfg.search_batch_wait_ms}")

# --- 埋点:记录每次实际发出的批大小 ---
sizes = []
orig_flush = b._flush
def spy(batch):
    sizes.append(len(batch))
    return orig_flush(batch)
b._flush = spy

QS = ["who wrote the novel Dune", "capital city of France", "atomic number of helium",
      "director of the film Inception", "when was the Eiffel Tower completed",
      "who painted the Mona Lisa", "largest planet in the solar system",
      "who discovered penicillin"]

# --- 单发基准(document 部分)---
def docs_of(passages):
    out = []
    for e in passages:
        d = e.get("document", e) if isinstance(e, dict) else {}
        out.append((str(d.get("id", "")), str(d.get("contents", ""))))
    return out

base = {}
with httpx.Client() as c:
    for q in QS:
        r = c.post("http://HOST_IP:8000/retrieve",
                   json={"queries": [q], "topk": 3, "return_scores": True}, timeout=30)
        base[q] = docs_of(r.json()["result"][0])

# --- ① 高并发合批:每个 query 重复 40 次,共 320 个并发请求 ---
tasks = [q for q in QS for _ in range(40)]
t = time.time()
with ThreadPoolExecutor(160) as ex:
    got = list(ex.map(b.search, tasks))
dt = time.time() - t

hist = collections.Counter(sizes)
multi = sum(n for sz, n in hist.items() if sz > 1)
print(f"\n① 实际批大小分布: {dict(sorted(hist.items()))}")
print(f"   合批(size>1)的批次数={multi} / 总批次={len(sizes)}  "
      f"{'✓ 已覆盖多 query 路径' if multi > 0 else '✗ 未覆盖!测试无效'}")
print(f"   最大批={max(sizes)}  平均={sum(sizes)/len(sizes):.1f}")

# --- ② 正确性:每个并发结果都必须与该 query 的单发文档一致(无串号)---
bad = 0
for q, res in zip(tasks, got):
    if docs_of(res) != base[q]:
        bad += 1
print(f"\n② 无串号: {len(tasks)-bad}/{len(tasks)} 一致  "
      f"{'PASS' if bad == 0 else f'FAIL(错配 {bad} 条)'}")

# --- ③ 吞吐 ---
print(f"\n③ 吞吐: {len(tasks)} queries / {dt:.2f}s = {len(tasks)/dt:.1f} q/s "
      f"(单 query 单请求天花板 ~15 q/s ⇒ {len(tasks)/dt/15:.1f}×)")

# --- ④ 形状校验能否抓住转置(把 return_scores 关掉模拟旧行为)---
print("\n④ 形状硬校验(模拟服务端转置):", end=" ")
import types
def bad_flush(batch):
    qs = [c["q"] for c in batch]
    try:
        resp = b._http.post(b._url, json={"queries": qs, "topk": b._topk}, timeout=30)
        results = resp.json()["result"]
        if len(results) != len(batch) or not all(isinstance(r, list) for r in results):
            raise RuntimeError(f"malformed: len={len(results)} for {len(batch)}")
        print("未抓住 ✗")
    except RuntimeError as e:
        print(f"抓住了 ✓ — {e}")
bad_flush([{"q": "a"}, {"q": "b"}])
