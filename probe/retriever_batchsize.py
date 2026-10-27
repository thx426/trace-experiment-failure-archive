#!/usr/bin/env python3
"""扫单请求批大小:服务端内部 retrieval_batch_size=512,我们只发 64,
可能没吃满。固定总 query 数,比较不同 batch 的吞吐与延迟。"""
import json, time, urllib.request, statistics
from concurrent.futures import ThreadPoolExecutor

URL = "http://HOST_IP:8000/retrieve"
TOTAL = 2048          # 约等于一次 val 的检索量级


def post(qs):
    body = {"queries": qs, "topk": 3, "return_scores": True}
    r = urllib.request.Request(URL, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(r, timeout=600) as f:
        res = json.load(f)["result"]
    assert len(res) == len(qs), f"shape {len(res)} != {len(qs)}"
    return time.time() - t


print(f"总 query={TOTAL};扫 (单请求批大小 × 在途批数)")
print(f"{'批大小':>6} {'在途':>4} {'吞吐 q/s':>10} {'中位批延迟':>11} {'最大批延迟':>11}")
for bs in (64, 128, 256, 512):
    for inflight in (2, 4, 8):
        nbatch = max(1, TOTAL // bs)
        qs = [f"who first described phenomenon {i} in physics" for i in range(bs)]
        lat = []
        t0 = time.time()
        try:
            with ThreadPoolExecutor(inflight) as ex:
                for d in ex.map(lambda _: post(qs), range(nbatch)):
                    lat.append(d)
            dt = time.time() - t0
            print(f"{bs:>6} {inflight:>4} {nbatch*bs/dt:>10.1f} "
                  f"{statistics.median(lat):>10.2f}s {max(lat):>10.2f}s")
        except Exception as e:
            print(f"{bs:>6} {inflight:>4}  失败: {type(e).__name__}: {str(e)[:50]}")
        time.sleep(2)
