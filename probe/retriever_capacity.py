#!/usr/bin/env python3
"""retriever 真实容量曲线:找"在途批数"的最佳工作点。

背景:冒烟 v3 实测双臂 val(各 2048 episodes)同时打,服务端单发往返从
0.10s 退化到 13.7s(136×),客户端 batcher 大面积超时。原因是我把
search_concurrency 配成 64(=64 批 × 64 query = 4096 query 同时在途),
远超服务端容量。本脚本扫在途批数,找吞吐峰值与延迟拐点。

注意:双臂 = 两个独立进程各持一个 batcher ⇒ 服务端实际承受 2× 本地设置。
"""
import json, time, urllib.request, statistics
from concurrent.futures import ThreadPoolExecutor

URL = "http://HOST_IP:8000/retrieve"
BATCH = 64


def post(qs):
    body = {"queries": qs, "topk": 3, "return_scores": True}
    r = urllib.request.Request(URL, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(r, timeout=300) as f:
        res = json.load(f)["result"]
    assert len(res) == len(qs), f"shape {len(res)} != {len(qs)}"
    return time.time() - t


def run(inflight, total_batches=32):
    """inflight 个批并发,共发 total_batches 批。"""
    qs = [f"who discovered element {i} and when was it named" for i in range(BATCH)]
    lat = []
    t0 = time.time()
    with ThreadPoolExecutor(inflight) as ex:
        for d in ex.map(lambda _: post(qs), range(total_batches)):
            lat.append(d)
    dt = time.time() - t0
    n = total_batches * BATCH
    return n / dt, statistics.median(lat), max(lat)


print(f"批大小固定 {BATCH};扫在途批数(单进程视角;双臂时服务端承受 2×)")
print(f"{'在途批':>6} {'吞吐 q/s':>10} {'中位延迟':>9} {'最大延迟':>9}")
for inflight in (1, 2, 4, 8, 16, 32):
    try:
        thr, med, mx = run(inflight)
        flag = ""
        if mx > 30: flag = "  ← 延迟已失控"
        elif mx > 10: flag = "  ← 延迟偏高"
        print(f"{inflight:>6} {thr:>10.1f} {med:>8.2f}s {mx:>8.2f}s{flag}")
    except Exception as e:
        print(f"{inflight:>6}  失败: {type(e).__name__}: {str(e)[:60]}")
    time.sleep(3)   # 让服务端喘口气,避免上一档的尾巴污染下一档
