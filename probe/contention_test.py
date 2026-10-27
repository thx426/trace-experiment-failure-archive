#!/usr/bin/env python3
"""量化 162 上 35B researcher 与 faiss retriever 的 GPU 争抢代价。

背景:两者共享同样 4 张卡(每卡 35B 58.8G + 索引 10G)。冒烟中机制臂
(有 researcher)大面积检索超时,基线臂(无 researcher)零错误。
本测试在「35B 空闲」与「35B 满负荷」两种条件下测 retriever 吞吐,
给出争抢的定量代价 —— 决定是放宽超时够用,还是必须重新分配资源。
"""
import json, time, threading, statistics, urllib.request
from concurrent.futures import ThreadPoolExecutor

RET = "http://HOST_IP:8000/retrieve"
LLM = ["http://HOST_IP:8880/v1/chat/completions",
       "http://HOST_IP:8881/v1/chat/completions"]
KEY = "token-abc123"
BATCH, INFLIGHT, NBATCH = 512, 2, 8


def retrieve(qs):
    r = urllib.request.Request(RET, data=json.dumps(
        {"queries": qs, "topk": 3, "return_scores": True}).encode(),
        headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(r, timeout=300) as f:
        res = json.load(f)["result"]
    assert len(res) == len(qs)
    return time.time() - t


def measure():
    qs = [f"who first proposed theory {i} in modern science" for i in range(BATCH)]
    lat, t0 = [], time.time()
    with ThreadPoolExecutor(INFLIGHT) as ex:
        for d in ex.map(lambda _: retrieve(qs), range(NBATCH)):
            lat.append(d)
    dt = time.time() - t0
    return NBATCH * BATCH / dt, statistics.median(lat), max(lat)


stop = threading.Event()
llm_done = [0]


def llm_load(idx):
    """模拟 researcher 会话:长 prompt + 中等生成。"""
    body = json.dumps({
        "model": "qwen3.6-35b-a3b",
        "messages": [{"role": "user", "content":
                      "Summarize the key failure modes an agent might hit when "
                      "searching Wikipedia for multi-hop questions. " * 20}],
        "max_tokens": 256, "temperature": 1.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    while not stop.is_set():
        try:
            r = urllib.request.Request(LLM[idx % 2], data=body, headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {KEY}"})
            with urllib.request.urlopen(r, timeout=180) as f:
                f.read()
            llm_done[0] += 1
        except Exception:
            pass


print("=== A. 35B 空闲时的 retriever 吞吐 ===")
thr, med, mx = measure()
print(f"   {thr:.1f} q/s   中位批延迟 {med:.2f}s   最大 {mx:.2f}s")

print("\n=== B. 35B 满负荷(64 并发会话)时的 retriever 吞吐 ===")
threads = [threading.Thread(target=llm_load, args=(i,), daemon=True) for i in range(64)]
for t in threads:
    t.start()
time.sleep(20)          # 让 35B 先压满
print(f"   (35B 预热完成,已完成 {llm_done[0]} 次会话)")
thr2, med2, mx2 = measure()
print(f"   {thr2:.1f} q/s   中位批延迟 {med2:.2f}s   最大 {mx2:.2f}s")
stop.set()
time.sleep(2)

print(f"\n=== 争抢代价 ===")
print(f"   吞吐  {thr:.0f} → {thr2:.0f} q/s   ({thr2/thr*100:.0f}% 保留)")
print(f"   延迟  {med:.2f}s → {med2:.2f}s   ({med2/max(med,1e-9):.1f}×)")
need = 2765
print(f"\n   每步需求 {need} query:")
print(f"     35B 空闲   {need/thr:.1f}s")
print(f"     35B 满负荷 {need/max(thr2,1e-9):.1f}s")
print(f"   每步总时长按 3-5 分钟计 ⇒ 争抢下检索占比 "
      f"{need/max(thr2,1e-9)/240*100:.1f}%")
