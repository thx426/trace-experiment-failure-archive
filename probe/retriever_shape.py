#!/usr/bin/env python3
"""确认 return_scores=True 路径的结构与 query 对应关系(修复依据)。"""
import json, urllib.request
URL = "http://HOST_IP:8000/retrieve"


def post(qs, scores, topk=3):
    body = {"queries": qs, "topk": topk}
    if scores:
        body["return_scores"] = True
    r = urllib.request.Request(URL, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=60) as f:
        return json.load(f)["result"]


def contents_of(elem):
    d = elem.get("document", elem) if isinstance(elem, dict) else {}
    return str(d.get("contents", ""))[:60].replace("\n", " ")


QS = ["who wrote the novel Dune", "capital city of France",
      "atomic number of helium", "director of the film Inception"]

print("=== return_scores=True:形状 ===")
res = post(QS, True)
print(f"  len(result)={len(res)} (queries={len(QS)})  各元素长度={[len(x) for x in res]}")
print(f"  首元素 keys={sorted(res[0][0].keys())}")
print()
print("=== 对应关系(每个 query 的 top1 是否语义匹配)===")
for q, per_q in zip(QS, res):
    print(f"  {q!r:34} → {contents_of(per_q[0])!r}")
print()
print("=== 与逐条单发对拍(逐字一致才算无串号)===")
ok = True
for q, per_q in zip(QS, res):
    solo = post([q], True)[0]
    same = json.dumps(solo, sort_keys=True) == json.dumps(per_q, sort_keys=True)
    ok &= same
    print(f"  {'✓' if same else '✗'} {q[:38]}")
print(f"结论: {'批量与单发完全一致 —— 可安全合批' if ok else '不一致,不可合批'}")
print()
print("=== 大批量(64)也验一次 ===")
big = [f"{QS[i % 4]} fact {i}" for i in range(64)]
rb = post(big, True)
print(f"  len(result)={len(rb)} (queries=64)  各元素长度前5={[len(x) for x in rb[:5]]}")
print(f"  形状正确: {len(rb) == 64 and all(len(x) == 3 for x in rb)}")
