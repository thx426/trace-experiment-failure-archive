#!/usr/bin/env python3
"""批量 vs 单发的差异定位:是文档内容不同,还是仅 score 浮点抖动?
(我们的 _passages_to_string 只消费 document.contents,不看 score。)"""
import json, urllib.request
URL = "http://HOST_IP:8000/retrieve"


def post(qs, topk=3):
    body = {"queries": qs, "topk": topk, "return_scores": True}
    r = urllib.request.Request(URL, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=60) as f:
        return json.load(f)["result"]


def docs_only(per_q):
    """剥掉 score,只留文档标识与正文(即 adapter 真正消费的部分)。"""
    out = []
    for e in per_q:
        d = e.get("document", e)
        out.append((str(d.get("id", "")), str(d.get("contents", ""))))
    return out


QS = ["who wrote the novel Dune", "capital city of France",
      "atomic number of helium", "director of the film Inception"]

batch = post(QS)
print("=== 逐 query:批量 vs 单发 ===")
all_doc_same = True
for q, per_q in zip(QS, batch):
    solo = post([q])[0]
    d_same = docs_only(solo) == docs_only(per_q)
    s_solo = [round(e.get("score", 0), 6) for e in solo]
    s_batch = [round(e.get("score", 0), 6) for e in per_q]
    s_same = s_solo == s_batch
    all_doc_same &= d_same
    print(f"  {q[:34]:<36} 文档{'一致 ✓' if d_same else '不一致 ✗'}  "
          f"score{'一致' if s_same else '有差异'}")
    if not s_same:
        print(f"      solo ={s_solo}")
        print(f"      batch={s_batch}")
    if not d_same:
        print(f"      solo  docs={[x[0] for x in docs_only(solo)]}")
        print(f"      batch docs={[x[0] for x in docs_only(per_q)]}")

print()
print(f"结论:文档内容{'完全一致 ⇒ 合批安全(score 差异不影响 adapter)' if all_doc_same else '不一致 ⇒ 合批不安全'}")
