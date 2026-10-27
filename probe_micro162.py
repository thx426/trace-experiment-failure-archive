"""① rkl step17 标本 dump 有无逐词数据;② 旧工作区的显微镜工具盘点。"""
import glob, json, os

W = "/data3/wushang/workspace/TRACE-OPD/trace_opd"
f = (W + "/runs/trace_opd_4b_rkl/rollouts/step_00017/"
     "train__pick_clean_then_place_in_recep-Fork-None-DiningTable-15__"
     "trial_T20190907_203144_140359_b7da3415.json")
d = json.load(open(f))
print("=== 标本 record 顶层键 ===")
print(sorted(d.keys()))
tr = d.get("trajectory") or {}
print("trajectory 键:", sorted(tr.keys()))
st = (tr.get("steps") or [])[11]
print("第 12 步的键:", sorted(st.keys()))
for k in ("token_kl", "kl", "per_token", "response_token_ids", "token_ids",
          "logp", "micro"):
    hits = [x for x in list(d.keys()) + list(tr.keys()) + list(st.keys())
            if k in x.lower()]
    if hits:
        print(f"  含 {k!r} 的键:", hits)

print("\n=== 同 run 是否有 microscope/逐词落盘目录 ===")
for sub in sorted(os.listdir(W + "/runs/trace_opd_4b_rkl")):
    print("  ", sub)

print("\n=== 旧工作区 scripts 里的显微镜/可视化工具 ===")
for pat in ("scripts/*micro*", "scripts/*viz*", "scripts/*serve*", "*micro*", "*viz*"):
    for p in glob.glob(os.path.join(W, pat)):
        print("  ", os.path.relpath(p, W))
