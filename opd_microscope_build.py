#!/usr/bin/env python3
"""OPD 显微镜(静态 SPA)—— 旧 4B 纯蒸馏 run 的全程逐词观察面。

数据源:runs/<run>/microscope/rank*.jsonl(ids/kl/mask,按 resp_hash)
     × runs/<run>/rollouts/step_*(元数据,全量长度曲线用全部 dump)。
通道语义:红底深浅 = 老师(带文档的自己)与学生在该词上的分歧(蒸馏损失强度);
        灰字 = 环境反馈(mask=0,不计损失);无底色 = 分歧≈0(老师同意)。
用法:venv_train/bin/python opd_microscope_build.py <run_dir> <out_dir>
"""
import glob, html, json, os, re, sys

RUN = sys.argv[1] if len(sys.argv) > 1 else \
    "/data3/wushang/workspace/TRACE-OPD/trace_opd/runs/trace_opd_4b_rkl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/data3/wushang/opd_microscope_rkl"
TOK = "/data3/wushang/model/Qwen/Qwen3-4B-Instruct-2507"
KL_LO = 0.1          # 低于此的连续段合并为"安静段"
os.makedirs(OUT + "/data", exist_ok=True)

from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained(TOK, trust_remote_code=True)
_cache = {}
def tok1(i):
    if i not in _cache:
        _cache[i] = tk.decode([i])
    return _cache[i]

print("载入 tape …")
tape = {}
for f in sorted(glob.glob(RUN + "/microscope/rank*.jsonl")):
    for line in open(f):
        try:
            r = json.loads(line)
            tape[int(r["resp_hash"])] = (r["ids"], r["kl"], r["mask"])
        except Exception:
            pass
print(f"tape 轨迹 {len(tape)}")

def segs_of(ids, kl, mask):
    """[(text, code)];code: -2 环境 / -1 安静 / 其余=KL 值"""
    out, buf, mode = [], [], None   # mode: 'obs' | 'quiet'
    def flush():
        nonlocal buf, mode
        if buf:
            out.append([tk.decode(buf), -2 if mode == "obs" else -1])
        buf, mode = [], None
    for i, k, m in zip(ids, kl, mask):
        if not m:
            if mode != "obs":
                flush(); mode = "obs"
            buf.append(i)
        elif k < KL_LO:
            if mode != "quiet":
                flush(); mode = "quiet"
            buf.append(i)
        else:
            flush()
            out.append([tok1(i), round(float(k), 2)])
    flush()
    return out

steps_meta = []
for d in sorted(p for p in glob.glob(RUN + "/rollouts/step_*")
                if not p.endswith("_val")):
    name = os.path.basename(d)
    sid = int(re.search(r"step_0*(\d+)", name).group(1))
    lens, wons, trajs, seen = [], [], [], set()
    for f in sorted(glob.glob(d + "/*.json")):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        h = int(r.get("response_token_hash") or 0)
        if h in seen:
            continue
        seen.add(h)
        n = int(r.get("response_token_len") or 0)
        lens.append(n); wons.append(bool(r.get("won")))
        if h in tape:
            ids, kl, mask = tape[h]
            mk = [k for k, m in zip(kl, mask) if m]
            trajs.append({
                "short": str(r.get("task_id", "")).split("/")[-2][:70]
                         if "/" in str(r.get("task_id", "")) else str(r.get("task_id", ""))[:70],
                "won": bool(r.get("won")),
                "outcome": str(r.get("outcome", "")),
                "ntok": len(ids),
                "mkl": round(sum(mk) / max(1, len(mk)), 3),
                "env": int(r.get("env_steps") or 0),
                "trunc": bool(r.get("truncated")),
                "segs": segs_of(ids, kl, mask),
            })
    if not lens:
        continue
    trajs.sort(key=lambda t: -t["ntok"])
    ls = sorted(lens)
    meta = {"id": sid, "dir": name, "n": len(lens), "ntape": len(trajs),
            "won": round(sum(wons) / len(wons), 3),
            "p50": ls[len(ls)//2], "p90": ls[int(len(ls)*0.9)], "max": ls[-1]}
    steps_meta.append(meta)
    with open(f"{OUT}/data/{name}.js", "w") as fh:
        fh.write('window.OPD_STEP_DATA=window.OPD_STEP_DATA||{};'
                 f'window.OPD_STEP_DATA[{json.dumps(name)}]='
                 + json.dumps({"trajs": trajs}, ensure_ascii=False) + ";")
    print(f"{name}: dump={len(lens)} tape={len(trajs)} p90={meta['p90']} max={meta['max']}")

run = {"name": os.path.basename(RUN), "crash": 17, "steps": steps_meta,
       "note": "红=老师·学生逐词分歧(蒸馏损失强度);灰=环境反馈(不计损失);无色=分歧≈0(老师同意)"}
with open(OUT + "/data/run.js", "w") as fh:
    fh.write("window.OPD_RUN=" + json.dumps(run, ensure_ascii=False) + ";")

INDEX = """<!doctype html><meta charset="utf-8"><title>OPD 显微镜 — 思考膨胀全程</title>
<style>
body{margin:0;font:13px/1.5 -apple-system,"PingFang SC",sans-serif;display:flex;height:100vh;color:#222}
#side{width:250px;min-width:250px;border-right:1px solid #ddd;overflow-y:auto;background:#fafafa;padding:10px}
#list{width:330px;min-width:330px;border-right:1px solid #ddd;overflow-y:auto;padding:8px}
#main{flex:1;overflow-y:auto;padding:14px 20px}
h3{margin:6px 0}.muted{color:#888;font-size:11px}
.step{padding:5px 8px;border-radius:6px;cursor:pointer;margin:2px 0;display:flex;justify-content:space-between}
.step:hover{background:#eee}.step.sel{background:#1e63d0;color:#fff}
.traj{padding:6px 8px;border-radius:6px;cursor:pointer;margin:3px 0;border:1px solid #eee}
.traj:hover{background:#f2f6ff}.traj.sel{border-color:#1e63d0;background:#eef4fd}
.tok{white-space:pre-wrap;word-break:break-word;font-size:12.5px;line-height:1.7}
.obs{color:#999}.kl{border-radius:2px}
#legend{position:sticky;top:0;background:#fff;padding:6px 0;border-bottom:1px solid #eee;margin-bottom:8px}
svg{width:100%;height:100px}
.badge{font-size:10px;border-radius:8px;padding:0 6px;color:#fff}
</style>
<div id=side>
 <h3 id=rname></h3><div class=muted id=rnote></div>
 <div class=muted style="margin-top:8px">每条轨迹词数(全量 dump):<span style="color:#c0392b">— max</span> <span style="color:#1e63d0">— p90</span> <span style="color:#999">— p50</span></div>
 <svg id=chart viewBox="0 0 230 100" preserveAspectRatio="none"></svg>
 <div id=steps></div>
</div>
<div id=list><div class=muted id=lhead>← 选一步</div><div id=trajs></div></div>
<div id=main><div id=legend></div><div id=view class=tok>点左侧步 → 中列轨迹,即可逐词查看。<br><br>建议路线:step_00001(正常)→ step_00009(相变,长度起飞)→ step_00012+(每步数千字思考)→ step_00017(撞崩)。</div></div>
<script src="data/run.js"></script>
<script>
const R=window.OPD_RUN;let cur=null,curT=null;
document.getElementById('rname').textContent=R.name;
document.getElementById('rnote').textContent=R.note;
document.getElementById('legend').innerHTML='通道:<span class=kl style="background:rgba(192,57,43,.15);padding:0 4px">分歧小</span> <span class=kl style="background:rgba(192,57,43,.45);padding:0 4px">中</span> <span class=kl style="background:rgba(192,57,43,.8);padding:0 4px;color:#fff">大</span> <span class=obs>灰=环境反馈</span> 无色=老师同意(分歧&lt;'+0.1+')';
// 侧栏演变曲线
(function(){const s=R.steps,W=230,H=100,mx=Math.max(...s.map(x=>x.max));
 const X=i=>8+i*(W-16)/(s.length-1),Y=v=>H-8-(v/mx)*(H-20);
 const line=(k,c)=>'<polyline fill="none" stroke="'+c+'" stroke-width="1.6" points="'+s.map((x,i)=>X(i)+','+Y(x[k])).join(' ')+'"/>';
 let g=line('p50','#999')+line('p90','#1e63d0')+line('max','#c0392b');
 s.forEach((x,i)=>{if(x.id===R.crash)g+='<text x="'+(X(i)-5)+'" y="12" fill="#c0392b" font-size="12">✗崩</text>';});
 document.getElementById('chart').innerHTML=g;})();
const sd=document.getElementById('steps');
R.steps.forEach(s=>{const d=document.createElement('div');d.className='step';d.id='st'+s.id;
 d.innerHTML='<span>步 '+s.id+(s.id===R.crash?' ✗':'')+'</span><span class=muted>p90 '+(s.p90/1000).toFixed(1)+'k · 赢'+Math.round(s.won*100)+'%</span>';
 d.onclick=()=>pick(s);sd.appendChild(d);});
function pick(s){cur=s;document.querySelectorAll('.step').forEach(e=>e.classList.remove('sel'));
 document.getElementById('st'+s.id).classList.add('sel');
 if(window.OPD_STEP_DATA&&window.OPD_STEP_DATA[s.dir])return show(s);
 const sc=document.createElement('script');sc.src='data/'+s.dir+'.js';sc.onload=()=>show(s);document.body.appendChild(sc);}
function show(s){const D=window.OPD_STEP_DATA[s.dir].trajs;
 document.getElementById('lhead').innerHTML='<b>'+s.dir+'</b> · dump '+s.n+' 条 / 逐词落盘 '+s.ntape+' 条(按词数降序)';
 const el=document.getElementById('trajs');el.innerHTML='';
 D.forEach((t,i)=>{const d=document.createElement('div');d.className='traj';
  d.innerHTML='<span class=badge style="background:'+(t.won?'#2e7d32':'#c0392b')+'">'+(t.won?'赢':'输')+'</span> '
   +'<b>'+(t.ntok/1000).toFixed(1)+'k 词</b> · 均分歧 '+t.mkl+(t.trunc?' · 截断':'')
   +'<div class=muted>'+t.short+'</div>';
  d.onclick=()=>{document.querySelectorAll('.traj').forEach(e=>e.classList.remove('sel'));d.classList.add('sel');view(t)};
  el.appendChild(d);});}
function view(t){const v=document.getElementById('view');v.innerHTML='';
 const h=document.createElement('div');h.style.cssText='margin-bottom:10px;padding:8px;background:#f6f8fa;border-radius:8px';
 h.textContent=(t.won?'✓赢':'✗输')+' · '+t.ntok+' 词 · 环境步 '+t.env+' · 掩码内平均分歧 '+t.mkl+' · '+t.short;
 v.appendChild(h);
 t.segs.forEach(([txt,c])=>{const sp=document.createElement('span');sp.textContent=txt;
  if(c===-2)sp.className='obs';
  else if(c!==-1){sp.className='kl';const a=Math.min(.85,.1+c/5);
   sp.style.background='rgba(192,57,43,'+a.toFixed(2)+')';if(a>.5)sp.style.color='#fff';
   sp.title='分歧 KL='+c;}
  v.appendChild(sp);});}
</script>"""
open(OUT + "/index.html", "w").write(INDEX)
print("done →", OUT)
