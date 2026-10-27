from pathlib import Path
from collections import defaultdict,Counter
from statistics import mean,median
import json,re,math,gzip,base64,hashlib,datetime
from zoneinfo import ZoneInfo
R=Path('/root/trace-webshop-arm-ratio/experiments/pass5_20261023');P=R/'code/runs/pass5_all_skill_s42';D=R.parent/'grpo_final_review_20261023'
def q(v):
 v=sorted(float(x) for x in v if x is not None)
 return {} if not v else dict(n=len(v),sum=sum(v),mean=mean(v),median=median(v),p90=v[max(0,math.ceil(.9*len(v))-1)],max=max(v),min=min(v))
def metrics(path):
 out={}
 for l in Path(path).read_text(errors='replace').replace('\r','\n').splitlines():
  m=re.search(r'step:(\d+) - ',l)
  if not m:continue
  vals={k:float(v) for k,v in re.findall(r'([A-Za-z0-9_./-]+):(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)',l)}
  out[int(m[1])]={k:v for k,v in vals.items() if k.startswith(('timing','rollout_extra/research','rollout_extra/steer','rollout_extra/generated','rollout_extra/env_steps','rollout_extra/invalid_action','global_seqlen','actor/perf'))}
 return out
M=metrics('/tmp/p5a100/command.log');done=sorted(s for s in M if 1<=s<=10)
o=dict(snapshot_at=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),completed_steps_available=sorted(M),analyzed_steps=done,all_skill_metrics={s:M[s] for s in done},grpo_metrics={s:v for s,v in metrics('/tmp/p5g100/command.log').items() if s in done},steps={},research_sessions=[],tape_summary={},timing_caveat='Per-episode/session durations overlap; do not sum them as step wall time. Archive timestamps approximate save completion.')
for step in done:
 rs=[]
 for p in (P/'rollouts'/f'step_{step:05d}').glob('*.json'):
  r=json.loads(p.read_text());a=r['steering'];m=r['trajectory']['meta'];rs.append(dict(task=r['task_id'],arm=a.get('steer_arm',0),wall=r.get('wall_time_seconds'),tokens=m.get('generated_token_count'),turns=r.get('env_steps'),seq_len=len(r.get('prompt_token_ids',[]))+len(r.get('response_token_ids',[])),mtime=p.stat().st_mtime,steer_count=a.get('steer_count',0),rejects=a.get('steer_reject_count',0),discarded=a.get('steer_n_draft_discarded',0),attempts=m.get('student_attempt_count',1),archive=r.get('research',{}).get('research_archive')))
 arms={str(a):{k:q([r[k] for r in rs if r['arm']==a]) for k in ['wall','tokens','turns','seq_len','steer_count','rejects','discarded','attempts']} for a in [0,1]}
 o['steps'][step]=dict(arms=arms,oversized8192=sum(r['seq_len']>8192 for r in rs),rows=len(rs),slowest=sorted(rs,key=lambda r:r['wall'],reverse=True)[:3],last_rollout_mtime=max(r['mtime'] for r in rs))
 for archive in sorted({r['archive'] for r in rs if r['archive']}):
  p=P/archive;m=json.loads((p/'meta.json').read_text());eps=[json.loads(x) for x in (p/'episodes.jsonl').read_text().splitlines() if x.strip()]
  ew=[x['trajectory'].get('meta',{}).get('wall_s',0) for x in eps];g=m.get('guidance_log',[]);roundmax=[]
  for i,z in enumerate(g):
   a=z.get('episodes_before',0);b=g[i+1].get('episodes_before',len(eps)) if i+1<len(g) else len(eps);roundmax.append(max(ew[a:b],default=0))
  o['research_sessions'].append(dict(step=step,task=m['task_id'],end=m['ts'],wall=m.get('research_wall_s'),turns=m.get('turns_used'),experiments=m.get('experiments_used'),episodes=m.get('episodes_consumed'),prompt_tokens=m.get('researcher_prompt_tokens'),completion_tokens=m.get('researcher_completion_tokens'),error=bool(m.get('error')),episode_walls=ew,experiment_round_max_sum=sum(roundmax),narrations=sum(bool(x.get('narration')) for x in eps),narration_chars=sum(len(x.get('narration') or '') for x in eps),experiment_tokens=sum(x['trajectory'].get('meta',{}).get('generated_token_count',0) for x in eps)))
lo=min(x['end']-x['wall'] for x in o['research_sessions']);hi=max(x['end'] for x in o['research_sessions']);calls=[]
for p in (P/'tape').glob('*.jsonl'):
 for l in p.open():
  try:r=json.loads(l)
  except json.JSONDecodeError:continue
  if 'latency_s' not in r or not lo<=r.get('ts',0)<=hi:continue
  calls.append(dict(role=r.get('role'),ctx=r.get('ctx'),ts=r['ts'],latency=r['latency_s'],attempts=r.get('attempts'),error=bool(r.get('error')),prompt=(r.get('usage') or {}).get('prompt_tokens',0),completion=(r.get('usage') or {}).get('completion_tokens',0)))
for role in {r['role'] for r in calls}:
 z=[r for r in calls if r['role']==role];o['tape_summary'][role]=dict(calls=len(z),latency=q([r['latency'] for r in z]),errors=sum(r['error'] for r in z),retries=sum(max(0,(r['attempts'] or 1)-1) for r in z),prompt_tokens=sum(r['prompt'] for r in z),completion_tokens=sum(r['completion'] for r in z),ctx_examples=list({r['ctx'] for r in z})[:3])
o['tape_call_metadata']=calls
raw=json.dumps(o,separators=(',',':')).encode();(D/'all_skill_timing_snapshot.json').write_bytes(raw);enc=base64.b64encode(gzip.compress(raw)).decode();(D/'all_skill_timing_transfer.txt').write_text(enc);print('TIMING_DONE',len(raw),len(enc),hashlib.sha256(raw).hexdigest());print('STEPS',done,'LATEST',max(M));print('RESEARCH',q([x['wall'] for x in o['research_sessions']]));print('TAPE',o['tape_summary'])
