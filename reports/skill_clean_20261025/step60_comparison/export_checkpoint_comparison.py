"""Read existing checkpoints' bare eval records; no training or inference changes."""
from pathlib import Path
import os, sys, json, re, datetime, hashlib, gzip, base64
R=Path('/root/trace-webshop-arm-ratio/experiments/pass5_clean_skill_20261025_v2')
os.environ['TRACE_LOCAL_RESEARCH_KEY']='loopback-only'
sys.path.insert(0,str(R))
from audit_run import read_evaluation, load_config, require
import pyarrow.parquet as pq
cfg=load_config(R/'configs/clean_skill.yaml')
dev=[d for p in cfg.train.val_files for d in pq.read_table(p).to_pylist()]
ids={d['extra_info']['task_id'] for d in dev}
require(len(ids)==256 and cfg.eval.n==5 and cfg.eval.temperature==0.6,'unexpected evaluation config')
runs={'clean':R/'code/runs/pass5_clean_skill_v2_s42','grpo':R.parent/'pass5_20261023/code/runs/pass5_grpo_s42'}
out={'checked_at':datetime.datetime.now().astimezone().isoformat(),'audits':{},'rows':[], 'runtime':{}}
seq=json.loads((R/'sequence_status.json').read_text());out['runtime']['sequence']=seq
pid=int((R/'sequence.pid').read_text());out['runtime']['sequence_pid_file']=pid
cmd=Path(f'/proc/{pid}/cmdline');out['runtime']['sequence_process']=cmd.read_bytes().replace(b'\0',b' ').decode(errors='replace') if cmd.exists() else None
log=Path(seq['owner_state'],'command.log').read_text(errors='replace')
completed=[int(m.group(1)) for line in log.splitlines() if 'timing_s/step' in line and (m:=re.search(r'step:\s*(\d+)',line))]
out['runtime']['completed_updates']=completed
out['runtime']['student_log_tail']=log[-7000:]
out['runtime']['rollout_counts']={p.name:len(list(p.glob('*.json'))) for p in sorted((runs['clean']/'rollouts').glob('step_*'))}
for name in ('baseline_reviews','baseline_stop'):
 p=R/(name+'.json');out['runtime'][name]=json.loads(p.read_text()) if p.exists() else None
out['runtime']['abort']=(R/'ABORT').read_text() if (R/'ABORT').exists() else None
out['runtime']['frozen_changed']=[p for p,h in json.loads((R/'frozen_manifest.json').read_text()).items() if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
results={}
for arm,run in runs.items():
 manifests=list(run.glob('launch_manifest_*.json'));require(len(manifests)==1,'ambiguous launch')
 launch=json.loads(manifests[0].read_text())['launch_id']
 for step in (20,40,60):
  key=f'{arm}_{step}';directory=run/'rollouts'/f'step_{step:05d}_val';files=sorted(directory.glob('*.json'))
  entry={'run':str(run),'launch_id':launch,'step':step,'rows':len(files),'expected_rows':1280,'complete':len(files)==1280}
  out['audits'][key]=entry
  if not entry['complete']:continue
  ev=read_evaluation(run,launch,step,ids,5,0.6);digest=hashlib.sha256();byid={t:[] for t in ids}
  for p in files:
   data=p.read_bytes();d=json.loads(data);digest.update(p.name.encode());digest.update(data)
   byid[d['task_id']].append({'sid':d['session_id'],'won':d['won'],'score':d['score'],'sample_id':d['sample_id'],'file':p.name})
  results[key]={t:sorted(v,key=lambda d:d['sid']) for t,v in byid.items()}
  successes=sum(map(sum,ev.values()));solved=sum(any(v) for v in ev.values())
  entry.update(passed=True,tasks=len(ev),successes=successes,solved=solved,pass1_estimate=successes/1280,pass5=solved/256,
               bare=True,n=5,temperature=0.6,raw_sha256=digest.hexdigest())
for d in sorted(dev,key=lambda d:d['extra_info']['task_id']):
 t=d['extra_info']['task_id'];r={'task_id':t,'instruction':d['extra_info'].get('instruction','')}
 for key,vals in results.items():
  r[key]=vals[t];r[key+'_k']=sum(x['won'] for x in vals[t])
 r['lost_at20']=r.get('grpo_20_k',0)>0 and r.get('clean_20_k')==0
 out['rows'].append(r)
out['exported_at']=datetime.datetime.now().astimezone().isoformat()
name='step60_bare_comparison_20261025.txt'
(R/name).write_text(base64.b64encode(gzip.compress(json.dumps(out,ensure_ascii=False).encode())).decode())
print('COMPARISON_READY',name,json.dumps(out['audits'],ensure_ascii=False),'completed_updates',completed[-1:])
