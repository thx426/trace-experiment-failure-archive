from pathlib import Path
import json,sys,re,collections,gzip,base64,datetime,subprocess,os
os.environ["TRACE_LOCAL_RESEARCH_KEY"]="loopback-only"
R=Path('/root/trace-webshop-arm-ratio/experiments/pass5_clean_skill_20261025_v1')
sys.path.insert(0,str(R));sys.path.insert(0,str(R/'code/src'))
from audit_run import read_evaluation,load_config
from trace_opd.research.clean_skill import exact_specs
import pyarrow.parquet as pq
cfg=load_config(R/'configs/clean_skill.yaml'); run=R/'code/runs/pass5_clean_skill_s42'
x={'checked_at':datetime.datetime.now().astimezone().isoformat()}
ids={r['extra_info']['task_id'] for p in cfg.train.val_files for r in pq.read_table(p).to_pylist()}
m=json.loads(next(run.glob('launch_manifest_*.json')).read_text())
ev=read_evaluation(run,m['launch_id'],0,ids,5,0.6)
x['initial_bare_evaluation']={'passed':True,'tasks':len(ev),'rows':sum(map(len,ev.values())),'successes':sum(map(sum,ev.values())),'solved':sum(any(v) for v in ev.values())}
rows=[json.loads(p.read_text()) for p in (run/'rollouts').glob('step_*/*.json')]
train=[r for r in rows if not r['validate']]
by_step=collections.defaultdict(list)
for d in train:by_step[d['global_step']].append(d)
x['steering_by_step']={}
for step,rs in by_step.items():
 g=[d['steering'] for d in rs if d['steering'].get('steer_arm')==1]
 x['steering_by_step'][step]={'rows':len(rs),'guided_assigned':len(g),'positive_count':sum(d.get('steer_count',0)>0 for d in g),'warmup_rows':sum(d.get('steer_entropy_theta')==-1 for d in g),'entropy_theta':dict(collections.Counter(d.get('steer_entropy_theta') for d in g)),'count_total':sum(d.get('steer_count',0) for d in g),'n_gate_bypassed':sum(d.get('steer_n_gate_bypassed',0) for d in g),'doc_overflow':sum(d.get('steer_doc_overflow',0) for d in g)}
seen=set(); issues=[];samples=[]
for d in train:
 key=(d['global_step'],d['task_id'])
 if key in seen:continue
 seen.add(key)
 arc=run/d['research']['research_archive']; meta=json.loads((arc/'meta.json').read_text()); a=json.loads(meta['self_assessment']);doc=(arc/'doc.md').read_text()
 instruction=re.search(r'Instruction:\s*\[SEP\]\s*(.*?)\s*\[SEP\]',d['trajectory']['initial_obs'],re.S).group(1)
 expected=exact_specs(instruction)
 actual=json.loads(re.search(r'Exact required option values from your instruction: (\{.*?\})\.',doc).group(1))
 if expected!=actual or actual!=a['required_options']:issues.append({'group':key,'expected':expected,'actual':actual})
 samples.append({'group':key,'exact_specs':expected,'origin':a['origin'],'exploration_successes':a['successes'],'words':len(doc.split())})
x['spec_audit']={'groups':len(seen),'issues':issues,'samples':samples}
roles=collections.defaultdict(list)
for p in run.rglob('*.jsonl'):
 if 'tape' not in str(p):continue
 for line in p.open():
  try:d=json.loads(line)
  except ValueError:continue
  if d.get('usage',{}).get('prompt_tokens'):roles[d.get('role','unknown')].append(d['usage']['prompt_tokens'])
x['input_by_role']={k:{'calls':len(v),'mean':sum(v)/len(v),'max':max(v),'above3072':sum(t>3072 for t in v)} for k,v in roles.items()}
p=Path('/tmp/p5compact3072-clean25v1/command.log'); log=p.read_text(errors='replace');x['training_metric_lines']=[l for l in log.splitlines() if 'timing_s/step' in l][-3:];x['log_tail']=log[-6500:]
x['gate_config']={k:getattr(cfg.steering,k) for k in ['gate_mode','entropy_theta_init','entropy_theta_update','entropy_target_rate']}
src=(R/'code/src/trace_opd/steering/gate.py').read_text();x['gate_warmup_code']=src[src.index('#: 自标定 warmup'):src.index('class GateThresholdTracker')]
subprocess.run([sys.executable,str(R/'inspect_runtime.py')],stdout=subprocess.DEVNULL,check=True)
x['snapshot']=json.loads((R/'runtime_snapshot.json').read_text())
(R/'review_0326.txt').write_text(base64.b64encode(gzip.compress(json.dumps(x).encode())).decode())
print('REVIEW_READY',x['steering_by_step'], 'SPEC_ISSUES',len(issues),'COMPLETED',x['snapshot']['completed_steps'])
