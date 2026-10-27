import json,sys
from pathlib import Path
from collections import defaultdict,Counter
R=Path('/root/trace-webshop-arm-ratio/experiments/pass5_20261023');sys.path.insert(0,str(R))
from audit_run import load_config
import pyarrow.parquet as pq
c=load_config(R/'configs/grpo.yaml');run=R/'code'/c.run.runs_root/c.run.name
old=json.loads((R.parent/'grpo_final_review_20261023/report.json').read_text())
H=[t for t in old['evaluations']['100']['by_task'] if all(not any(old['evaluations'][str(s)]['by_task'][t]) for s in [60,80,100])]
trainids={x['extra_info']['task_id'] for x in pq.read_table(c.train.train_file).to_pylist()}
g=defaultdict(list);q=Counter();ends=Counter();seqs=defaultdict(set)
for p in (run/'rollouts').glob('step_*/*.json'):
 if p.parent.name.endswith('_val'):
  if p.parent.name not in [f'step_{s:05d}_val' for s in [60,80,100]]:continue
  r=json.loads(p.read_text())
  if r['task_id'] not in H:continue
  ss=r['trajectory']['steps'];q['episodes']+=1;q['positive_score']+=int(r['score']>0);q['score_sum']+=r['score'];q['turns']+=len(ss);q['invalid_actions']+=sum(x.get('valid') is False for x in ss);q['repeat_adjacent']+=sum(a.get('action_norm')==b.get('action_norm') for a,b in zip(ss,ss[1:]));ends[r['outcome']]+=1;seqs[r['task_id']].add(tuple(x.get('action_norm','') for x in ss))
 else:
  r=json.loads(p.read_text());g[(r['task_id'],r['global_step'])].append((int(r['won']),r['score']))
per=defaultdict(list)
for (t,s),v in sorted(g.items()):per[t].append(dict(step=s,wins=sum(x[0] for x in v),n=len(v),partial=sum(not x[0] and x[1]>0 for x in v)))
never={t:v for t,v in per.items() if sum(x['wins'] for x in v)==0}
late={t:v for t,v in per.items() if len([x for x in v if x['step']>60])>=2 and all(x['wins']==0 for x in v if x['step']>60)}
o=dict(dev_hard_count=len(H),train_intersection=sorted(trainids.intersection(H)),dev_hard_ids=H,dev_hard_quality=dict(q),dev_hard_outcomes=dict(ends),dev_hard_unique_action_sequences={t:len(v) for t,v in seqs.items()},train_never_success=never,train_late_zero=late,train_per_task=per)
p=R.parent/'grpo_final_review_20261023/hard_audit.json';p.write_text(json.dumps(o))
import gzip,base64,hashlib
raw=p.read_bytes();(p.parent/'hard_transfer.txt').write_text(base64.b64encode(gzip.compress(raw)).decode());print('DONE',len(H),len(trainids.intersection(H)),len(never),len(late),hashlib.sha256(raw).hexdigest(),flush=True)
print('QUALITY',dict(q));print('NEVER',never)
