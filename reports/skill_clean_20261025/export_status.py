from pathlib import Path
import json,gzip,base64,datetime,subprocess
R=Path('/root/trace-webshop-arm-ratio/experiments/pass5_clean_skill_20261025_v1')
subprocess.run(['/root/trace-webshop-arm-ratio/.venvs/train/bin/python',str(R/'inspect_runtime.py')],stdout=subprocess.DEVNULL,check=True)
out={'exported_at':datetime.datetime.now().astimezone().isoformat()}
for name in ['runtime_snapshot.json','cpu_validation.json','experiment_contract.json','sequence_status.json','baseline_reviews.json']:
 p=R/name
 if p.exists():out[name]=json.loads(p.read_text())
out['sequence_log']=(R/'sequence.log').read_text()[-5000:]
for owner in ['/tmp/p5compact3072-clean25v1','/tmp/p5compact3072-research25clean1']:
 p=Path(owner)/'status.json'
 if p.exists():out[owner]=json.loads(p.read_text())
 p=Path(owner)/'command.log'
 if p.exists():out[owner+'_log_tail']=p.read_text(errors='replace')[-5000:]
old=R.parent/'pass5_compact3072_20261024_r3'
d=json.loads((old/'missing_skill_diagnosis_20261025.json').read_text())
targets=[g for g in d['groups'] if g['fields']['research_doc_missing'] and g['last_messages'][-1]['role']=='assistant']
out['early_no_finalize_tape']=[]
for path in d['tape_paths']:
 for line in Path(path).open():
  if not any(t['task_id'] in line for t in targets):continue
  row=json.loads(line)
  for t in targets:
   if row.get('ctx','').startswith('research:'+t['task_id']+':') and t['meta']['ts']-t['meta']['research_wall_s']-5 < row['ts'] <= t['meta']['ts']:
    out['early_no_finalize_tape'].append({'step':t['step'],'task_id':t['task_id'],'ctx':row['ctx'],'finish_reason':row['finish_reason'],'usage':row.get('usage'),'ts':row['ts']})
encoded=base64.b64encode(gzip.compress(json.dumps(out,ensure_ascii=False).encode())).decode()
(R/'startup_review.txt').write_text(encoded)
print('STARTUP_EXPORT_READY',len(encoded),out['sequence_status.json']['phase'])
