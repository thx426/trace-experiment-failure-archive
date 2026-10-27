from pathlib import Path
from collections import Counter
import json,base64,gzip,hashlib,subprocess,datetime
R=Path('/root/trace-webshop-arm-ratio/experiments/pass5_compact3072_20261024_r3')
run=R/'code/runs/pass5_compact3072_all_skill_s42'
out={'checked_at':datetime.datetime.now().astimezone().isoformat(),'groups':[],'sources':{}}
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv'],text=True)
ps=subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True)
out['processes']=[s for s in ps.splitlines() if any(k in s for k in ['launch_train.py','run_sequence.py','run_owned_job.py','vllm serve','webshop_service.py']) and 'base64' not in s]
for step in range(1,21):
 seen=set()
 for p in sorted((run/'rollouts'/f'step_{step:05d}').glob('*.json')):
  row=json.loads(p.read_text()); tid=row['task_id']
  if tid in seen:continue
  seen.add(tid); r=row['research'];a=run/r['research_archive'];m=json.loads((a/'meta.json').read_text())
  msgs=[json.loads(s) for s in (a/'session.jsonl').read_text().splitlines()]
  calls=[]
  for i,msg in enumerate(msgs):
   for c in msg.get('tool_calls',[]):
    fn=c.get('function',c);args=fn.get('arguments',{})
    if isinstance(args,str):
     try:args=json.loads(args)
     except:args={'unparsed':args}
    entry={'i':i,'name':fn.get('name'),'doc_words':len(str(args.get('experience_doc','')).split()),'guidance_words':len(str(args.get('guidance','')).split())}
    if fn.get('name')=='finalize':entry['document']=args.get('experience_doc','')
    if i+1<len(msgs):entry['response']=str(msgs[i+1].get('content',''))[:1800]
    calls.append(entry)
  out['groups'].append({'step':step,'task_id':tid,'fields':r,'archive':str(a),'meta':m,'calls':calls,
    'last_messages':msgs[-3:],'assistant_count':sum(x['role']=='assistant' for x in msgs),
    'doc_exists':(a/'doc.md').exists(),'session_sha256':hashlib.sha256((a/'session.jsonl').read_bytes()).hexdigest()})
for rel in ['src/trace_opd/research/session.py','src/trace_opd/research/tools.py','src/trace_opd/research/compact.py','src/trace_opd/common/llm.py','src/trace_opd/docstore/archive.py','src/trace_opd/train/agent_loop.py','src/trace_opd/train/trainer.py','scripts/launch_train.py']:
 p=R/'code'/rel
 if p.exists():out['sources'][rel]=p.read_text()
for rel in ['run_sequence.py','inspect_runtime.py','audit_run.py','experiment_contract.json','prepare_repair.py','validate_remote.py']:
 out['sources'][rel]=(R/rel).read_text()
out['tape_paths']=[str(p) for p in run.rglob('*.jsonl') if 'tape' in str(p)]
if out['tape_paths']:
 lines=Path(out['tape_paths'][0]).read_text().splitlines();out['tape_sample']=[json.loads(x) for x in lines[:2]]
text=json.dumps(out,ensure_ascii=False)
(R/'missing_skill_diagnosis_20261025.json').write_text(text)
(R/'missing_skill_diagnosis_20261025.txt').write_text(base64.b64encode(gzip.compress(text.encode())).decode())
print('DIAGNOSIS_READY',len(out['groups']),dict(Counter((x['fields']['research_doc_missing'],x['meta']['turns_used'],x['meta']['finalized']) for x in out['groups'])),flush=True)
