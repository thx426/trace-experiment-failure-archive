from pathlib import Path
import json, yaml, datetime, gzip, base64, re

R=Path('/root/trace-webshop-arm-ratio/experiments/pass5_clean_skill_20261025_v2')
run=R/'code/runs/pass5_clean_skill_v2_s42'
out={'checked_at':datetime.datetime.now().astimezone().isoformat(),'configs':{}}
roots={'clean':R/'configs/clean_skill.yaml','selective':R.parent/'pass5_compact3072_20261024_r3/configs/selective_skill.yaml','grpo':R.parent/'pass5_20261023/configs/grpo.yaml'}
fields={'train':['adv_estimator','norm_adv_by_std','seed','train_batch_size','rollout_n','rollout_temperature','total_training_steps','model_path','reward_from_score','test_freq','lr'], 'eval':['n','temperature'], 'steering':['enabled','doc_source','task_gate_path','steered_fraction','steered_adv_decay_steps','w_beh_mode'], 'research':['request_token_limit','n_exp','k_default','max_turns']}
for arm,p in roots.items():
 c=yaml.safe_load(p.read_text())
 out['configs'][arm]={'path':str(p),**{section:{k:c.get(section,{}).get(k) for k in keys} for section,keys in fields.items()}}
manifest=json.loads(next(run.glob('launch_manifest_*.json')).read_text())
args=manifest['cmd'];args=args.split() if isinstance(args,str) else args
out['clean_actual_launch']=[x for x in args if any(k in x for k in ['adv_estimator=','norm_adv_by_std','data.seed=','rollout.seed=','data_loader_seed=','total_training_steps=','train_batch_size=','rollout.n=','val_kwargs.','steered_adv'])]
error_t=datetime.datetime.fromisoformat('2026-10-25T18:19:11+08:00').timestamp()
allrows=[]
for p in (run/'tape').glob('researcher.*.jsonl'):
 for i,line in enumerate(p.read_text().splitlines()):
  d=json.loads(line)
  if abs(d.get('ts',0)-error_t)>600:continue
  resp=d.get('response') or {};content=resp.get('content') or ''
  allrows.append({'path':str(p),'line':i+1,**{k:d.get(k) for k in ['ts','ctx','finish_reason','latency_s','usage','error']},'tools':[t.get('function',{}).get('name') for t in resp.get('tool_calls',[])],'parameter_headers':re.findall(r'<parameter=[^\n]*',content)[:10],'function_names':re.findall(r'<function=([^>\n]+)',content)})
out['candidate_records']=[d for d in allrows if abs(d['ts']-error_t)<4]
tids={d['ctx'].split(':')[1] for d in out['candidate_records'] if d.get('ctx') and not d['tools']}
out['followup']=[d for d in allrows if d.get('ctx') and d['ctx'].split(':')[1] in tids]
out['metadata']=[]
for tid in tids:
 for p in (run/'research').glob(tid+'*/*/meta.json'):
  d=json.loads(p.read_text())
  if abs(d.get('ts',0)-error_t)<600:
   out['metadata'].append({'path':str(p),**{k:d.get(k) for k in ['task_id','policy_version','global_step','finalized','no_finalize','turns_used','experiments_used','doc_words','self_assessment','flags']}})
out['sequence']=json.loads((R/'sequence_status.json').read_text())
out['final_eval_rows']=len(list((run/'rollouts/step_00100_val').glob('*.json')))
out['baseline_reviews']=json.loads((R/'baseline_reviews.json').read_text())
out['completion_audit']=json.loads((run/'completion_audit.json').read_text()) if (run/'completion_audit.json').exists() else None
(R/'method_contract_1845.txt').write_text(base64.b64encode(gzip.compress(json.dumps(out).encode())).decode())
print('METHOD_CONTRACT_READY',out['sequence']['phase'],out['final_eval_rows'])
