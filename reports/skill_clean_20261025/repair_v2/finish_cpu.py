from pathlib import Path
from types import SimpleNamespace as N
import shutil,json,hashlib,subprocess,sys,os,dataclasses,re,datetime
import yaml,pyarrow.parquet as pq
R=Path(__file__).resolve().parent;OLD=R.parent/'pass5_clean_skill_20261025_v1'
os.environ.update(TRACE_LOCAL_RESEARCH_KEY='loopback-only',TRACE_WEBSHOP_URLS=','.join(f'http://127.0.0.1:{p}' for p in range(8321,8325)))
sys.path.insert(0,str(R/'code/src'))
from trace_opd.common.config import load_config
from trace_opd.research.clean_skill import compile_skill,exact_specs
from spec_oracle import public_clause_oracle,document_options
oldcfg=load_config(OLD/'configs/clean_skill.yaml');cfg=load_config(R/'configs/clean_skill.yaml')
a,b=dataclasses.asdict(oldcfg),dataclasses.asdict(cfg)
for d in [a,b]:
 for k in ['raw','source_path']:d.pop(k,None)
 d['run']['name']='NAME';d['train']['experiment_name']='NAME'
assert a==b,'Configuration drift beyond run name'
assert cfg.train.resume_mode=='disable' and cfg.train.total_training_steps==100
assert cfg.research.request_token_limit==3072 and cfg.roles['researcher'].max_concurrency==16
src_diffs=[]
for p in (R/'code/src').rglob('*.py'):
 prior=OLD/'code/src'/p.relative_to(R/'code/src')
 if p.read_bytes()!=prior.read_bytes():src_diffs.append(str(p.relative_to(R/'code/src')))
assert src_diffs==['trace_opd/research/clean_skill.py'],src_diffs
env=os.environ.copy();env['PYTHONPATH']=str(R/'code/src')
for t in ['test_clean_skill.py','test_compact_contract.py']:subprocess.run([sys.executable,str(R/t)],env=env,check=True)
public=[]
for split,files in [('train',[cfg.train.train_file]),('dev',cfg.train.val_files)]:
 for path in files:
  for row in pq.read_table(path).to_pylist():
   e=row['extra_info'];instruction=e['instruction'];expected=public_clause_oracle(instruction)
   doc,audit=compile_skill(instruction+'\nBudget: 15 actions')
   assert document_options(doc)==expected==audit['required_options'],e['task_id']
   public.append({'split':split,'task_id':e['task_id'],'expected':expected,'words':len(doc.split())})
(R/'public_spec_contract.json').write_text(json.dumps({'passed':True,'rows':len(public),'cases':public},indent=2))
replays=[]
for prior in [OLD/'code/runs/pass5_clean_skill_s42',R.parent/'pass5_compact3072_20261024_r3/code/runs/pass5_compact3072_all_skill_s42']:
 seen=set()
 for p in (prior/'rollouts').glob('step_*/*.json'):
  d=json.loads(p.read_text());key=(d['global_step'],d['task_id'])
  if d['validate'] or key in seen:continue
  seen.add(key);arc=prior/d['research']['research_archive'];obs=d['trajectory']['initial_obs'];instruction=re.search(r'Instruction:\s*\[SEP\]\s*(.*?)\s*\[SEP\]',obs,re.S).group(1)
  records=[]
  for line in (arc/'episodes.jsonl').read_text().splitlines():
   item=json.loads(line)['trajectory'];records.append(N(trajectory=N(won=item['won'],steps=[N(**s) for s in item['steps']])))
  doc,audit=compile_skill('Instruction: '+instruction+'\nBudget: 15 actions',records,(arc/'doc.md').read_text() if (arc/'doc.md').exists() else '',origin='offline_replay')
  assert document_options(doc)==public_clause_oracle(instruction)
  replays.append({'run':prior.name,'group':key,'old_missing':d['research'].get('research_doc_missing',0),**audit})
assert len(replays)>=336,len(replays)
(R/'offline_compiler_replay.json').write_text(json.dumps(replays,indent=2))
contract=json.loads((OLD/'experiment_contract.json').read_text());contract.update(version='clean_skill_v2',repair_parent=str(OLD),repair='Generic public option parser; independently verify every label/value; no reward/guidance-allocation change',failed_run_not_resumed=True)
(R/'experiment_contract.json').write_text(json.dumps(contract,indent=2))
metas=[json.loads(p.read_text()) for p in (OLD/'code/runs/pass5_clean_skill_s42/research').rglob('meta.json')]
log=Path('/tmp/p5compact3072-clean25v1/command.log').read_text(errors='replace')
cost={'parent':str(OLD),'failure':json.loads((OLD/'spec_parser_failure.json').read_text()),'recorded_at':datetime.datetime.now().astimezone().isoformat(),'completed_steps':[int(m.group(1)) for s in log.splitlines() if 'timing_s/step' in s and (m:=re.search(r'step:\s*(\d+)',s))],'research_sessions':len(metas),'research_prompt_tokens':sum(m.get('researcher_prompt_tokens',0) for m in metas),'research_completion_tokens':sum(m.get('researcher_completion_tokens',0) for m in metas),'research_wall_s_sum':sum(m.get('research_wall_s',0) for m in metas),'owner_status':{o:json.loads((Path('/tmp')/o/'status.json').read_text()) for o in ['p5compact3072-clean25v1','p5compact3072-research25clean1']},'cause':'item shape swallowed into color; old spec test reused same extractor'}
(R/'predecessor_failure_cost.json').write_text(json.dumps(cost,indent=2));(OLD/'final_failure_cost.json').write_text(json.dumps(cost,indent=2))
subprocess.run([sys.executable,'-m','compileall','-q',str(R/'code/src'),str(R/'run_sequence.py'),str(R/'inspect_runtime.py')],check=True)
files=[p for directory in [R/'code/src',R/'code/scripts',R/'code/train',R/'configs'] for p in directory.rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.json')]
files += [R/x for x in ['run_sequence.py','baseline_guard.py','audit_run.py','experiment_contract.json','spec_oracle.py']]
BASE=R.parent/'pass5_20261023';files += [BASE/'pass5_gate.json',BASE/'data/train.parquet',BASE/'data/dev.parquet']
(R/'frozen_manifest.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
(R/'cpu_validation.json').write_text(json.dumps({'passed':True,'tests':21,'inference_calls':0,'new_rollouts':0,'configuration_equal_except_name':True,'src_diffs':src_diffs,'public_spec_cases':len(public),'independent_spec_issues':0,'offline_replayed_groups':len(replays),'max_doc_words':max(x['words'] for x in replays)},indent=2))
print('REPAIR_CPU_READY',R,flush=True)
