"""Prepare isolated formal Clean-Skill run; CPU checks only, no model generation."""
from pathlib import Path
from types import SimpleNamespace as N
import shutil,json,hashlib,subprocess,sys,os,dataclasses,re
import yaml
R=Path(__file__).resolve().parent
OLD=R.parent/'pass5_compact3072_20261024_r3'
assert not (R/'code').exists(),'Refuse overwrite'
(R/'code').mkdir()
for name in ['src','scripts','configs','train']:
 shutil.copytree(OLD/'code'/name,R/'code'/name,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
os.symlink((OLD/'code/verl').resolve(),R/'code/verl')
for rel,digest in json.loads((R/'expected_source_hashes.json').read_text()).items():
 assert hashlib.sha256((OLD/'code'/rel).read_bytes()).hexdigest()==digest,rel
for p in (R/'patch').rglob('*.py'):
 dest=R/'code'/p.relative_to(R/'patch');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
shutil.copy2(OLD/'audit_run.py',R/'audit_run.py')
(R/'configs').mkdir()
cfg=yaml.safe_load((OLD/'configs/all_skill.yaml').read_text())
cfg['run']['name']='pass5_clean_skill_s42'
cfg['train']['experiment_name']='pass5_clean_skill_s42'
(R/'configs/clean_skill.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True))
os.environ.update(TRACE_WEBSHOP_URLS=','.join(f'http://127.0.0.1:{p}' for p in range(8321,8325)),TRACE_LOCAL_RESEARCH_KEY='loopback-only')
sys.path.insert(0,str(R/'code/src'))
from trace_opd.common.config import load_config
a=load_config(OLD/'configs/all_skill.yaml');b=load_config(R/'configs/clean_skill.yaml')
da,db=dataclasses.asdict(a),dataclasses.asdict(b)
for d in (da,db):
 for k in ['raw','source_path']:d.pop(k,None)
 d['run']['name']='NAME';d['train']['experiment_name']='NAME'
assert da==db,'Configuration drift beyond run name'
assert b.train.total_training_steps==100 and b.train.resume_mode=='disable'
assert b.research.request_token_limit==3072 and b.roles['researcher'].max_concurrency==16
env=os.environ.copy();env['PYTHONPATH']=str(R/'code/src')
subprocess.run([sys.executable,str(R/'test_clean_skill.py')],env=env,check=True)
subprocess.run([sys.executable,str(R/'test_compact_contract.py')],env=env,check=True)

# Replay archived evidence through the new compiler without running a policy.
from trace_opd.research.clean_skill import compile_skill
prior=OLD/'code/runs/pass5_compact3072_all_skill_s42'
replay=[]
for step in range(1,21):
 seen=set()
 for p in sorted((prior/'rollouts'/f'step_{step:05d}').glob('*.json')):
  row=json.loads(p.read_text());tid=row['task_id']
  if tid in seen:continue
  seen.add(tid);arc=prior/row['research']['research_archive']
  obs=row['trajectory']['initial_obs']
  instruction=re.search(r'Instruction:\s*\[SEP\]\s*(.*?)\s*\[SEP\]',obs,re.S).group(1)
  records=[]
  for line in (arc/'episodes.jsonl').read_text().splitlines():
   item=json.loads(line)['trajectory']
   records.append(N(trajectory=N(won=item['won'],steps=[N(**s) for s in item['steps']])))
  proposal=(arc/'doc.md').read_text() if (arc/'doc.md').exists() else ''
  doc,audit=compile_skill('Instruction: '+instruction+'\nBudget: 15 actions',records,proposal,'offline_replay')
  assert doc and audit['words']<=500
  replay.append({'step':step,'task_id':tid,'old_doc_missing':row['research']['research_doc_missing'],**audit})
assert len(replay)==320
(R/'offline_compiler_replay.json').write_text(json.dumps(replay,indent=2))
contract={'version':'clean_skill_v1','original_root':str(OLD),'order':['clean_skill'],'steps':100,
 'student_checkpoint':b.train.model_path,'researcher':'Qwen3.6-35B-A3B','input_limit':3072,'concurrency':16,
 'unchanged':['reward','student initial checkpoint','data splits','seed42','optimizer','GRPO objective','8 rollouts/task','0.5 guided fraction','bare eval n5 temperature0.6'],
 'changes':['mechanical evidence instead of narrative excerpt','reserve finalization turns','at most two fetch calls after experiment budget exhaustion','compile exploratory/final Skill from exact public specifications and checked evidence','label fallback checklist when normal session exhausts without finalize','abort on missing Skill or infrastructure failure','stop at complete same-step bare eval if pass1 or pass5 below GRPO'],
 'comparability':'New guidance variant, not pure infrastructure fix; compare to archived original PGS/GRPO, count all researcher cost. No claims that this isolates each component.',
 'evaluation':'Existing dev protocol is retained for stop decisions; not a new independent final test.',
 'fallback':'No new inference; task checklist only, explicitly labelled and never counted as exploration success.',
 'approval':'2026-10-25 user authorized stopping both old arms, minimum Skill clean-up, one new guided experiment and 30-minute review/repair.',
 'resume':'Disabled. On confirmed unrecoverable implementation fault, keep failed directory and restart in a new directory only after diagnosis and CPU contract checks; never auto-restart a below-baseline stop.'}
(R/'experiment_contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2))
subprocess.run([sys.executable,'-m','compileall','-q',str(R/'code/src'),str(R/'run_sequence.py'),str(R/'baseline_guard.py')],check=True)
files=[p for directory in [R/'code/src',R/'code/scripts',R/'code/train',R/'configs'] for p in directory.rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.json')]
files += [R/'run_sequence.py',R/'baseline_guard.py',R/'audit_run.py',R/'experiment_contract.json']
BASE=R.parent/'pass5_20261023';files += [BASE/'pass5_gate.json',BASE/'data/train.parquet',BASE/'data/dev.parquet']
(R/'frozen_manifest.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
(R/'cpu_validation.json').write_text(json.dumps({'passed':True,'inference_calls':0,'new_rollouts':0,'configuration_equal_except_name':True,'offline_replayed_groups':len(replay),'old_missing_covered':sum(x['old_doc_missing'] for x in replay),'max_doc_words':max(x['words'] for x in replay)},indent=2))
print('CLEAN_SKILL_CPU_READY',R,flush=True)
