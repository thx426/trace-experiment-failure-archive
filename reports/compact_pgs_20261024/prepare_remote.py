"""Prepare an isolated formal run from the frozen previous experiment."""
from pathlib import Path
import hashlib,json,shutil,sys,subprocess,os
import yaml

R=Path(__file__).resolve().parent
OLD=R.parent/'pass5_20261023'
code=R/'code'
assert not code.exists(),'Refuse to overwrite prepared code'
code.mkdir()
for name in ['src','scripts','configs','train']:
    shutil.copytree(OLD/'code'/name,code/name,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
os.symlink((OLD/'code/verl').resolve(),code/'verl')
original=json.loads((R/'original_hashes.json').read_text())
for rel,digest in original.items():
    assert hashlib.sha256((OLD/'code'/rel).read_bytes()).hexdigest()==digest,rel
for p in (R/'patch').rglob('*.py'):
    rel=p.relative_to(R/'patch');dest=code/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
shutil.copy2(OLD/'audit_run.py',R/'audit_run.py')
(R/'configs').mkdir()
for arm in ['all_skill','selective_skill']:
    cfg=yaml.safe_load((OLD/f'configs/{arm}.yaml').read_text())
    name=f'pass5_compact3072_{arm}_s42'
    cfg['run']['name']=name;cfg['train']['experiment_name']=name
    cfg['research'].update(context_mode='bounded',request_token_limit=3072,context_target_tokens=2500,
                          context_tokenizer_path='/root/trace-webshop-arm-ratio/models/Qwen3.6-35B-A3B',
                          raw_page_tokens=512,max_prompt_tokens=3072)
    assert cfg['roles']['researcher']['max_concurrency']==16
    (R/f'configs/{arm}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True))
contract={'version':'compact3072_v1','request_token_limit':3072,'raw_page_tokens':512,
          'server_max_num_seqs':16,'server_max_model_len':7168,'max_tokens':4096,
          'steps_per_arm':100,'order':['all_skill','selective_skill'],
          'baseline':str(OLD/'code/runs/pass5_grpo_s42'),'pilot':False,'resume':False,
          'initial_student':'/root/trace-webshop-arm-ratio/models/Qwen2.5-3B-Instruct',
          'original_code':str(OLD/'code'),'approval':'User approved 2026-10-24; no pilot; strict input 3072 and concurrency16',
          'failure_policy':'Stop on contract violation, OOM, sustained preemption or failed completion audit; no automatic retry',
          'controlled_training':'Unchanged student, reward, data, seed, optimizer, group size, guidance fraction and evaluation',
          'research_changes':['bounded structured context','paged raw evidence','dynamic tools','concise document prompt','hard document word rejection after one repair']}
(R/'experiment_contract.json').write_text(json.dumps(contract,indent=2))
for directory in [code/'src',code/'scripts',code/'train']:
    subprocess.run([sys.executable,'-m','compileall','-q',str(directory)],check=True)
files=[p for directory in [code/'src',code/'scripts',code/'train',R/'configs'] for p in directory.rglob('*')
       if p.is_file() and p.suffix in ('.py','.yaml','.json')]
files += [R/'run_sequence.py',R/'audit_run.py',R/'experiment_contract.json',OLD/'pass5_gate.json',
          OLD/'data/train.parquet',OLD/'data/dev.parquet']
(R/'frozen_manifest.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
print('PREPARED',R,flush=True)
