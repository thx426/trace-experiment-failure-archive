"""CPU-only read of saved experiment evidence. No inference or training."""
from pathlib import Path
import json, gzip, base64, datetime, hashlib, difflib, re, yaml, sys, os
import pyarrow.parquet as pq
R=Path('/root/trace-webshop-arm-ratio/experiments/pass5_clean_skill_20261025_v2')
B=R.parent/'pass5_20261023'
os.environ['TRACE_LOCAL_RESEARCH_KEY']='loopback-only'
sys.path.insert(0,str(R))
from audit_run import load_config
runs={'clean':R/'code/runs/pass5_clean_skill_v2_s42','grpo':B/'code/runs/pass5_grpo_s42'}
out={'time':datetime.datetime.now().astimezone().isoformat(),'configs':{},'datasets':{},'trajectories':{},'sources':{},'research_docs':[]}
def strip_private(d):
    if isinstance(d,dict):
        return {k:strip_private(v) for k,v in d.items() if not any(s in k.lower() for s in ['thought','reasoning','message','prompt','completion','raw_response'])}
    if isinstance(d,list):return [strip_private(v) for v in d]
    return d
for arm,root in [('clean',R),('grpo',B)]:
    cfg=yaml.safe_load((root/('configs/clean_skill.yaml' if arm=='clean' else 'configs/grpo.yaml')).read_text())
    resolved=load_config(root/('configs/clean_skill.yaml' if arm=='clean' else 'configs/grpo.yaml'))
    out['configs'][arm]={k:cfg.get(k) for k in ['env','train','eval','steering']}
    out['datasets'][arm]={}
    for split,field in [('train','train_files'),('dev','val_files')]:
        rows=[]
        paths=[resolved.train.train_file] if split=='train' else resolved.train.val_files
        for f in paths:rows.extend(pq.read_table(f).to_pylist())
        out['datasets'][arm][split]=[{'task_id':d['extra_info']['task_id'],'extra_info':strip_private(d['extra_info'])} for d in rows]
    vals=[]
    for p in sorted((runs[arm]/'rollouts/step_00100_val').glob('*.json')):
        d=json.loads(p.read_text()); vals.append((p,d))
    out['trajectories'][arm]=[]
    # Export all eval actions to permit complete outcome and behaviour comparisons.
    for p,d in vals:
        t=d.get('trajectory',{})
        out['trajectories'][arm].append({'file':p.name,**{k:strip_private(v) for k,v in d.items() if k not in ['trajectory','messages','prompt','response','thoughts'] and not isinstance(v,(list,dict))},
          'steering':d.get('steering'),'research':d.get('research'),
          'trajectory':strip_private(t)})
    for rel in ['src/trace_opd/train/agent_loop.py','src/trace_opd/steering/arms.py','src/trace_opd/envs/webshop.py','scripts/launch_train.py']:
        p=root/'code'/rel
        if p.exists():out['sources'][arm+':'+rel]=p.read_text()
out['compiler']=(R/'clean_skill.py').read_text()
for p in sorted((runs['clean']/'research').glob('*/*/meta.json')):
    meta=json.loads(p.read_text())
    docs={str(x.name):x.read_text() for x in p.parent.iterdir() if x.is_file() and x.suffix in ['.md','.txt'] and x.stat().st_size<30000}
    out['research_docs'].append({'path':str(p), 'meta':strip_private(meta),'docs':docs})
out['sequence']=json.loads((R/'sequence_status.json').read_text())
out['manifest_changed']=[p for p,h in json.loads((R/'frozen_manifest.json').read_text()).items() if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
raw=json.dumps(out,ensure_ascii=False).encode()
target=R/'step100_action_audit.txt'
target.write_text(base64.b64encode(gzip.compress(raw)).decode())
print('ACTION_AUDIT_READY',target.name,'bytes',len(raw),'encoded',target.stat().st_size,'rows',{a:len(v) for a,v in out['trajectories'].items()},'docs',len(out['research_docs']),'manifest_changed',out['manifest_changed'])
