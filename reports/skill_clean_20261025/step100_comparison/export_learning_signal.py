"""Summarize existing training outcomes, not a new model run."""
from pathlib import Path
import json,collections,sys,os,gzip,base64,datetime
R=Path('/root/trace-webshop-arm-ratio/experiments/pass5_clean_skill_20261025_v2')
os.environ['TRACE_LOCAL_RESEARCH_KEY']='loopback-only'
sys.path.insert(0,str(R))
from audit_run import load_config
runs={'clean':R/'code/runs/pass5_clean_skill_v2_s42','grpo':R.parent/'pass5_20261023/code/runs/pass5_grpo_s42'}
out={'time':datetime.datetime.now().astimezone().isoformat(),'groups':{},'reward_from_score':{}}
for arm,run in runs.items():
    cfg=load_config((R/'configs/clean_skill.yaml') if arm=='clean' else (R.parent/'pass5_20261023/configs/grpo.yaml'))
    out['reward_from_score'][arm]=cfg.train.reward_from_score
    groups=[]
    for step in range(1,101):
        byid=collections.defaultdict(list)
        for p in sorted((run/'rollouts'/f'step_{step:05d}').glob('*.json')):
            d=json.loads(p.read_text());st=d.get('steering',{})
            byid[d['task_id']].append({'sid':d['session_id'],'won':int(d['won']),'score':d['score'],'arm':st.get('steer_arm',0),'steer_count':st.get('steer_count',0),'sample_id':d['sample_id']})
        for tid,rows in byid.items():
            assert len(rows)==8 and {x['sid'] for x in rows}==set(range(8))
            groups.append({'step':step,'task_id':tid,'rows':sorted(rows,key=lambda x:x['sid'])})
    out['groups'][arm]=groups
out['same_step_task_sets']=all({(g['step'],g['task_id']) for g in out['groups'][a]}=={(g['step'],g['task_id']) for g in out['groups']['grpo']} for a in runs)
(R/'step100_learning_signal.txt').write_text(base64.b64encode(gzip.compress(json.dumps(out).encode())).decode())
print('SIGNAL_READY', {k:len(v) for k,v in out['groups'].items()},'same_task_sets',out['same_step_task_sets'],'reward_from_score',out['reward_from_score'])
