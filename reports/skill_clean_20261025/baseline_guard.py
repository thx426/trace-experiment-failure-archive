"""Stop the owned formal run after a complete worse bare checkpoint evaluation."""
from pathlib import Path
import json,time

def below(a,b):
    return a['successes'] < b['successes'] or a['solved'] < b['solved']

def summarize(ev):
    return {'tasks':len(ev),'attempts':sum(map(len,ev.values())),
            'successes':sum(map(sum,ev.values())), 'solved':sum(any(v) for v in ev.values())}

def review_completed_evaluations(root,run,cfg):
    import pyarrow.parquet as pq
    from audit_run import read_evaluation,require
    baseline=root.parent/'pass5_20261023/code/runs/pass5_grpo_s42'
    ids={x['extra_info']['task_id'] for p in cfg.train.val_files for x in pq.read_table(p).to_pylist()}
    statepath=root/'baseline_reviews.json'
    reviews=json.loads(statepath.read_text()) if statepath.exists() else {}
    for directory in sorted((run/'rollouts').glob('step_*_val')):
        step=int(directory.name.split('_')[1])
        if step==0 or str(step) in reviews:continue
        rows=list(directory.glob('*.json'))
        if len(rows)<len(ids)*cfg.eval.n:continue
        manifests=list(run.glob('launch_manifest_*.json'));require(len(manifests)==1,'multiple launches')
        basem=list(baseline.glob('launch_manifest_*.json'));require(len(basem)==1,'ambiguous baseline')
        cur=read_evaluation(run,json.loads(manifests[0].read_text())['launch_id'],step,ids,cfg.eval.n,cfg.eval.temperature)
        base=read_evaluation(baseline,json.loads(basem[0].read_text())['launch_id'],step,ids,cfg.eval.n,cfg.eval.temperature)
        a,b=summarize(cur),summarize(base)
        review={'step':step,'time':time.time(),'new':a,'grpo':b,'below_baseline':below(a,b),
                'criterion':'stop if either bare pass@1 estimate or pass@5 is below same-step GRPO',
                'audit':'complete single-launch 256 x 5 bare evaluation; no infrastructure error'}
        reviews[str(step)]=review
        tmp=statepath.with_suffix('.tmp');tmp.write_text(json.dumps(reviews,indent=2));tmp.replace(statepath)
        if review['below_baseline']:
            (root/'baseline_stop.json').write_text(json.dumps(review,indent=2))
            (root/'ABORT').write_text(f'User baseline stop rule at checkpoint {step}; do not restart this variant.\n')
            raise RuntimeError(f'Clean-Skill below GRPO at checkpoint {step}')
