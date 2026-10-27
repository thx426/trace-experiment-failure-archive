"""Offline export only: no model generation, training or service launch."""
from pathlib import Path
import json, sys, os, hashlib, random, collections, math, gzip, base64, datetime

R = Path('/root/trace-webshop-arm-ratio/experiments/pass5_compact3072_20261024_r3')
OLD = R.parent / 'pass5_20261023'
os.environ['TRACE_LOCAL_RESEARCH_KEY'] = 'loopback-only'
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / 'code/src'))
from audit_run import read_evaluation, load_config
import pyarrow.parquet as pq

cfg = load_config(R / 'configs/all_skill.yaml')
train_ids = {x['extra_info']['task_id'] for x in pq.read_table(cfg.train.train_file).to_pylist()}
dev_ids = {x['extra_info']['task_id'] for p in cfg.train.val_files for x in pq.read_table(p).to_pylist()}
gate = json.loads((OLD / 'pass5_gate.json').read_text())
runs = {'all_skill': R / 'code/runs/pass5_compact3072_all_skill_s42',
        'grpo': OLD / 'code/runs/pass5_grpo_s42'}
out = {'created_at': datetime.datetime.now().astimezone().isoformat(),
       'checkpoint_step': 20, 'sampling_seed': 20261024,
       'gate_metadata': {k:v for k,v in gate.items() if k != 'tasks'},
       'train_dev_overlap': sorted(train_ids & dev_ids),
       'run_paths': {k:str(v) for k,v in runs.items()},
       'evaluations': {}, 'audit': {}, 'training_summary': {}}
all_rows = {}
for arm, run in runs.items():
    manifests = list(run.glob('launch_manifest_*.json'))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    vals = read_evaluation(run, manifest['launch_id'], 20, dev_ids, cfg.eval.n, cfg.eval.temperature)
    out['evaluations'][arm] = vals
    out['audit'][arm] = {'passed':True, 'launch_id':manifest['launch_id'], 'n':cfg.eval.n,
                        'temperature':cfg.eval.temperature, 'tasks':len(vals),
                        'attempts':sum(map(len, vals.values()))}
    all_rows[arm] = {}
    for p in sorted((run / 'rollouts/step_00020_val').glob('*.json')):
        d = json.loads(p.read_text())
        d['source_path'] = str(p)
        d['source_sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
        all_rows[arm].setdefault(d['task_id'], []).append(d)
    groups = {}
    for step in range(1,21):
        ps = list((run/'rollouts'/f'step_{step:05d}').glob('*.json'))
        assert len(ps) == cfg.train.train_batch_size * cfg.train.rollout_n
        for p in ps:
            d=json.loads(p.read_text());g=groups.setdefault((step,d['task_id']),[]);g.append(d)
    research = [v[0]['research'] for v in groups.values()]
    out['training_summary'][arm] = {
        'steps':20,'groups':len(groups),'rows':sum(map(len,groups.values())),
        'score_heterogeneous_groups':sum(len({x['score'] for x in v})>1 for v in groups.values()),
        'score_variation_is_advantage_proxy_only':True,
        'research_doc_missing_groups':sum(x.get('research_doc_missing',0) for x in research),
        'research_error_groups':sum(x.get('research_error',0) for x in research),
        'research_doc_nonempty_groups':sum(x.get('research_doc_words',0)>0 for x in research),
        'arm_outcomes':{str(a):{'n':sum(1 for v in groups.values() for x in v if x.get('steering',{}).get('steer_arm',0)==a),
                              'won':sum(x['won'] for v in groups.values() for x in v if x.get('steering',{}).get('steer_arm',0)==a)} for a in [0,1]}}

index = []
for tid in sorted(dev_ids):
    g=gate['tasks'][tid]
    p=g['pass_at_5'];expected=1-math.comb(g['n']-g['c'],5)/math.comb(g['n'],5) if g['n']-g['c']>=5 else 1
    assert abs(p-expected)<1e-9
    a=sum(out['evaluations']['all_skill'][tid]);b=sum(out['evaluations']['grpo'][tid])
    outcome='both' if a and b else 'all_only' if a else 'grpo_only' if b else 'neither'
    bin_name='zero' if p==0 else 'low_positive' if g['stratum']=='low' else g['stratum']
    index.append({'task_id':tid, 'base_n':g['n'],'base_c':g['c'],'base_pass5':p,
                  'stratum':g['stratum'],'sampling_bin':bin_name,'grpo_successes':b,
                  'all_skill_successes':a,'outcome':outcome,'in_training_split':tid in train_ids})
out['index']=index
candidates=[x for x in index if x['outcome']=='grpo_only']
out['candidate_ids']=[x['task_id'] for x in candidates]
out['strata_counts']={s:dict(collections.Counter(x['outcome'] for x in index if x['stratum']==s)) for s in ['low','middle','high']}
out['sampling_bin_counts']={s:sum(x['sampling_bin']==s for x in candidates) for s in ['zero','low_positive','middle','high']}
rng=random.Random(out['sampling_seed']);selected=[]
for s in ['zero','low_positive','middle','high']:
    pool=sorted(x['task_id'] for x in candidates if x['sampling_bin']==s)
    selected.extend(rng.sample(pool,min(3,len(pool))))
out['selected_ids']=selected
out['sampling_rule']='Split low stratum into exactly zero and positive; draw min(3,N) uniformly without replacement per bin, sorted pools, Python Random seed 20261024. Empty bins not replaced.'
out['candidate_episodes']={}
for tid in out['candidate_ids']:
    out['candidate_episodes'][tid]={}
    for arm in runs:
        episode_rows=[]
        for row in sorted(all_rows[arm][tid],key=lambda x:x['session_id']):
            d={k:v for k,v in row.items() if k not in ['response_token_ids','turn_token_spans']}
            d['trajectory']={k:v for k,v in row['trajectory'].items() if k!='steps'}
            d['trajectory']['steps']=[{k:v for k,v in st.items() if k not in ['thought','action_raw']} for st in row['trajectory']['steps']]
            episode_rows.append(d)
        out['candidate_episodes'][tid][arm]=episode_rows
out['export_omissions']='Token IDs, turn token spans, generated thought text and redundant raw actions omitted. Normalized actions, observations, validity, quality metrics and source hashes preserved.'
out['configuration']={'all_skill':(R/'configs/all_skill.yaml').read_text()}
out['source_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'configs/all_skill.yaml',OLD/'pass5_gate.json',R/'audit_run.py']}
text=json.dumps(out,ensure_ascii=False)
(R/'stratified_failures_20261024.json').write_text(text)
encoded=base64.b64encode(gzip.compress(text.encode())).decode()
(R/'stratified_failures_20261024.txt').write_text(encoded)
print('STRATIFIED_EXPORT_COMPLETE',out['created_at'],'candidates',len(candidates),'bins',out['sampling_bin_counts'],'sample',selected,'transfer_chars',len(encoded),flush=True)
