from pathlib import Path
import json,datetime,httpx,collections,re,os
R=Path(__file__).resolve().parent
sequence=json.loads((R/'sequence_status.json').read_text())
out={'checked_at':datetime.datetime.now().astimezone().isoformat(),'sequence':sequence,'webshop':{}}
for port in range(8321,8325):
    try:
        m=httpx.get(f'http://127.0.0.1:{port}/meta',timeout=5).json()
        log=(R/f'webshop-{port}.log').read_text(errors='replace')
        out['webshop'][str(port)]={'active':m['active'],'stats':m['transport_stats'],
            'http_statuses':dict(collections.Counter(re.findall(r'POST /(?:step|reset|release) HTTP/1.1" (\d+)',log))),
            'disconnects':log.count('client disconnected'),'tracebacks':log.count('Traceback')}
    except Exception as e:out['webshop'][str(port)]={'error':type(e).__name__}
run=Path(sequence.get('run',str(R/'code/runs/pass5_clean_skill_v2_s42')))
out['rollout_rows']={p.name:len(list(p.glob('*.json'))) for p in (run/'rollouts').glob('*') if p.is_dir()}
logpath=Path(sequence.get('owner_state','/tmp/p5compact3072-clean25v2'))/'command.log'
if logpath.exists():
    text=logpath.read_text(errors='replace')
    out['error_mentions']={k:text.count(k) for k in ['webshop service unreachable','webshop HTTP 409','student episode infra error','OutOfMemoryError','ReplayBuffer:']}
    out['training_tail']=text.splitlines()[-4:]
    lines=[s for s in text.splitlines() if 'timing_s/step' in s]
    out['completed_steps']=[int(m.group(1)) for s in lines if (m:=re.search(r'step:\s*(\d+)',s))]
    out['timing_rows']=[{k:float(v) for k,v in re.findall(r'(timing_s/[a-z_]+):([0-9.eE+\-]+)',s)} for s in lines]
actual=[]
for p in run.rglob('*.jsonl'):
    if 'tape' not in str(p):continue
    for line in p.open(errors='replace'):
        try:d=json.loads(line)
        except json.JSONDecodeError:continue
        if d.get('role')=='researcher' and d.get('usage'):
            n=d['usage'].get('prompt_tokens',0)
            if n:actual.append(n)
out['researcher_inputs']={'calls':len(actual),'mean':sum(actual)/len(actual) if actual else None,'max':max(actual) if actual else None,'above_3072':sum(n>3072 for n in actual)}

import subprocess
out['experiment_root']=str(R)
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv'],text=True)
out['gpu_processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv'],text=True)
out['process_identity']={}
for label,pid in [('sequence',sequence.get('pid')),('student_owner',sequence.get('owner_pid')),('researcher_owner',sequence.get('researcher_owner_pid'))]:
    cmd=Path(f'/proc/{pid}/cmdline')
    out['process_identity'][label]={'pid':pid,'alive':cmd.exists(),'cmdline':cmd.read_bytes().replace(bytes([0]),b' ').decode(errors='replace') if cmd.exists() else None}
skills=[]
for p in (run/'research').rglob('meta.json'):
    try:m=json.loads(p.read_text())
    except (ValueError,OSError):continue
    try:a=json.loads(m.get('self_assessment','{}'))
    except (ValueError,TypeError):a={}
    skills.append({'step':m.get('global_step'),'task_id':m.get('task_id'),'doc_words':m.get('doc_words',0),
        'origin':a.get('origin','unlabelled'),'version':a.get('version'),'error':m.get('error'),
        'turns':m.get('turns_used'),'experiments':m.get('experiments_used'),'wall_s':m.get('research_wall_s')})
out['skill_compiler']={'sessions':len(skills),'missing_docs':sum(not x['doc_words'] for x in skills),
    'over_500_words':sum(x['doc_words']>500 for x in skills),'origins':dict(collections.Counter(x['origin'] for x in skills)),
    'max_words':max((x['doc_words'] for x in skills),default=None),'research_errors':sum(bool(x['error']) for x in skills),
    'mean_turns':sum(x['turns'] or 0 for x in skills)/len(skills) if skills else None}
groups=collections.defaultdict(list);guided=[]
for p in (run/'rollouts').glob('step_*/*.json'):
    try:d=json.loads(p.read_text())
    except (ValueError,OSError):continue
    if d.get('validate'):continue
    groups[(d['global_step'],d['task_id'])].append(d['score'])
    if d.get('steering',{}).get('steer_arm')==1:guided.append(d['steering'].get('steer_count',0))
out['training_diagnostics']={'complete_groups':sum(len(v)==8 for v in groups.values()),
    'score_heterogeneous_complete_groups':sum(len(v)==8 and len(set(v))>1 for v in groups.values()),
    'advantage_proxy_only':True,'guided_assigned_rows':len(guided),'positive_steering_rows':sum(x>0 for x in guided)}
for name in ['baseline_reviews','baseline_stop','cpu_validation']:
    f=R/(name+'.json')
    out[name]=json.loads(f.read_text()) if f.exists() else None
out['last_progress_mtime']=max((p.stat().st_mtime for root in [run/'rollouts',run/'tape',run/'research'] for p in root.rglob('*') if p.is_file()),default=None)
out['preflight_note']='researcher_inputs contains role=researcher only; startup tokenize and researcher-preflight excluded. Zero calls means no actual formal-input samples.'

# Independent public-clause audit; do not reuse compiler exact_specs.
import yaml
from spec_oracle import public_clause_oracle, document_options
import pyarrow.parquet as pq
cfg=yaml.safe_load((R/'configs/clean_skill.yaml').read_text())
instructions={d['extra_info']['task_id']:d['extra_info']['instruction'] for d in pq.read_table(cfg['train']['train_file']).to_pylist()}
issues=[];checked=0
for p in (run/'research').rglob('meta.json'):
    m=json.loads(p.read_text());doc=p.parent/'doc.md'
    if not doc.exists():continue
    try:
        wanted=public_clause_oracle(instructions[m['task_id']]);actual=document_options(doc.read_text())
        if wanted!=actual:issues.append({'path':str(p),'expected':wanted,'actual':actual})
        checked+=1
    except Exception as e:issues.append({'path':str(p),'error':str(e)})
out['independent_spec_audit']={'checked':checked,'issues':issues}
by_step=collections.defaultdict(list)
for p in (run/'rollouts').glob('step_*/*.json'):
    d=json.loads(p.read_text())
    if not d.get('validate') and d.get('steering',{}).get('steer_arm')==1:by_step[d['global_step']].append(d['steering'])
out['steering_by_step']={step:{'guided_rows':len(v),'warmup_rows':sum(s.get('steer_entropy_theta')==-1 for s in v),'positive_steering_rows':sum(s.get('steer_count',0)>0 for s in v),'steer_count_total':sum(s.get('steer_count',0) for s in v)} for step,v in by_step.items()}
(R/'runtime_snapshot.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
