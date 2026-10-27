"""CPU-only supplementary audit; executes existing inspector, never mutates training."""
from pathlib import Path
import base64, collections, datetime, gzip, hashlib, json, os, re, subprocess, sys

R = Path('/root/trace-webshop-arm-ratio/experiments/pass5_clean_skill_20261025_v2')
PY = '/root/trace-webshop-arm-ratio/.venvs/train/bin/python'
os.environ['TRACE_LOCAL_RESEARCH_KEY'] = 'loopback-only'
raw = subprocess.check_output([PY, str(R/'inspect_runtime.py')], text=True)
snapshot = json.loads(raw)
run = Path(snapshot['sequence'].get('run', str(R/'code/runs/pass5_clean_skill_v2_s42')))
out = {'snapshot': snapshot, 'sequence_pid_file': (R/'sequence.pid').read_text(),
       'sequence_log': (R/'sequence.log').read_text(),
       'exported_at': datetime.datetime.now().astimezone().isoformat()}
out['frozen_changed'] = [p for p,h in json.loads((R/'frozen_manifest.json').read_text()).items()
                         if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
out['abort'] = (R/'ABORT').read_text() if (R/'ABORT').exists() else None
out['owner_status'] = {}
metric_lines = []
for owner in ('p5compact3072-clean25v2','p5compact3072-research25clean2'):
    path = Path('/tmp')/owner/'command.log'
    log = path.read_text(errors='replace')
    out[owner+'_tail'] = log[-16000:]
    out[owner+'_mtime'] = path.stat().st_mtime
    patterns = ['CUDA out of memory','OutOfMemoryError','is preempted|preempted by',
                'ReadTimeout|ConnectTimeout','Traceback','HTTP/1.1. 500','HTTP/1.1. 409']
    out[owner+'_errors'] = {p:len(re.findall(p,log)) for p in patterns}
    out[owner+'_tracebacks'] = [log[max(0,m.start()-250):m.start()+2500] for m in re.finditer('Traceback',log)][-4:]
    s = path.parent/'status.json'
    out['owner_status'][owner] = json.loads(s.read_text()) if s.exists() else None
    if owner.endswith('clean25v2'):
        metric_lines = [line for line in log.splitlines() if 'timing_s/step' in line]
        out['update_metric_lines'] = metric_lines[-12:]
        gate_warnings = [line for line in log.splitlines() if 'gate tracker step' in line]
        out['gate_warning_audit'] = {
            'below_floor_lines': [line for line in gate_warnings if 'intervention starved' in line],
            'above_ceiling_line_count': sum('dosage overshoot' in line for line in gate_warnings),
            'note': 'Worker histogram warnings at step boundaries; not aggregate candidate rates or missing steering. Log counts may be affected by Ray deduplication.'}
    else:
        out['research_server_http_statuses'] = dict(collections.Counter(re.findall(r'POST /v1/chat/completions HTTP/1.1" (\d+)',log)))
out['actual_update_evidence'] = []
for line in metric_lines:
    step = re.search(r'step:\s*(\d+)',line)
    if not step: continue
    item = {'training/global_step':int(step.group(1))}
    for key in ('actor/grad_norm','actor/lr','critic/advantages/min','critic/advantages/max',
                'timing_s/update_actor','timing_s/update_weights','timing_s/step'):
        m = re.search(re.escape(key)+r':\s*([0-9.eE+\-]+)',line)
        item[key] = float(m.group(1)) if m else None
    out['actual_update_evidence'].append(item)
out['gpu_process_identity'] = {}
for line in snapshot['gpu_processes'].splitlines()[1:]:
    pid = line.split(',')[0].strip()
    p = Path('/proc')/pid/'cmdline'
    out['gpu_process_identity'][pid] = p.read_bytes().replace(b'\0',b' ').decode(errors='replace') if p.exists() else None

sys.path.insert(0,str(R))
from audit_run import read_evaluation, load_config, require
import pyarrow.parquet as pq
cfg = load_config(R/'configs/clean_skill.yaml')
ids = {d['extra_info']['task_id'] for p in cfg.train.val_files for d in pq.read_table(p).to_pylist()}
trainids = {d['extra_info']['task_id'] for d in pq.read_table(cfg.train.train_file).to_pylist()}
manifests = list(run.glob('launch_manifest_*.json'))
require(len(manifests)==1,'multiple launches')
launch = json.loads(manifests[0].read_text())['launch_id']
detail = {'checked_at':datetime.datetime.now().astimezone().isoformat(),'train_steps':{},'posttraining_evaluations':{}}
for directory in sorted((run/'rollouts').glob('step_*_val')):
    step = int(directory.name.split('_')[1]); count = len(list(directory.glob('*.json')))
    ev = {'rows':count,'expected_rows':len(ids)*cfg.eval.n,'complete':count==len(ids)*cfg.eval.n}
    if ev['complete']:
        try:
            vals = read_evaluation(run,launch,step,ids,cfg.eval.n,cfg.eval.temperature)
            ev.update(passed=True,tasks=len(vals),successes=sum(map(sum,vals.values())),solved=sum(any(x) for x in vals.values()),
                      step=step,bare=True,n=cfg.eval.n,temperature=cfg.eval.temperature)
            ev['pass1_estimate']=ev['successes']/count;ev['pass5']=ev['solved']/len(vals)
        except Exception as e:ev.update(passed=False,error=str(e))
    if step==0:detail['initial_eval']=ev
    else:detail['posttraining_evaluations'][str(step)]=ev
for step in snapshot['completed_steps']:
    rows = [json.loads(p.read_text()) for p in (run/'rollouts'/f'step_{step:05d}').glob('*.json')]
    issues=[];groups=collections.defaultdict(list);seen=set();guided=[]
    for d in rows:
        key=(d['task_id'],d['session_id'])
        if key in seen or d['task_id'] not in trainids:issues.append('identity')
        seen.add(key);groups[d['task_id']].append(d)
        if d.get('validate') is not False or d.get('split')!='train' or d.get('global_step')!=step or d.get('outcome')=='ENV_ERROR':issues.append('row contract')
        st=d.get('steering',{});rs=d.get('research',{})
        if st.get('steer_arm')!=int(d['session_id']<4):issues.append('guided assignment')
        if rs.get('research_error')!=0 or rs.get('research_doc_missing')!=0 or rs.get('research_doc_words',0)<=0:issues.append('research/doc')
        if rs.get('research_policy_version')!=step-1:issues.append('research policy')
        if st.get('steer_arm')==1:guided.append(st)
    if len(rows)!=128 or len(groups)!=16 or any({d['session_id'] for d in ds}!=set(range(8)) for ds in groups.values()):issues.append('incomplete groups')
    detail['train_steps'][str(step)]={'passed':not issues,'issues':dict(collections.Counter(issues)),
        'rows':len(rows),'groups':len(groups),'won_heterogeneous_groups':sum(len({x['won'] for x in ds})>1 for ds in groups.values()),
        'score_heterogeneous_groups':sum(len({x['score'] for x in ds})>1 for ds in groups.values()),
        'binary_successes':sum(d['won'] for d in rows),'guided_rows':len(guided),
        'theta_min':min(s.get('steer_entropy_theta',-1) for s in guided),
        'theta_max':max(s.get('steer_entropy_theta',-1) for s in guided),
        'positive_steering_rows':sum(s.get('steer_count',0)>0 for s in guided),
        'steer_count_total':sum(s.get('steer_count',0) for s in guided),
        'note':'group heterogeneity is an advantage proxy; actual advantage extrema and gradient norms logged separately'}
out['detail_audit']=detail
out['completion_audit']=json.loads((run/'completion_audit.json').read_text()) if (run/'completion_audit.json').exists() else None
out['latest_checkpoint']=(run/'checkpoints/latest_checkpointed_iteration.txt').read_text() if (run/'checkpoints/latest_checkpointed_iteration.txt').exists() else None
out['config_contract']={'researcher_model':cfg.research.model if hasattr(cfg,'research') and hasattr(cfg.research,'model') else None,
    'eval_n':cfg.eval.n,'eval_temperature':cfg.eval.temperature,'train_rollout_n':cfg.train.rollout_n,
    'guided_fraction':cfg.steering.steered_fraction,'config_unchanged':not out['frozen_changed']}
name = sys.argv[1] if len(sys.argv)>1 else datetime.datetime.now().strftime('monitor_%Y%m%d_%H%M%S.txt')
(R/name).write_text(base64.b64encode(gzip.compress(json.dumps(out,ensure_ascii=False).encode())).decode())
print('MONITOR_READY',name,'updates',snapshot['completed_steps'][-1:],'inputs',snapshot['researcher_inputs'],
      'skills',snapshot['skill_compiler'],'errors',out['p5compact3072-research25clean2_errors'])
