"""Read saved evidence only. No model calls or environment interaction."""
from pathlib import Path
import json, hashlib, datetime

ROOT = Path('/root/trace-webshop-arm-ratio/experiments/pass5_clean_skill_20261025_v2')
RUN = ROOT / 'code/runs/pass5_clean_skill_v2_s42'
SELECTED = [('webshop_00510', 6), ('webshop_00512', 16),
            ('webshop_05104', 2), ('webshop_06866', 36), ('webshop_01418', 5)]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def trajectory(t):
    return {**{k: t.get(k) for k in ['episode_id','task_id','guidance_hash','sample_idx',
        'initial_obs','outcome','won','score','error']},
        'steps': [{k: s.get(k) for k in ['t','action_norm','observation','valid','acted']}
                  for s in t.get('steps', [])]}

out = {'export_time': datetime.datetime.now().astimezone().isoformat(),
       'root': str(ROOT), 'selection': 'purposeful mechanism cases, not a random sample',
       'inference_calls': 0, 'cases': [], 'sources': {}}
for tid, step in SELECTED:
    matches = [p for p in (RUN/'research').glob(tid+'*/*/meta.json')
               if json.loads(p.read_text())['global_step'] == step]
    assert len(matches) == 1, (tid, step, matches)
    mp = matches[0]; arc = mp.parent; meta = json.loads(mp.read_text())
    calls = []
    for number, line in enumerate((arc/'session.jsonl').read_text().splitlines(), 1):
        message = json.loads(line)
        for tc in message.get('tool_calls', []) or []:
            fun = tc['function']; args = json.loads(fun['arguments'])
            if fun['name'] in ['run_experiment', 'finalize']:
                calls.append({'line': number, 'name': fun['name'],
                              'arguments': {k:v for k,v in args.items()
                                  if k in ['guidance','k','experience_doc']}})
    episodes = [trajectory(json.loads(line)['trajectory'])
                for line in (arc/'episodes.jsonl').read_text().splitlines()]
    rolls = []
    for p in sorted((RUN/'rollouts'/f'step_{step:05d}').glob('*.json')):
        d = json.loads(p.read_text())
        if d['task_id'] != tid: continue
        rolls.append({'file': str(p), 'sha256': digest(p),
            **{k: d.get(k) for k in ['task_id','session_id','sample_id','global_step',
                'won','score','steering','research','policy_version','outcome','env_steps']},
            'trajectory': trajectory(d['trajectory'])})
    assert len(rolls) == 8
    out['cases'].append({'task_id': tid, 'step': step, 'archive': str(arc),
        'source_sha256': {n:digest(arc/n) for n in ['meta.json','session.jsonl','doc.md','episodes.jsonl']},
        'meta': {k:meta.get(k) for k in ['task_id','global_step','policy_version',
            'finalized','no_finalize','turns_used','experiments_used','episodes_consumed',
            'self_assessment','flags','guidance_log','research_wall_s','gt_injected']},
        'raw_skill_calls': calls, 'compiled_skill': (arc/'doc.md').read_text(),
        'research_episodes': episodes, 'student_rollouts': sorted(rolls,key=lambda x:x['session_id'])})
for rel in ['clean_skill.py','code/src/trace_opd/research/tools.py']:
    out['sources'][rel] = (ROOT/rel).read_text()
target=ROOT/'five_skill_cases_20261026.json'
target.write_text(json.dumps(out,ensure_ascii=False,indent=2))
print('SAVED_CASES_READY',target,'bytes',target.stat().st_size,'sha256',digest(target))
print('COUNTS',[(c['task_id'],c['step'],len(c['raw_skill_calls']),len(c['research_episodes']),len(c['student_rollouts'])) for c in out['cases']])
