"""Read-only status and researcher input audit for the formal sequence."""
from pathlib import Path
import json, subprocess, collections

R=Path(__file__).resolve().parent
state=json.loads((R/'sequence_status.json').read_text())
print('FORMAL SEQUENCE',json.dumps(state,ensure_ascii=False))
print('ABORT', (R/'ABORT').read_text() if (R/'ABORT').exists() else 'absent')
run=Path(state['run']) if state.get('run') else None
if run:
    print('ROLLOUT FILES', {p.name:len(list(p.glob('*.json'))) for p in (run/'rollouts').glob('*')})
    counts=[];errors=0
    for p in (run/'tape').glob('researcher.*.jsonl'):
        for line in p.read_text(errors='replace').splitlines():
            try:d=json.loads(line)
            except json.JSONDecodeError:continue
            if d.get('role')!='researcher':continue
            if d.get('error'):errors+=1
            if d.get('usage'):counts.append(d['usage']['prompt_tokens'])
    print('RESEARCH INPUTS',json.dumps({'completed_calls':len(counts),'mean':sum(counts)/len(counts) if counts else None,
        'max':max(counts) if counts else None,'violations':sum(n>3072 or n<=0 for n in counts),'call_errors':errors}))
    log=Path(state['owner_state'])/'command.log'
    if log.exists():
        lines=log.read_text(errors='replace').splitlines()
        print('TRAIN LOG TAIL',*lines[-8:],sep='\n')
else:
    log=Path('/tmp/p5compact3072-research24r2/command.log')
    if log.exists():print('RESEARCHER LOG TAIL',*log.read_text(errors='replace').splitlines()[-8:],sep='\n')
subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv'])
