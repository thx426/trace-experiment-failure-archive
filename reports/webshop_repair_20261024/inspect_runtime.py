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
run=Path(sequence.get('run',str(R/'code/runs/pass5_compact3072_all_skill_s42')))
out['rollout_rows']={p.name:len(list(p.glob('*.json'))) for p in (run/'rollouts').glob('*') if p.is_dir()}
logpath=Path(sequence.get('owner_state','/tmp/p5compact3072-all24r3'))/'command.log'
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
(R/'runtime_snapshot.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
