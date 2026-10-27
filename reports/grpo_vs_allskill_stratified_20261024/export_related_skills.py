"""Read existing training artifacts for attribution; does not run models."""
from pathlib import Path
import json,re,collections,base64,gzip,hashlib
R=Path('/root/trace-webshop-arm-ratio/experiments/pass5_compact3072_20261024_r3')
d=json.loads((R/'stratified_failures_20261024.json').read_text())
run=Path(d['run_paths']['all_skill'])
refs={}
for t in d['selected_ids']:
    products=set()
    for r in d['candidate_episodes'][t]['grpo']:
        if not r['won']:continue
        matches=[m.group(1) for s in r['trajectory']['steps'] if s['valid'] and 'buy now' in s['observation'].lower() and (m:=re.fullmatch(r'click\[([a-z0-9]{10})\]',s['action_norm']))]
        if matches:products.add(matches[-1])
    refs[t]=sorted(products)
allrefs=set(x for ids in refs.values() for x in ids)
groups=collections.defaultdict(list)
for step in range(1,21):
    for p in sorted((run/'rollouts'/f'step_{step:05d}').glob('*.json')):
        row=json.loads(p.read_text());row['source_path']=str(p);groups[(step,row['task_id'])].append(row)
out={'selected_reference_products':refs,'related_training_sessions':[]}
for (step,tid),rows in sorted(groups.items()):
    visited={m.group(1) for r in rows for s in r['trajectory']['steps'] if s['valid'] and 'buy now' in s['observation'].lower() and (m:=re.fullmatch(r'click\[([a-z0-9]{10})\]',s['action_norm']))}
    if not visited & allrefs:continue
    rs=rows[0]['research'];archive=run/rs['research_archive'] if rs.get('research_archive') else None
    doc=archive/'doc.md' if archive else None
    entry={'training_step':step,'task_id':tid,'related_products':sorted(visited&allrefs),'research':rs,
           'initial_obs':rows[0]['trajectory']['initial_obs'],
           'doc':doc.read_text() if doc and doc.exists() else None,
           'doc_path':str(doc) if doc else None,
           'doc_sha256':hashlib.sha256(doc.read_bytes()).hexdigest() if doc and doc.exists() else None,
           'rollouts':[]}
    for r in rows:
        entry['rollouts'].append({k:r[k] for k in ['session_id','score','won','source_path']})
        for key in ['steering','quality']:
            entry['rollouts'][-1][key]={k:v for k,v in r[key].items() if isinstance(v,(int,float,bool)) or (isinstance(v,str) and len(v)<150)}
        entry['rollouts'][-1]['actions']=[{'t':s['t'],'action':s['action_norm'],'valid':s['valid']} for s in r['trajectory']['steps']]
    out['related_training_sessions'].append(entry)
text=json.dumps(out,ensure_ascii=False)
(R/'related_skills_20261024.json').write_text(text)
encoded=base64.b64encode(gzip.compress(text.encode())).decode()
(R/'related_skills_20261024.txt').write_text(encoded)
print('RELATED_SKILLS_EXPORT_COMPLETE','sessions',len(out['related_training_sessions']),'docs',sum(bool(x['doc']) for x in out['related_training_sessions']),'chars',len(encoded))
