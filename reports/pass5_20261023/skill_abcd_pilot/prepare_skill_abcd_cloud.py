"""Materialize only a new pilot directory, never edit the source experiment."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(dest):
    import pandas as pd
    original = Path('/root/trace-webshop-arm-ratio/experiments/pass5_20261023')
    if (dest/'code').exists(): raise RuntimeError('already prepared; refusing overwrite')
    shutil.copytree(original/'code/src/trace_opd', dest/'code/src/trace_opd',
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copyfile(original/'configs/selective_skill.yaml', dest/'config.yaml')
    spec = json.loads((original/'sequence_spec_v6.json').read_text())
    frozen = spec['sha256']
    if not isinstance(frozen, dict): raise RuntimeError('unexpected frozen manifest schema')
    for path,h in frozen.items():
        p=Path(path); p=p if p.is_absolute() else original/p
        if sha(p) != h: raise RuntimeError(f'original frozen source drift: {path}')
    (dest/'original_frozen_hashes.json').write_text(json.dumps(frozen,indent=2))
    rows = pd.read_parquet(original/'data/train.parquet')
    tasks = [dict(x) for x in rows['extra_info'] if x['task_id'] in ('webshop_00510','webshop_00512')]
    tasks.sort(key=lambda x:x['task_id'])
    if len(tasks)!=2: raise RuntimeError('missing/duplicate target tasks')
    (dest/'tasks.json').write_text(json.dumps(tasks, ensure_ascii=False,
        default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))
    source = original/'code/verl/verl/workers/rollout/sglang_rollout/async_sglang_server.py'
    text=source.read_text(); tree=ast.parse(text)
    names={'_extract_prompt_logprobs_sglang','_extract_response_topk_sglang','_normalize_finish_reason'}
    funcs=[ast.get_source_segment(text,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if len(funcs)!=3: raise RuntimeError('missing audited SGLang converters')
    (dest/'sglang_converters.py').write_text('# Verbatim pure functions from '+str(source)+'\n\n'+'\n\n'.join(funcs)+'\n')
    (dest/'source_provenance.json').write_text(json.dumps({
        'source_experiment':str(original), 'sglang_adapter_source':str(source),
        'sglang_adapter_sha256':sha(source),'trace_package_copied_verbatim':True,
        'original_frozen_files_verified':len(frozen), 'training_mutated':False},indent=2))
    excluded={'input_hashes.json','preflight.json','queue_status.json','queue_status.tmp','queue_error.txt','launcher.lock'}
    hashes={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and p.name not in excluded
        and not p.name.endswith('.log')}
    (dest/'input_hashes.json').write_text(json.dumps(hashes,indent=2))
    print('PREPARED',dest,'TASKS',len(tasks),'FROZEN_ORIGINAL',len(frozen),'PILOT_INPUTS',len(hashes),flush=True)


if __name__=='__main__': main(Path(sys.argv[1]))
