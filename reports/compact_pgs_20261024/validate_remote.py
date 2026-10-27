"""CPU validation only; no generation, rollout, optimizer step or GPU workload."""
import dataclasses,json,os,sys,subprocess
from pathlib import Path
from types import SimpleNamespace as NS
R=Path(__file__).resolve().parent
OLD=R.parent/'pass5_20261023'
os.environ.update(TRACE_WEBSHOP_URLS='http://127.0.0.1:8310',TRACE_LOCAL_RESEARCH_KEY='loopback-only')
sys.path.insert(0,str(R/'code/src'))
from trace_opd.common.config import load_config
from trace_opd.research.compact import CompactContext
subprocess.run([sys.executable,str(R/'test_compact_contract.py')],check=True)
results={}
for arm in ['all_skill','selective_skill']:
    old=load_config(OLD/f'configs/{arm}.yaml');new=load_config(R/f'configs/{arm}.yaml')
    a,b=dataclasses.asdict(old.train),dataclasses.asdict(new.train)
    a.pop('experiment_name');b.pop('experiment_name');assert a==b,'student training changed'
    for key in ['episode','eval','steering','narration','webshop']:
        assert dataclasses.asdict(getattr(old,key))==dataclasses.asdict(getattr(new,key)),key
    assert new.train.resume_mode=='disable' and new.train.total_training_steps==100
    assert new.roles['researcher'].max_concurrency==16
    assert new.research.request_token_limit==3072 and new.research.raw_page_tokens==512
    assert new.research.n_exp==old.research.n_exp==3
    # Real model tokenizer validates pagination, Unicode, and pathological long text.
    ctx=CompactContext(NS(cfg=new.research,adapter=NS(task_brief=lambda:'Check a short task.')),16)
    for sample in ['中文'*4000,'é 🙂 '*4000,'x'*10000]:
        excerpt=ctx.clip(sample,512)
        assert len(ctx.ids(excerpt))<=512,(arm,len(ctx.ids(excerpt)))
    results[arm]={'same_student_and_reward':True,'same_sampling_and_eval':True,'fresh_start':True,
                  'strict_input_limit':3072,'concurrency':16,'real_tokenizer_clip':'passed'}
(R/'cpu_validation.json').write_text(json.dumps({'passed':True,'inference_calls':0,'rollouts':0,'arms':results},indent=2))
print('CPU_VALIDATION_PASSED',results,flush=True)
