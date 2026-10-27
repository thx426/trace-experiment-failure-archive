"""Approved formal single run: Clean-Skill GRPO; no old All/Selective queue.
No pilot, smoke run, generation probe, training resume, or automatic retry.
"""
from pathlib import Path
import json, os, signal, subprocess, sys, time, traceback, hashlib, re
import httpx

R=Path(__file__).resolve().parent
OLD=R.parent/'pass5_20261023'
PY='/root/trace-webshop-arm-ratio/.venvs/train/bin/python'
SERVER='/root/trace-webshop-arm-ratio/.venvs/researcher/bin/vllm'
MODEL='/root/trace-webshop-arm-ratio/models/Qwen3.6-35B-A3B'
ABORT=R/'ABORT'
STATE=R/'sequence_status.json'
SERVICE_STATE='/tmp/p5compact3072-research25clean2'
OWNERS={'clean_skill':'/tmp/p5compact3072-clean25v2'}
os.environ.update(TRACE_WEBSHOP_URLS=','.join(f'http://127.0.0.1:{p}' for p in range(8321,8325)),TRACE_LOCAL_RESEARCH_KEY='loopback-only',
                  TRACE_COMPACT_ABORT_FILE=str(ABORT),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                  HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false')
os.environ['PYTHONPATH']=str(R/'code/src')+':'+str(R/'code/verl')
sys.path.insert(0,str(R/'code/src'))
from audit_run import audit,load_config,require

def status(phase,**extra):
    value={'phase':phase,'pid':os.getpid(),'time':time.time(),
           'input_limit':3072,'server_concurrency':16,'order':['clean_skill'],**extra}
    tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(STATE)
    print(phase,extra,flush=True)

def stop_requested(signum,_frame):
    ABORT.write_text(f'sequence received signal {signum}\n')

for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,stop_requested)

def verify_frozen():
    for path,digest in json.loads((R/'frozen_manifest.json').read_text()).items():
        require(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,'frozen file changed: '+path)
    require(not ABORT.exists(),'abort marker exists')

def owned(command,state,gpus,logname):
    require(not Path(state).exists(),'owner state already exists: '+state)
    args=[PY,str(R/'code/scripts/cloud/run_owned_job.py'),'--state-dir',state,'--gpus',gpus,
          '--no-deadline','--abort-file',str(ABORT),'--cwd',str(R/'code'),'--',*command]
    env=os.environ.copy()
    if command[0]==PY:
        # Match the established activate_experiment.sh runtime for spawned
        # SGLang schedulers; their memory-saver hook loads CUDA before torch.
        venv=Path(PY).parent.parent
        cuda=venv/'lib/python3.12/site-packages/nvidia/cuda_runtime/lib'
        require((cuda/'libcudart.so.12').is_file(),'bundled CUDA runtime missing')
        env['LD_LIBRARY_PATH']=str(cuda)+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        env['VIRTUAL_ENV']=str(venv)
        env['PATH']=str(venv/'bin')+':'+env['PATH']
        env['PYTHONNOUSERSITE']='1'
    with (R/logname).open('x') as out:
        return subprocess.Popen(args,stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)

offset=0
preemptions=[]
def inspect_service_log():
    global offset
    p=Path(SERVICE_STATE)/'command.log'
    if not p.exists():return
    with p.open(errors='replace') as f:f.seek(offset);chunk=f.read();offset=f.tell()
    if re.search(r'CUDA out of memory|OutOfMemoryError',chunk,re.I):
        ABORT.write_text('researcher OOM; no concurrency fallback\n');raise RuntimeError('researcher OOM')
    now=time.monotonic()
    preemptions.extend([now]*len(re.findall(r'(?:is preempted|preempted by)',chunk,re.I)))
    preemptions[:]=[t for t in preemptions if now-t<60]
    if len(preemptions)>=3:
        ABORT.write_text('sustained researcher preemption (>=3 events/60s)\n');raise RuntimeError('sustained researcher preemption')

webshop_services=[]

def start_webshop():
    # Read-only reference; old service and failed runs remain untouched.
    reference=httpx.get('http://127.0.0.1:8310/meta',timeout=30).json()['provenance']
    (R/'webshop_reference.json').write_text(json.dumps(reference,indent=2))
    stable=[k for k in reference if k not in ('service_revision','service_sha256')]
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
               JAVA_HOME='/root/trace-webshop-arm-ratio/environment/java11')
    env['PATH']=env['JAVA_HOME']+'/bin:'+env['PATH']
    for port in range(8321,8325):
        with (R/f'webshop-{port}.log').open('x') as log:
            p=subprocess.Popen(['/root/trace-webshop-arm-ratio/environment/webshop-v2/venv/bin/python',
                '-u',str(R/'code/scripts/cloud/webshop_service.py'),
                '--upstream','/root/trace-webshop-arm-ratio/environment/webshop-v2/WebShop',
                '--port',str(port),'--pool-size','1024','--option-fix','on',
                '--manifest',str(R/f'webshop-{port}.json')],
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,
                cwd=R/'code',start_new_session=True)
        webshop_services.append(p)
    metas=[]
    for port,p in zip(range(8321,8325),webshop_services):
        ready=False
        for _ in range(300):
            require(p.poll() is None,f'WebShop {port} exited during startup')
            require(not ABORT.exists(),'abort requested')
            try:
                response=httpx.get(f'http://127.0.0.1:{port}/meta',timeout=5)
                if response.status_code==200:
                    meta=response.json()
                    require(meta.get('transport_protocol')==3,'wrong WebShop transport')
                    require(all(meta['provenance'].get(k)==reference[k] for k in stable),
                            f'WebShop {port} reward/data/runtime provenance mismatch')
                    metas.append({'port':port,'pid':p.pid,'provenance':meta['provenance']})
                    ready=True;break
            except httpx.TransportError:pass
            time.sleep(2)
        require(ready,f'WebShop {port} readiness timeout')
    (R/'webshop_services.json').write_text(json.dumps({'shards':metas,'semantic_equivalence':True,
        'compared_fields':stable,'blas_threads':1,'transport_protocol':3},indent=2))
    status('webshop_ready',ports=list(range(8321,8325)),pids=[p.pid for p in webshop_services],
           semantic_equivalence=True)

service=None
training=None
try:
    verify_frozen()
    require(json.loads((R/'cpu_validation.json').read_text())['passed'],'CPU validation missing')
    require(json.loads((OLD/'code/runs/pass5_grpo_s42/completion_audit.json').read_text())['passed'],'GRPO baseline incomplete')
    status('starting_webshop')
    start_webshop()
    status('starting_researcher')
    service=owned([SERVER,'serve',MODEL,'--served-model-name','qwen3.6-35b-a3b','--host','127.0.0.1',
                   '--port','18881','--api-key','loopback-only','--tensor-parallel-size','2',
                   '--dtype','bfloat16','--max-model-len','7168','--max-num-seqs','16',
                   '--max-num-batched-tokens','4096','--gpu-memory-utilization','0.95',
                   '--enforce-eager','--language-model-only','--enable-auto-tool-choice',
                   '--tool-call-parser','qwen3_coder','--reasoning-parser','qwen3'],
                  SERVICE_STATE,'0,3','researcher-owner.log')
    ready=False
    for _ in range(240):
        require(service.poll() is None,'researcher service exited during startup')
        require(not ABORT.exists(),'abort requested')
        inspect_service_log()
        try:
            resp=httpx.get('http://127.0.0.1:18881/v1/models',headers={'Authorization':'REDACTED REDACTED'},timeout=5)
            if resp.status_code==200:
                require(any(x['id']=='qwen3.6-35b-a3b' for x in resp.json()['data']),'wrong model served')
                ready=True;break
        except httpx.TransportError:pass
        time.sleep(5)
    require(ready,'researcher readiness timeout')
    # Tokenization-only check of native tool/template accounting. No generation.
    from trace_opd.common.llm import make_client
    from trace_opd.research.compact import SYSTEM,schemas
    from types import SimpleNamespace
    import asyncio
    cfg=load_config(R/'configs/clean_skill.yaml')
    async def count_only():
        client=make_client('researcher',cfg.roles['researcher'])
        state=SimpleNamespace(experiments_used=0,revision_requested=False,episodes={},fetch_used=0)
        try:
            n=await client.count_prompt([{'role':'system','content':SYSTEM.format(doc_max_words=500)},
                                         {'role':'user','content':'Startup tokenization contract check.'}],schemas(cfg.research,state))
            require(0<n<=3072,'startup template exceeds input cap')
            return n
        finally:await client.aclose()
    n=asyncio.run(count_only())
    status('researcher_ready',researcher_owner_pid=service.pid,tokenization_only_count=n)
    for name in ['clean_skill']:
        verify_frozen()
        cfg=load_config(R/f'configs/{name}.yaml')
        run=R/'code'/cfg.run.runs_root/cfg.run.name
        require(not run.exists(),'fresh run already exists: '+str(run))
        training=owned([PY,'-u','scripts/launch_train.py','--config',str(R/f'configs/{name}.yaml')],
                       OWNERS[name],'1,2',f'{name}-owner.log')
        status('running',job=name,run=str(run),owner_pid=training.pid,owner_state=OWNERS[name],
               researcher_owner_pid=service.pid,next_job=None)
        while training.poll() is None:
            require(not ABORT.exists(),'abort requested; inspect ABORT')
            require(service.poll() is None,'researcher service exited')
            require(all(p.poll() is None for p in webshop_services),'WebShop shard exited')
            inspect_service_log()
            from baseline_guard import review_completed_evaluations
            review_completed_evaluations(R, run, cfg)
            time.sleep(5)
        require(training.returncode==0,f'{name} owner exit {training.returncode}')
        result=audit(R/f'configs/{name}.yaml',Path(OWNERS[name]))
        # Actual completed researcher inputs must all satisfy the contract.
        actual=[]
        for p in run.rglob('*.jsonl'):
            if 'tape' not in str(p):continue
            for line in p.open(errors='replace'):
                try:d=json.loads(line)
                except json.JSONDecodeError:continue
                if d.get('role')=='researcher' and d.get('usage'):
                    count=d['usage'].get('prompt_tokens',0);require(0<count<=3072,'researcher input audit failed');actual.append(count)
        require(actual,'researcher usage evidence missing')
        result['researcher_inputs']={'calls':len(actual),'mean':sum(actual)/len(actual),'max':max(actual),'limit':3072}
        (run/'compact_completion_audit.json').write_text(json.dumps(result,indent=2))
        status('audited',job=name,audit=result)
    status('completed',baseline=str(OLD/'code/runs/pass5_grpo_s42'))
except BaseException as exc:
    if not ABORT.exists():ABORT.write_text(f'{type(exc).__name__}: {exc}\n')
    if (R/'baseline_stop.json').exists():
        status('stopped_below_baseline', comparison=json.loads((R/'baseline_stop.json').read_text()))
    else:
        status('stopped_on_error',error=repr(exc));traceback.print_exc()
    raise
finally:
    for p in [training,service]:
        if p is not None and p.poll() is None:
            p.terminate()
            try:p.wait(timeout=60)
            except subprocess.TimeoutExpired:print('Owned cleanup pending; inspect state directory',flush=True)

    for p in webshop_services:
        if p.poll() is None:
            p.terminate()
    for p in webshop_services:
        try:p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGKILL)
            p.wait(timeout=10)
