from pathlib import Path
import json,hashlib,gzip,base64
ROOT=Path(__file__).resolve().parents[2]
R=Path(__file__).resolve().parent
old=ROOT/'reports/compact_pgs_20261024'
s=(old/'run_sequence.py').read_text().replace('24r2','24r3')
s=s.replace("TRACE_WEBSHOP_URLS='http://127.0.0.1:8310'", "TRACE_WEBSHOP_URLS=','.join(f'http://127.0.0.1:{p}' for p in range(8321,8325))")
s=s.replace('service=None\ntraining=None', '''webshop_services=[]

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
training=None''')
s=s.replace("status('starting_researcher')", "status('starting_webshop')\n    start_webshop()\n    status('starting_researcher')")
s=s.replace("require(service.poll() is None,'researcher service exited')", "require(service.poll() is None,'researcher service exited')\n            require(all(p.poll() is None for p in webshop_services),'WebShop shard exited')")
s += '''
    for p in webshop_services:
        if p.poll() is None:
            p.terminate()
    for p in webshop_services:
        try:p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGKILL)
            p.wait(timeout=10)
'''
(R/'run_sequence.py').write_text(s)
compile(s,'run_sequence.py','exec')
files={'run_sequence.py':s,
       'patch/scripts/cloud/webshop_service.py':(ROOT/'scripts/cloud/webshop_service.py').read_text(),
       'patch/src/trace_opd/envs/webshop/adapter.py':(ROOT/'src/trace_opd/envs/webshop/adapter.py').read_text(),
       'tests/test_cloud_webshop_service.py':(ROOT/'tests/test_cloud_webshop_service.py').read_text(),
       'tests/test_webshop_idempotent_transport.py':(ROOT/'tests/test_webshop_idempotent_transport.py').read_text()}
files['prepare_repair.py'] = '''from pathlib import Path
import shutil,json,hashlib,subprocess,sys,os
R=Path(__file__).resolve().parent
OLD=R.parent/'pass5_compact3072_20261024_r2'
assert not (R/'code').exists()
(R/'code').mkdir()
for name in ['src','scripts','configs','train']:
    shutil.copytree(OLD/'code'/name,R/'code'/name,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
os.symlink((OLD/'code/verl').resolve(),R/'code/verl')
shutil.copytree(OLD/'configs',R/'configs')
for name in ['audit_run.py','validate_remote.py','test_compact_contract.py','experiment_contract.json']:
    shutil.copy2(OLD/name,R/name)
for p in (R/'patch').rglob('*.py'):
    dst=R/'code'/p.relative_to(R/'patch');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
contract=json.loads((R/'experiment_contract.json').read_text())
contract['transport_repair']={'protocol':3,'shards':4,'ports':[8321,8322,8323,8324],
  'idempotent_reset_and_step':True,'http_backlog':1024,'blas_threads_per_shard':1,
  'preserved':['student checkpoint','reward function','data and goals','training schedule','sampling','research input cap 3072','research concurrency 16'],
  'approval':'2026-10-24 user explicitly approved diagnosis, repair and direct formal restart; no pilot'}
(R/'experiment_contract.json').write_text(json.dumps(contract,indent=2))
env=os.environ.copy();env['PYTHONPATH']=str(R/'code/src')
for pattern in ['test_cloud_webshop_service.py','test_webshop_idempotent_transport.py']:
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(R/'tests'),'-p',pattern,'-v'],env=env,check=True)
subprocess.run([sys.executable,str(R/'validate_remote.py')],env=env,check=True)
subprocess.run([sys.executable,'-m','compileall','-q',str(R/'code/src'),str(R/'code/scripts'),str(R/'run_sequence.py')],check=True)
files=[p for directory in [R/'code/src',R/'code/scripts',R/'code/train',R/'configs'] for p in directory.rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.json')]
files += [R/'run_sequence.py',R/'audit_run.py',R/'experiment_contract.json']
BASE=R.parent/'pass5_20261023'
files += [BASE/'pass5_gate.json',BASE/'data/train.parquet',BASE/'data/dev.parquet']
(R/'frozen_manifest.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
print('REPAIR_PREPARED_CPU_VALIDATED',R,flush=True)
'''
# Test module paths resolve under R/code, not R/scripts.
files['tests/test_cloud_webshop_service.py']=files['tests/test_cloud_webshop_service.py'].replace(' / "scripts/cloud/webshop_service.py"',' / "code/scripts/cloud/webshop_service.py"')
(R/'deployment.json').write_text(json.dumps(files))
(R/'deployment.b64').write_text(base64.b64encode(gzip.compress(json.dumps(files).encode())).decode())
print('Payload bytes', (R/'deployment.b64').stat().st_size)
