"""Independent, bounded resource waiter; does not stop or edit other jobs."""
import argparse
import fcntl
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time


def main(root):
    root = root.resolve()
    lock = (root/'launcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    spec = json.loads((root/'design.json').read_text())
    sequence = Path(spec['original_sequence_status'])
    env = dict(os.environ, PYTHONPATH=str(root/'code/src'),
               TRACE_WEBSHOP_URLS='http://127.0.0.1:8310',
               TRACE_LOCAL_RESEARCH_KEY='loopback-only',
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    trace = Path('/root/trace-webshop-arm-ratio')
    env['PATH'] = str(trace/'.venvs/train/bin')+os.pathsep+env.get('PATH','')
    env['HF_HOME'] = str(trace/'.cache/huggingface')
    env['XDG_CACHE_HOME'] = str(trace/'.cache/xdg')
    env['TORCH_EXTENSIONS_DIR'] = str(trace/'.cache/torch_extensions')
    def state(phase, **kw):
        obj = {'phase': phase, 'pid': os.getpid(), 'time': time.time(), **kw}
        tmp = root/'queue_status.tmp'; tmp.write_text(json.dumps(obj, indent=2))
        tmp.replace(root/'queue_status.json')
    if (root/'episodes.jsonl').exists():
        raise RuntimeError('refusing to overwrite or mix an existing sample run')
    cmd = [sys.executable, str(root/'run_skill_abcd_pilot.py'), '--root', str(root)]
    state('preflight')
    subprocess.run(cmd+['--preflight'], env=env, check=True, timeout=180)
    start = time.monotonic(); stable = 0
    while time.monotonic() - start < 48*3600:
        # Never allocate during any of the queued original arms, even if a GPU
        # briefly appears empty between phases. Require successful completion.
        seq = json.loads(sequence.read_text())
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                    '--format=csv,noheader,nounits'], text=True, timeout=20)
        gpu = next(x.split(',') for x in raw.strip().splitlines() if x.split(',')[0].strip() == '1')
        uuid = gpu[1].strip(); free = int(gpu[2].strip())
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                    '--format=csv,noheader'], text=True, timeout=20)
        vacant = uuid not in apps and free >= 35000
        allowed = seq.get('phase') == 'completed_analysis' and vacant
        stable = stable+1 if allowed else 0
        state('waiting_for_original_experiments', original_phase=seq.get('phase'),
              original_job=seq.get('job'), gpu=1, free_mib=free,
              gpu_vacant=vacant, consecutive_ready_checks=stable,
              deadline_seconds_remaining=round(48*3600-(time.monotonic()-start)))
        if stable >= 3: break
        time.sleep(60)
    else:
        state('resource_wait_expired'); return 3
    env['CUDA_VISIBLE_DEVICES'] = '1'
    state('launching_frozen_checkpoint_probe', gpu=1)
    with (root/'sampling.log').open('x') as log:
        child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        state('sampling', child_pid=child.pid, gpu=1)
        # A timeout affects only this owned child, never original workers.
        try: rc = child.wait(timeout=12*3600)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM); rc = 124
            try: child.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=30)
    state('completed' if rc == 0 else 'failed', returncode=rc)
    return rc


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    try: sys.exit(main(args.root))
    except Exception as e:
        (args.root/'queue_error.txt').write_text(repr(e))
        raise
