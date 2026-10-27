"""Frozen-checkpoint, mixed 4-bare/4-SPS diagnostic; never trains a model.

Run against an isolated copy of trace_opd and the audited SGLang converters.
The parent launcher waits for the existing sequence to finish before allocating
a GPU. No connection to the live Ray cluster or researcher is made.
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import random
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def summarize(rows, groups=4, size=8):
    """Never turn an incomplete group or environment failure into a zero."""
    result = {}
    for arm in 'ABCD':
        selected = [r for r in rows if r['arm'] == arm]
        buckets = {}
        for r in selected:
            key = (r['task_id'], r['group'])
            buckets.setdefault(key, []).append(r)
        complete = [rs for rs in buckets.values() if len(rs) == size
                    and len({r['sample'] for r in rs}) == size
                    and all(r['outcome'] != 'ENV_ERROR' for r in rs)]
        attempts = [r for r in selected if r['outcome'] != 'ENV_ERROR']
        nz = sum(0 < sum(r['reward'] for r in rs) < size for rs in complete)
        result[arm] = {
            'wins': sum(r['reward'] for r in attempts), 'attempts': len(attempts),
            'success_rate': sum(r['reward'] for r in attempts) / len(attempts) if attempts else None,
            'nonzero_advantage_groups': nz, 'complete_groups': len(complete),
            'expected_groups': 2 * groups,
            'nonzero_advantage_group_ratio': nz / len(complete) if complete else None,
            'env_errors': len(selected) - len(attempts),
            'complete': len(complete) == 2 * groups,
            'by_task': {},
        }
        for task in sorted({r['task_id'] for r in selected}):
            rr = [r for r in attempts if r['task_id'] == task]
            gg = [rs for rs in complete if rs[0]['task_id'] == task]
            result[arm]['by_task'][task] = {
                'wins': sum(r['reward'] for r in rr), 'attempts': len(rr),
                'success_rate': sum(r['reward'] for r in rr)/len(rr) if rr else None,
                'nonzero_advantage_groups': sum(0 < sum(r['reward'] for r in rs) < size for rs in gg),
                'complete_groups': len(gg),
                'nonzero_advantage_group_ratio': sum(0 < sum(r['reward'] for r in rs) < size for rs in gg)/len(gg) if gg else None,
            }
        for guided in (False, True):
            rr = [r for r in attempts if r['guided'] == guided]
            result[arm]['guided' if guided else 'bare'] = {
                'wins': sum(r['reward'] for r in rr), 'attempts': len(rr),
                'success_rate': sum(r['reward'] for r in rr)/len(rr) if rr else None,
                'steered_tokens': sum(r['steer_count'] for r in rr),
            }
    return result


async def experiment(root, preflight=False):
    import numpy as np
    from transformers import AutoTokenizer
    from trace_opd.common.config import load_config
    from trace_opd.core import protocol, template_delta
    from trace_opd.core.episode import GenResult, run_episode
    from trace_opd.core.template_delta import TemplateDelta, render_ids, turn_stop_token_ids
    from trace_opd.core.episode import inject_guidance
    from trace_opd.envs.registry import make_env_factory
    from trace_opd.steering import IntensityGate, TrajectoryBudget, TurnRejectCap
    from trace_opd.steering.hook import SpsGenerate, TrajectorySteerState, truncated_entropy
    from sglang_converters import (_extract_prompt_logprobs_sglang,
                                  _extract_response_topk_sglang, _normalize_finish_reason)
    spec = json.loads((root/'design.json').read_text())
    cfg = load_config(root/'config.yaml')
    tasks = json.loads((root/'tasks.json').read_text())
    model = Path(spec['checkpoint'])
    tokenizer = AutoTokenizer.from_pretrained(model)
    protocol.configure_episode_caps(cfg.episode.max_env_steps, cfg.episode.max_assistant_turns,
                                    cfg.episode.max_tokens_per_turn)
    protocol.configure_reasoning_tag(cfg.episode.reasoning_tag)
    template_delta.configure_thinking(cfg.episode.enable_thinking)
    delta = TemplateDelta(tokenizer)
    protocol.configure_fc_dialect(delta.fc_dialect)
    factory = make_env_factory(cfg)
    stop_ids = turn_stop_token_ids(tokenizer)
    st = cfg.steering
    budget = cfg.episode.session_token_budget or cfg.train.response_length
    docs = {a: {t['task_id']: (root/'skills'/a/(t['task_id']+'.txt')).read_text()
                for t in tasks} for a in 'BCD'}
    def prefix(adapter, doc):
        return render_ids(tokenizer, [
            {'role': 'system', 'content': inject_guidance(adapter.executor_scaffold(), doc)},
            {'role': 'user', 'content': protocol.truncate_obs(adapter.initial_obs)}],
            True, tools=adapter.tools())
    prompt_lengths = {}
    for task in tasks:
        adapter = factory.adapter_from_item(task)
        bare = prefix(adapter, '')
        for a in 'BCD':
            p = prefix(adapter, docs[a][task['task_id']])
            if p[-len(delta.gen_prefix)-2:] != bare[-len(delta.gen_prefix)-2:]:
                raise RuntimeError('teacher/student prefix mismatch')
            if len(p) + budget + 8 > cfg.train.rollout_max_model_len:
                raise RuntimeError('guidance overflows context; no silent bare fallback')
            prompt_lengths[f'{a}/{task["task_id"]}'] = len(p)
    (root/'preflight.json').write_text(json.dumps({
        'passed': True, 'prompt_lengths': prompt_lengths,
        'response_budget': budget, 'model_context': cfg.train.rollout_max_model_len,
        'tasks': [t['task_id'] for t in tasks], 'gpu_allocated': False,
        'runtime_sampling_not_yet_tested': True,
    }, indent=2))
    if preflight:
        print('PREFLIGHT_OK', prompt_lengths, flush=True)
        return
    for p, expected in json.loads((root/'input_hashes.json').read_text()).items():
        if sha(root/p) != expected:
            raise RuntimeError(f'input changed: {p}')
    manifest = json.loads((model/'MODEL_MANIFEST.json').read_text())
    model_hashes = {}
    for entry in manifest['files']:
        p = model/entry['path']
        if sha(p) != entry['sha256']:
            raise RuntimeError(f'checkpoint hash mismatch: {p.name}')
        model_hashes[p.name] = entry['sha256']
    (root/'checkpoint_verified.json').write_text(json.dumps(model_hashes, indent=2))
    # Import/allocate only after the launcher's no-competing-process gate.
    from sglang import Engine
    engine = Engine(model_path=str(model), tp_size=1, dtype='bfloat16',
                    attention_backend='flashinfer', mem_fraction_static=0.65,
                    disable_cuda_graph=True, skip_tokenizer_init=True,
                    context_length=cfg.train.rollout_max_model_len,
                    max_running_requests=1, random_seed=spec['seed'],
                    trust_remote_code=False)
    rows = []
    entropy = []
    def status(phase, **kw):
        (root/'status.json').write_text(json.dumps({'phase': phase, 'time': time.time(),
            'pid': os.getpid(), 'completed_episodes': len(rows), **kw}, indent=2))
    class Generate:
        def __init__(self, seed, collect=False):
            self.rng = random.Random(seed)
            self.collect = collect
        async def __call__(self, ids, params):
            p = dict(params)
            p['max_new_tokens'] = p.pop('max_tokens')
            p.setdefault('stop_token_ids', stop_ids)
            rk = p.pop('logprobs', 0)
            pk = p.pop('prompt_logprobs', None)
            tail = p.pop('prompt_logprobs_tail', None)
            p.pop('flat_logprobs', None); p.pop('detokenize', None)
            p['sampling_seed'] = self.rng.randrange(1, 2**31-1)
            if p.get('top_k') == 0: p['top_k'] = -1
            k = pk if pk is not None else max(int(rk), 16 if self.collect else 0)
            start = max(0, len(ids)-int(tail)) if pk is not None and tail else (0 if pk is not None else -1)
            output = await engine.async_generate(input_ids=ids, sampling_params=p,
                return_logprob=True, top_logprobs_num=k, logprob_start_len=start)
            meta = output['meta_info']; tokens = output['output_ids']
            lp = meta['output_token_logprobs']
            if len(lp) != len(tokens) or any(int(x[1]) != t for x,t in zip(lp,tokens)):
                raise RuntimeError('sample/logprob alignment failed')
            ft = (meta.get('finish_reason') or {}).get('type')
            extra = {'log_probs': [float(x[0]) for x in lp],
                     'finish_type': ft, 'stop_reason': _normalize_finish_reason(ft)}
            if pk is not None:
                _extract_prompt_logprobs_sglang(meta, pk, len(ids), extra, start)
            elif k:
                _extract_response_topk_sglang(meta, k, len(tokens), 'abcd', extra)
                if self.collect:
                    entropy.extend(truncated_entropy(x) for x in extra['response_topk_logprobs'])
            return GenResult(token_ids=tokens, text=tokenizer.decode(tokens, skip_special_tokens=True), extra=extra)
    async def one(task, arm, group, sample, theta=None, calibration=False):
        seed = spec['seed'] + (int(task['task_id'][-5:])*10000) + group*100 + sample
        adapter = factory.adapter_from_item(task)
        base = Generate(seed, collect=calibration)
        guided = not calibration and arm != 'A' and sample < 4
        state = None; gen = base
        if guided:
            state = TrajectorySteerState(
                gate=IntensityGate(mode='entropy', eta_max=st.eta_max, epsilon=st.epsilon,
                                   entropy_theta=theta, w_min=st.intensity_w_min),
                budget=TrajectoryBudget(st.b_max, st.c), reject_cap=TurnRejectCap(st.r_max),
                rng=np.random.default_rng(seed+700000001))
            gen = SpsGenerate(draft=base, verify=Generate(seed+900000001),
                teacher_prefix_ids=prefix(adapter, docs[arm][task['task_id']]),
                student_prompt_ids=prefix(adapter, ''), state=state, tokenizer=tokenizer,
                topk=st.topk, c=st.c, obs_topk=st.obs_topk,
                stop_token_ids=frozenset(stop_ids), draft_horizon=0)
        res = await asyncio.wait_for(run_episode(adapter=adapter, generate=gen,
            tokenizer=tokenizer, delta=delta, guidance='',
            episode_id=f'abcd-{arm}-{group}-{sample}', sample_idx=sample,
            response_budget=budget, sampling_params={'temperature': 1.0, 'top_p': 1.0, 'top_k': -1}),
            timeout=1800)
        tr = res.trajectory
        record = dict(arm=arm, task_id=task['task_id'], group=group, sample=sample,
            seed=seed, guided=guided, outcome=tr.outcome, won=bool(tr.won), reward=int(tr.won),
            score=float(tr.score), error=tr.error, entropy_theta=theta,
            steer_count=sum(r.eta > 0 for r in state.ledger.records) if state else 0,
            steer_reject_count=sum(r.rejected for r in state.ledger.records) if state else 0,
            initial_observation=tr.initial_obs,
            # Deliberately export observations/actions, never private reasoning.
            steps=[{k:s.get(k) for k in ('t','action_norm','observation','valid','acted')}
                   for s in tr.to_dict()['steps']])
        dest = 'calibration.jsonl' if calibration else 'episodes.jsonl'
        with (root/dest).open('a') as f: f.write(json.dumps(record, ensure_ascii=False)+'\n')
        if tr.outcome == 'ENV_ERROR': raise RuntimeError('ENV_ERROR: run incomplete, not a model failure')
        return record
    try:
        status('calibrating')
        # Bare-only independent calibration, common frozen threshold for B/C/D.
        for task in tasks:
            for s in range(4): await one(task, 'A', 99, s, calibration=True)
        if not entropy: raise RuntimeError('empty entropy calibration')
        theta = max(float(np.quantile(entropy, 0.95)), 1e-6)
        (root/'calibration_summary.json').write_text(json.dumps({
            'episodes': 8, 'tokens': len(entropy), 'quantile': .95,
            'frozen_entropy_theta': theta, 'excluded_from_primary_metrics': True}, indent=2))
        for g in range(spec['groups_per_task']):
            order = list('ABCD'); random.Random(spec['seed']+g).shuffle(order)
            for task in tasks:
                for s in range(8):
                    for a in order:
                        rows.append(await one(task, a, g, s, theta))
                        (root/'summary.json').write_text(json.dumps(summarize(rows), indent=2))
                        status('sampling', arm=a, task_id=task['task_id'], group=g, sample=s)
        if not all(v['complete'] for v in summarize(rows).values()):
            raise RuntimeError('incomplete primary groups')
        group_audit = []
        for task in tasks:
            for a in 'ABCD':
                for g in range(spec['groups_per_task']):
                    rr = sorted([r for r in rows if r['task_id']==task['task_id']
                                 and r['arm']==a and r['group']==g], key=lambda r:r['sample'])
                    rewards = [r['reward'] for r in rr]; mean = sum(rewards)/8
                    group_audit.append(dict(task_id=task['task_id'],arm=a,group=g,
                        rewards=rewards, advantages=[r-mean for r in rewards],
                        nonzero_advantage=0<sum(rewards)<8))
        (root/'group_advantages.json').write_text(json.dumps(group_audit,indent=2))
        for name,h in model_hashes.items():
            if sha(model/name) != h: raise RuntimeError('checkpoint changed during evaluation')
        status('completed', checkpoint_unchanged=True)
    finally:
        engine.shutdown()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--preflight', action='store_true'); args = p.parse_args()
    asyncio.run(experiment(args.root, args.preflight))
