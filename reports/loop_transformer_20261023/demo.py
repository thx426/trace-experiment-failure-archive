"""Exploratory mechanism demo, NOT a reproduction of a pretrained language model.

Random, unseen permutation tables define f. Predict f^k(start).
One shared Transformer-style cross-attention/FFN block reads the same table
at every hop. Compare discrete token feedback with raw hidden-state feedback,
then adapt the latter with endpoint-only or intermediate supervision.
The discrete arm is a symbolic CoT surrogate: history is logged but need not
be attended to because this particular task has a sufficient Markov state.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class PointerBlock(nn.Module):
    def __init__(self, n=16, d=48, heads=4, tied_readout=False):
        super().__init__()
        self.n, self.d, self.heads = n, d, heads
        self.tied_readout = tied_readout
        self.embedding = nn.Embedding(n, d)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2*d), nn.GELU(), nn.Linear(2*d, d))
        self.readout = nn.Linear(d, n)
        if tied_readout:
            self.readout.weight = self.embedding.weight

    def memory(self, table):
        b = table.shape[0]
        keys = self.k(self.embedding.weight).view(self.n, self.heads, -1)
        vals = self.v(self.embedding(table)).view(b, self.n, self.heads, -1)
        return keys, vals

    def step(self, h, memory):
        keys, vals = memory
        q = self.q(h).view(-1, self.heads, self.d//self.heads)
        scores = torch.einsum('bhd,nhd->bhn', q, keys)/math.sqrt(self.d//self.heads)
        attention = scores.softmax(-1)
        read = torch.einsum('bhn,bnhd->bhd', attention, vals).flatten(1)
        h = self.norm1(h + self.o(read))
        h = self.norm2(h + self.ff(h))
        return h, self.readout(h)

    def rollout(self, table, start, hops, feedback='latent', intervene=None):
        h, memory = self.embedding(start), self.memory(table)
        states, logits = [], []
        for i in range(hops):
            h, score = self.step(h, memory)
            states.append(h)
            logits.append(score)
            if feedback == 'discrete':
                h = self.embedding(score.argmax(-1))
            elif feedback == 'soft':
                h = score.softmax(-1) @ self.embedding.weight
            if i == 0 and intervene is not None:
                h = self.embedding(intervene)
        return torch.stack(logits, 1), torch.stack(states, 1)


def sample(batch, n, hops, generator):
    table = torch.rand(batch, n, generator=generator).argsort(-1)
    start = torch.randint(n, (batch,), generator=generator)
    state = start
    targets = []
    for _ in range(hops):
        state = table.gather(1, state[:, None]).squeeze(1)
        targets.append(state)
    return table, start, torch.stack(targets, 1)


def fit(model, steps, seed, objective, batch=128, depth=4):
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3 if objective == 'one_step' else 7e-4,
                                  weight_decay=0.01)
    losses, begin = [], time.perf_counter()
    model.train()
    for step in range(steps):
        table, start, target = sample(batch, model.n, 1 if objective == 'one_step' else depth, generator)
        scores, _ = model.rollout(table, start, target.shape[1])
        if objective == 'endpoint':
            loss = F.cross_entropy(scores[:, -1], target[:, -1])
        else:
            loss = F.cross_entropy(scores.flatten(0, 1), target.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 200 == 0 or step == steps-1:
            item = {'step': step+1, 'loss': float(loss.detach())}
            losses.append(item)
            print(f'{objective} seed={seed} step={step+1}/{steps} loss={item["loss"]:.4f}', flush=True)
    model.eval()
    return {'losses': losses, 'seconds': time.perf_counter()-begin, 'optimizer_steps': steps,
            'batch_size': batch, 'unroll_depth': 1 if objective == 'one_step' else depth,
            'training_seed': seed}


@torch.inference_mode()
def evaluate(model, feedback, table, start, target):
    parts = []
    states_all = []
    for t, s in zip(table.split(256), start.split(256)):
        scores, states = model.rollout(t, s, target.shape[1], feedback)
        parts.append(scores)
        states_all.append(states)
    scores, states = torch.cat(parts), torch.cat(states_all)
    pred = scores.argmax(-1)
    correct = pred == target
    confidence = scores.softmax(-1).gather(-1, target[..., None]).squeeze(-1)
    alignment = F.cosine_similarity(states, model.embedding(target), dim=-1)
    metrics = {'accuracy_by_hop': correct.float().mean(0).tolist(),
               'entire_prefix_accuracy': correct.long().cumprod(1).float().mean(0).tolist(),
               'gold_probability_by_hop': confidence.mean(0).tolist(),
               'cosine_to_gold_embedding': alignment.mean(0).tolist()}
    return metrics, pred, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path(__file__).parent / 'results')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--base-steps', type=int, default=1000)
    parser.add_argument('--adapt-steps', type=int, default=1000)
    parser.add_argument('--eval-size', type=int, default=2048)
    parser.add_argument('--max-hops', type=int, default=16)
    parser.add_argument('--tied-readout', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    records = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        base = PointerBlock(tied_readout=args.tied_readout)
        train = {'base': fit(base, args.base_steps, 1000+seed, 'one_step')}
        models = {'base': base}
        # Both adaptations use identical initial weights, batches, unroll depth,
        # optimizer, and number of updates. Only supervision placement changes.
        for objective in ['endpoint', 'intermediate']:
            model = copy.deepcopy(base)
            train[objective] = fit(model, args.adapt_steps, 2000+seed, objective)
            models[objective] = model
        table, start, target = sample(args.eval_size, base.n, args.max_hops,
                                      torch.Generator().manual_seed(90000+seed))
        assert torch.equal(target[:, 0], table.gather(1, start[:, None]).squeeze(1))
        arms = [('discrete', base, 'discrete'), ('naive_latent', base, 'latent'),
                ('soft_reembed', base, 'soft'), ('trained_endpoint', models['endpoint'], 'latent'),
                ('trained_intermediate', models['intermediate'], 'latent')]
        metrics, predictions, logits_by_arm = {}, {}, {}
        for name, model, feedback in arms:
            metrics[name], predictions[name], logits_by_arm[name] = evaluate(model, feedback, table, start, target)
        # The first step is exactly the same computation in these three arms.
        assert torch.equal(predictions['discrete'][:, 0], predictions['naive_latent'][:, 0])
        assert torch.equal(logits_by_arm['discrete'][:, 0], logits_by_arm['soft_reembed'][:, 0])
        intervention = {}
        with torch.inference_mode():
            replacement = {'predicted_embedding': predictions['naive_latent'][:, 0],
                           'gold_embedding': target[:, 0],
                           'wrong_embedding': (target[:, 0]+1) % base.n}
            for name, values in replacement.items():
                scores, _ = base.rollout(table, start, 2, intervene=values)
                intervention[name] = (scores[:, 1].argmax(-1) == target[:, 1]).float().mean().item()
            intervention['no_intervention'] = metrics['naive_latent']['accuracy_by_hop'][1]
        # Select by an explicit predicate, not by hand: first example where
        # first-hop prediction is correct but the second latent hop fails.
        mask = ((predictions['naive_latent'][:, 0] == target[:, 0]) &
                (predictions['naive_latent'][:, 1] != target[:, 1]) &
                (predictions['discrete'][:, :8] == target[:, :8]).all(1))
        candidates = mask.nonzero().flatten()
        idx = int(candidates[0]) if len(candidates) else 0
        case = {'selection': 'first first-hop-correct/second-hop-wrong naive-latent case with correct discrete prefix, else index 0',
                'index': idx, 'table': table[idx].tolist(), 'start': int(start[idx]),
                'gold': target[idx].tolist(), 'predictions': {k: v[idx].tolist() for k,v in predictions.items()},
                'probabilities': {k: v[idx].softmax(-1).tolist() for k,v in logits_by_arm.items()}}
        record = {'seed': seed, 'parameters': sum(p.numel() for p in base.parameters()),
                  'training': train, 'metrics': metrics, 'intervention_hop2': intervention, 'case': case}
        records.append(record)
        for name, model in models.items():
            torch.save(model.state_dict(), args.out / f'seed{seed}_{name}.pt')
        (args.out / f'seed{seed}.json').write_text(json.dumps(record, indent=2))
        print('EVAL', seed, {k: [round(v['accuracy_by_hop'][i],4) for i in [0,1,3,7,15] if i < args.max_hops] for k,v in metrics.items()}, flush=True)
    summary = {'experiment': 'symbolic pointer chasing with a learned Transformer-style cross-attention block',
               'limitations': ['Not a natural-language or pretrained-LLM benchmark.',
                               'Naive latent feedback deliberately changes the single-step-trained interface.',
                               'Explicit arm only feeds back the latest token; full CoT history is unnecessary on this Markov task.',
                               'Adaptation adds training compute; endpoint and intermediate arms are matched to each other, not to base.',
                               'Loops 5-16 extrapolate beyond adaptation depth 4.',
                               'Intermediate predictions are probe readouts, not generated tokens in latent arms.',
                               'All loops receive the unchanged relation table; this does not test memory capacity.'],
               'config': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
               'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                               'numpy': np.__version__, 'platform': platform.platform(), 'device': 'cpu', 'threads': 2},
               'runs': records}
    (args.out/'results.json').write_text(json.dumps(summary, indent=2))
    print('SAVED', args.out/'results.json', flush=True)


if __name__ == '__main__':
    main()
