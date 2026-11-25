#!/usr/bin/env python3
"""analysis_bound_signal.py -- how informative the bound-difference reward is
along the greedy trajectories of the SLAB policy, and how far a station
capacity relaxation would tighten the bound (evaluation only, CPU, no
training, no timing).

For every order of a set, the SLAB policy of seed 301
(10x25+ppvc-mixed+a1lagx05-2k-s301) is rolled out greedily on the CPU in the
default environment, exactly as analysis_bound_tightness.py does. At every
state s_0..s_T it records the lag-augmented and the lag-free chain bound,
recomputed directly from the state with analysis_bound_tightness.chain_bounds
(the recomputation equals the environment's own bound to 1e-6 h, checked at
every step). From these it computes, per order:

  zero_share_<b>      share of the T steps whose reward
                      r_t = Chat_b(s_t) - Chat_b(s_{t+1}) is zero (|r| < 1e-9 h),
                      i.e. steps that carry no reward signal under bound b;
  early_share_<b>     share of the total bound rise Cmax - Chat_b(s_0) that
                      arrives in the first half of the steps (t < T/2);
  s0_ratio_<b>        Chat_b(s_0) / Cmax.

At s_0 it also computes a station-capacity bound from the instance data:
for every distinct compatible station set S, the operations whose compatible
set lies inside S must run on |S| stations, so
  LB_S = min head + (sum of minimum processing times) / |S| + min tail
over those operations, where the head of O_ij is the shortest chain of
processing times and waits before it and the tail the shortest chain of
waits and processing times after its completion. LB_cap = max_S LB_S, and
LB_max = max(chain, cap) is a valid lower bound that combines both
relaxations. The capacity bound is reported only to show where the chain
bound is loose; the policy never sees it.

Sets: the five test sets at the true waits, and the 10-module validation
orders with halved waits on which the policy was selected (the training
distribution). Every schedule is re-validated against the waits of its set.

Writes test_results/PPVC/bound_signal/bound_signal_<set>.json, a summary
bound_signal.json and bound_signal.md. Existing files are not overwritten.

usage
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 taskset -c 8 \
      python analysis/analysis_bound_signal.py --set 10x25+ppvc-mixed
  python analysis/analysis_bound_signal.py --summarize
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
OUT = os.path.join(REPO, 'test_results', 'PPVC', 'bound_signal')
MODEL = '10x25+ppvc-mixed+a1lagx05-2k-s301'
SEED_TEST = 50
TEST_SETS = ('10x25+ppvc-mixed', '20x25+ppvc-mixed', '30x25+ppvc-mixed',
             '50x25+ppvc-mixed', '80x25+ppvc-mixed')
TRAIN_SET = 'vali:10x25+ppvc-mixed-lagx0.5'
TOL = 1e-6


def set_dir(name):
    if name.startswith('vali:'):
        return os.path.join(REPO, 'data', 'data_train_vali', 'PPVC', name[5:])
    return os.path.join(REPO, 'data', 'PPVC', name)


def head_tail(jl, min_pt, lag):
    head = np.zeros(len(min_pt))
    tail = np.zeros(len(min_pt))
    first = 0
    for n in jl:
        idx = np.arange(first, first + int(n))
        h = 0.0
        for o in idx:
            head[o] = h
            h += min_pt[o] + lag[o]
        t = 0.0
        for o in idx[::-1]:
            # time from the completion of o to the completion of its module
            tail[o] = t if o == idx[-1] else lag[o] + t
            t = min_pt[o] + (lag[o] + t if o != idx[-1] else 0.0)
        first += int(n)
    return head, tail


def capacity_bound(jl, pt, lag):
    comp = pt > 0
    min_pt = np.where(comp, pt, np.inf).min(axis=1).astype(float)
    head, tail = head_tail(jl, min_pt, lag)
    sets = {tuple(np.flatnonzero(row)) for row in comp}
    best = 0.0
    for S in sets:
        inside = np.all(~comp[:, [k for k in range(pt.shape[1]) if k not in S]], axis=1) \
            if len(S) < pt.shape[1] else np.ones(len(min_pt), bool)
        if not inside.any():
            continue
        lb = head[inside].min() + min_pt[inside].sum() / len(S) + tail[inside].min()
        best = max(best, float(lb))
    return best


def run_set(name, max_orders=0):
    from eval_byproducts import import_eval_ppvc, load_policy
    ep = import_eval_ppvc()
    import torch
    from ppvc_instance_generator import load_instance
    from schedule_validator import validate_schedule
    from common_utils import greedy_select_action
    from analysis_bound_tightness import chain_bounds

    ppo, use_lag, use_type, ckpt = load_policy(ep, MODEL, SEED_TEST)
    from params import configs
    assert not bool(getattr(configs, 'bound_lag_free', False))
    assert not use_lag and not use_type
    stems = ep.list_instances(set_dir(name))
    if max_orders > 0:
        stems = stems[:max_orders]
    orders, t0 = [], time.time()
    for i, stem in enumerate(stems):
        jl, pt, meta = load_instance(stem)
        n_ops, M = int(pt.shape[0]), pt.shape[1]
        env, state = ep._new_env(jl, pt, meta, use_lag, use_type, bound_lag_free=False)
        scale = env.pt_upper_bound - env.pt_lower_bound + 1e-8
        min_pt = np.where(pt > 0, pt, np.inf).min(axis=1).astype(float)
        lag = np.asarray(meta['time_lag'], dtype=float)
        aug, free = [], []
        assigned = np.full(n_ops, -1, dtype=int)
        for step in range(n_ops + 1):
            sched = env.op_scheduled_flag[0].astype(bool)
            d = chain_bounds(jl, min_pt, lag, env.true_op_ct[0], sched, 0.0)
            a_env = float(env.max_endTime[0] * scale) if step > 0 else float(env.init_quality[0] * scale)
            assert abs(a_env - d[0]) < TOL, (stem, step, a_env, d[0])
            aug.append(float(d[0]))
            free.append(float(d[1]))
            if step == n_ops:
                break
            with torch.no_grad():
                pi, _ = ppo.policy(fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor,
                                   candidate=state.candidate_tensor, fea_m=state.fea_m_tensor,
                                   mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor,
                                   dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                                   fea_pairs=state.fea_pairs_tensor,
                                   op_type=state.op_type_tensor, mch_type=state.mch_type_tensor)
            a = int(greedy_select_action(pi).cpu().numpy()[0])
            assigned[int(env.candidate[0, a // M])] = a % M
            state, _, _ = env.step(actions=np.array([a]))
        cmax = float(env.current_makespan[0])
        val = validate_schedule(jl, pt, meta['time_lag'], assigned, env.true_op_ct[0])
        assert val['feasible'] and abs(val['makespan'] - cmax) < TOL
        o = {'instance': os.path.basename(stem), 'T': n_ops, 'cmax': cmax}
        for b, arr in (('aug', np.array(aug)), ('free', np.array(free))):
            assert abs(arr[-1] - cmax) < TOL and np.all(arr[:-1] <= cmax + TOL)
            r = arr[:-1] - arr[1:]
            rise = cmax - arr[0]
            o[f's0_ratio_{b}'] = float(arr[0] / cmax)
            o[f'zero_share_{b}'] = float(np.mean(np.abs(r) < 1e-9))
            o[f'early_share_{b}'] = (float((arr[n_ops // 2] - arr[0]) / rise)
                                     if rise > TOL else None)
        cap = capacity_bound(jl, pt, lag)
        assert cap <= cmax + TOL, (stem, cap, cmax)
        o['s0_ratio_cap'] = cap / cmax
        o['s0_ratio_max'] = max(cap, aug[0]) / cmax
        orders.append(o)
        if (i + 1) % 10 == 0 or i == len(stems) - 1:
            print(f'[{name}] {i + 1}/{len(stems)} cmax={cmax:.0f} {time.time() - t0:.0f}s', flush=True)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"bound_signal_{name.replace(':', '_')}.json")
    assert not os.path.exists(path), f'{path} exists; not overwritten'
    json.dump({'set': name, 'model': MODEL, 'checkpoint': os.path.relpath(ckpt, '.'),
               'device': 'cpu', 'decoding': 'greedy', 'n_orders': len(orders),
               'affinity': sorted(os.sched_getaffinity(0)), 'wall_s': time.time() - t0,
               'orders': orders}, open(path, 'w'), indent=1)
    print('wrote', path)


def summarize():
    rows = []
    for name in (TRAIN_SET,) + TEST_SETS:
        p = os.path.join(OUT, f"bound_signal_{name.replace(':', '_')}.json")
        o = json.load(open(p))['orders']
        r = {'set': name, 'n': len(o), 'source': os.path.relpath(p, REPO)}
        for k in ('s0_ratio_aug', 's0_ratio_free', 's0_ratio_cap', 's0_ratio_max',
                  'zero_share_aug', 'zero_share_free', 'early_share_aug', 'early_share_free'):
            v = [x[k] for x in o if x[k] is not None]
            r[k] = float(np.mean(v))
        r['mean_cmax'] = float(np.mean([x['cmax'] for x in o]))
        rows.append(r)
    json.dump({'script': 'analysis_bound_signal.py', 'model': MODEL, 'rows': rows},
              open(os.path.join(OUT, 'bound_signal.json'), 'w'), indent=1)
    L = ['# Reward signal of the two bounds and a capacity bound (SLAB seed 301, CPU, greedy)', '',
         '| Set | n | aug s0/C | free s0/C | cap s0/C | max(aug,cap) s0/C | zero-reward steps aug | free | '
         'rise in first half aug | free |', '|---|---:|' + '---:|' * 8]
    for r in rows:
        L.append(f"| {r['set']} | {r['n']} | {r['s0_ratio_aug']:.3f} | {r['s0_ratio_free']:.3f} | "
                 f"{r['s0_ratio_cap']:.3f} | {r['s0_ratio_max']:.3f} | {r['zero_share_aug']:.3f} | "
                 f"{r['zero_share_free']:.3f} | {r['early_share_aug']:.3f} | {r['early_share_free']:.3f} |")
    open(os.path.join(OUT, 'bound_signal.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    os.chdir(REPO)
    ap = argparse.ArgumentParser()
    ap.add_argument('--set')
    ap.add_argument('--max_orders', type=int, default=0)
    ap.add_argument('--summarize', action='store_true')
    a = ap.parse_args()
    if a.summarize:
        summarize()
    else:
        run_set(a.set, a.max_orders)
