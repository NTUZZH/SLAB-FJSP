"""
eval_wip_peaks.py
=================
Peaks of work in progress (WIP) and of lag-buffer occupancy on the makespan
schedules of the SLAB policy and of the SPT and LLF rules, on the nominal test
sets of 10, 20, 30, 50 and 80 modules. Extends the by-product measures of
eval_byproducts.py (whose stored files are not touched) with peak and
upper-percentile measures.

The schedules are rebuilt on the CPU with one torch thread exactly as
eval_byproducts.py builds them (eval_byproducts.import_eval_ppvc, load_policy
and run_validated: greedy decoding for the policy, eval_ppvc's tie seeds
seed_test + order index for the rules, every schedule validated against the
true waits). The run time is recorded for bookkeeping only and is not a timing
measurement.

MEASURES (per order; times in hours, release time 0 for every module)
  makespan     max completion time
  WIP(t)       number of modules that have started their first operation and
               not finished their last at time t: module j is in process on
               the half-open interval [S_j, C_j), S_j the start of its first
               and C_j the completion of its last operation
  wip_peak     max of WIP(t) over [0, makespan]
  wip_mean     time average of WIP(t) over [0, makespan] = sum_j (C_j - S_j) /
               makespan (eval_byproducts wip_mean; checked)
  wip_p95      95th percentile of WIP(t) over time: the smallest w such that
               WIP(t) <= w for at least 95% of [0, makespan]
  buffer_peak  max number of modules in a post-operation wait at the same
               instant: every non-terminal operation i with wait l_i > 0 holds
               its module in a buffer over [c_i, c_i + l_i) (the window of
               eval_buffers.py); also per lag type (curing, ponding, drying)
  buffer_mean  time average of the buffer occupancy over [0, makespan]
  flow_mean    mean over modules of C_j - S_j (eval_byproducts flow_mean; checked)
  At equal times an interval end is counted before a start, so [a, t) and
  [t, b) do not overlap.

CHECKS (stored in each output file and repeated by --summarize)
  * makespan per order equals test_results/PPVC/<set>/byproducts_hstar/
    byproducts_<greedy|SPT|LLF>.json for SLAB seed 301, SPT and LLF;
  * wip_mean and flow_mean equal the stored by-product values;
  * buffer_peak and its per-type split equal the stored
    test_results/PPVC/<set>/buffers_occupancy_SPT.json for SPT.

OUTPUT
  test_results/PPVC/wip_peaks/<set>/wip_<method>.json   per method and set
  test_results/PPVC/wip_peaks/wip_peaks_summary.json and .md   (--summarize)
  Existing files are never overwritten.

CLI (run from the repository root)
  python eval_wip_peaks.py --set 80x25+ppvc-mixed [--methods SLAB-s301 SPT LLF]
  python eval_wip_peaks.py --summarize
"""
import argparse
import datetime
import json
import os
import sys
import time

import numpy as np

from eval_ood import SLAB_MODELS, provenance, sha256_of

OUT_ROOT = os.path.join('test_results', 'PPVC', 'wip_peaks')
SETS = ['10x25+ppvc-mixed', '20x25+ppvc-mixed', '30x25+ppvc-mixed',
        '50x25+ppvc-mixed', '80x25+ppvc-mixed']
RULES = ('SPT', 'LLF')
ALL_METHODS = tuple(SLAB_MODELS) + RULES
HSTAR_NAME = {'SLAB-s301': 'greedy', 'SPT': 'SPT', 'LLF': 'LLF'}
LAG_TYPE_BY_OPNAME = {'concrete_pour': 'curing', 'waterproofing': 'ponding',
                      'paint_coat_1': 'drying', 'paint_coat_2': 'drying'}
LAG_TYPES = ('curing', 'ponding', 'drying')
SCALARS = ('makespan', 'wip_peak', 'wip_mean', 'wip_p95', 'buffer_peak',
           'buffer_mean', 'flow_mean')


# ---------------------------------------------------------------------------
# Pure measures (numpy only)
# ---------------------------------------------------------------------------

def step_profile(starts, ends, horizon):
    """Occupancy of half-open intervals [s, e) as a step function on
    [0, horizon]: returns (breakpoints t_0 = 0 < ... < t_K = horizon, values
    w_k on [t_k, t_{k+1})). Ends are processed before starts at equal times."""
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)
    assert (ends >= starts).all() and (starts >= 0).all() and (ends <= horizon + 1e-9).all()
    times = np.concatenate([starts, ends])
    delta = np.concatenate([np.ones(len(starts)), -np.ones(len(ends))])
    order = np.lexsort((delta, times))    # time, then -1 (end) before +1 (start)
    times, delta = times[order], delta[order]
    bps, vals = [0.0], []
    cur = 0
    k = 0
    while k < len(times):
        t = times[k]
        if t > bps[-1]:
            vals.append(cur)
            bps.append(float(t))
        while k < len(times) and times[k] == t:
            cur += int(delta[k])
            k += 1
    assert cur == 0
    if horizon > bps[-1]:
        vals.append(cur)
        bps.append(float(horizon))
    return np.array(bps), np.array(vals, dtype=int)


def time_quantile(bps, vals, q):
    """Smallest w with WIP <= w on at least a share q of [t_0, t_K]."""
    dur = np.diff(bps)
    order = np.argsort(vals, kind='stable')
    cum = np.cumsum(dur[order])
    target = q * (bps[-1] - bps[0])
    k = int(np.searchsorted(cum, target - 1e-9))
    return int(vals[order][min(k, len(order) - 1)])


def wip_measures(job_length, op_pt, time_lag, op_name, assigned_mch, op_ct):
    op_pt = np.asarray(op_pt, dtype=float)
    ct = np.asarray(op_ct, dtype=float)
    asg = np.asarray(assigned_mch, dtype=int)
    lag = np.asarray(time_lag, dtype=float)
    st = ct - op_pt[np.arange(len(ct)), asg]
    makespan = float(ct.max())
    first, last, idx = [], [], 0
    for n in np.asarray(job_length, dtype=int):
        first.append(idx)
        last.append(idx + int(n) - 1)
        idx += int(n)
    S, C = st[first], ct[last]
    bps, vals = step_profile(S, C, makespan)
    area = float(np.sum(np.diff(bps) * vals))
    flow = C - S
    assert abs(area - flow.sum()) < 1e-6, 'WIP area differs from the sum of flow times'

    # lag buffer windows [c_i, c_i + l_i) of non-terminal ops with a wait
    nonterminal = np.ones(len(ct), dtype=bool)
    nonterminal[last] = False
    win = np.nonzero(nonterminal & (lag > 0))[0]
    cats = []
    for i in win:
        base = op_name[i].split(':', 1)[1] if ':' in op_name[i] else op_name[i]
        assert base in LAG_TYPE_BY_OPNAME, f'wait on an unmapped operation {op_name[i]}'
        cats.append(LAG_TYPE_BY_OPNAME[base])
    ws, we = ct[win], ct[win] + lag[win]
    b_bps, b_vals = step_profile(ws, we, max(makespan, float(we.max()) if len(we) else 0.0))
    by_type = {}
    for t in LAG_TYPES:
        sel = np.array([c == t for c in cats], dtype=bool)
        if sel.any():
            _b, v = step_profile(ws[sel], we[sel], max(makespan, float(we[sel].max())))
            by_type[t] = int(v.max())
        else:
            by_type[t] = 0
    return {
        'makespan': makespan,
        'wip_peak': int(vals.max()),
        'wip_mean': area / makespan,
        'wip_p95': time_quantile(bps, vals, 0.95),
        'buffer_peak': int(b_vals.max()) if len(b_vals) else 0,
        'buffer_peak_by_type': by_type,
        'buffer_mean': float(np.sum(np.diff(b_bps) * b_vals)) / makespan,
        'n_buffer_windows': int(len(win)),
        'flow_mean': float(flow.mean()),
        'flow_max': float(flow.max()),
        'first_start_max': float(S.max()),
    }


# ---------------------------------------------------------------------------
# Stored files for the checks
# ---------------------------------------------------------------------------

def stored_byproducts(set_name, method):
    name = HSTAR_NAME.get(method)
    if name is None:
        return None
    path = os.path.join('test_results', 'PPVC', set_name, 'byproducts_hstar',
                        f'byproducts_{name}.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    if method.startswith('SLAB'):
        assert d['model'] == SLAB_MODELS[method], f'{path}: model {d["model"]}'
    return path, {r['instance']: r for r in d['per_instance']}


def stored_buffers(set_name, method):
    if method != 'SPT':
        return None
    path = os.path.join('test_results', 'PPVC', set_name, 'buffers_occupancy_SPT.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return path, {r['instance']: r for r in d['per_instance']}


def run_checks(set_name, method, rows):
    out = {}
    sb = stored_byproducts(set_name, method)
    if sb is not None:
        path, ref = sb
        keys = [r['instance'] for r in rows]
        assert all(k in ref for k in keys), f'{path}: orders missing'
        ms = sum(rows[i]['makespan'] != ref[k]['makespan'] for i, k in enumerate(keys))
        wip = max(abs(rows[i]['wip_mean'] - ref[k]['wip_mean']) for i, k in enumerate(keys))
        flo = max(abs(rows[i]['flow_mean'] - ref[k]['flow_mean']) for i, k in enumerate(keys))
        out['byproducts_hstar'] = {'file': path, 'orders': len(keys),
                                   'makespan_mismatches': int(ms),
                                   'wip_mean_max_abs_diff': float(wip),
                                   'flow_mean_max_abs_diff': float(flo)}
    sbuf = stored_buffers(set_name, method)
    if sbuf is not None:
        path, ref = sbuf
        keys = [r['instance'] for r in rows if r['instance'] in ref]
        by = {r['instance']: r for r in rows}
        out['buffers_occupancy'] = {
            'file': path, 'orders': len(keys),
            'makespan_mismatches': int(sum(by[k]['makespan'] != ref[k]['makespan'] for k in keys)),
            'peak_mismatches': int(sum(by[k]['buffer_peak'] != ref[k]['peak'] for k in keys)),
            'peak_by_type_mismatches': int(sum(by[k]['buffer_peak_by_type'] != ref[k]['peak_by_type']
                                               for k in keys))}
    return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_set(set_name, methods, seed_test, max_instances, save_dir):
    import eval_byproducts as eb
    ep = eb.import_eval_ppvc()
    import torch
    from ppvc_instance_generator import load_instance
    assert torch.get_num_threads() == 1, 'one torch thread expected'
    data_path = os.path.join('data', 'PPVC', set_name)
    stems = ep.list_instances(data_path)
    if max_instances > 0:
        stems = stems[:max_instances]
    ep.ROUTE_VARIANT = ep.detect_route_variant(data_path, 'auto')
    assert ep.ROUTE_VARIANT == 'fixed'
    out_dir = save_dir or os.path.join(OUT_ROOT, set_name)
    os.makedirs(out_dir, exist_ok=True)
    instances = [load_instance(s) for s in stems]
    names = [os.path.basename(s) for s in stems]
    for method in methods:
        path = os.path.join(out_dir, f'wip_{method}.json')
        if os.path.exists(path):
            print(f'  {path} exists; skipped (never overwritten)', flush=True)
            continue
        started = datetime.datetime.now().isoformat(timespec='seconds')
        t0 = time.time()
        policy, ckpt, ckpt_sha, model = None, None, None, None
        if method in SLAB_MODELS:
            model = SLAB_MODELS[method]
            ppo, use_lag, use_type, ckpt = eb.load_policy(ep, model, seed_test)
            policy = (ppo, use_lag, use_type)
            ckpt_sha = sha256_of(ckpt)
            run_method = 'greedy'
        else:
            run_method = method
        rows = []
        for i, (stem, (jl, pt, meta)) in enumerate(zip(names, instances)):
            ms, _sec, asg, ct = eb.run_validated(ep, run_method, jl, pt, meta, i,
                                                 seed_test, policy)
            r = wip_measures(jl, pt, meta['time_lag'], meta['op_name'], asg, ct)
            assert abs(r['makespan'] - ms) < 1e-6
            r['instance'] = stem
            rows.append(r)
        elapsed = time.time() - t0
        checks = run_checks(set_name, method, rows) if max_instances == 0 else {}
        out = {
            'dataset': set_name, 'method': method, 'n_instances': len(rows),
            'model': model, 'checkpoint': ckpt, 'checkpoint_sha256': ckpt_sha,
            'decoding': 'greedy' if model else None, 'seed_test': seed_test,
            'tie_seed': None if model else 'seed_test + order index (eval_ppvc rules)',
            'validated': True,
            'definitions': {k: v for k, v in [
                ('wip', 'modules between the start of their first and the completion of their last '
                        'operation, half-open [S_j, C_j)'),
                ('wip_p95', 'smallest w with WIP(t) <= w on at least 95% of [0, makespan]'),
                ('buffer', 'modules in a post-operation wait, windows [c_i, c_i + l_i) of '
                           'non-terminal operations with l_i > 0'),
                ('ties', 'an interval end is counted before a start at the same time')]},
            'checks': checks,
            'provenance': provenance(),
            'started': started, 'elapsed_s': elapsed,
            'elapsed_note': 'bookkeeping only, not a timing measurement',
            'aggregate': {k: float(np.mean([r[k] for r in rows])) for k in SCALARS},
            'per_instance': rows,
        }
        with open(path, 'w') as f:
            json.dump(out, f, indent=1)
        a = out['aggregate']
        print(f'  {set_name} {method}: makespan {a["makespan"]:.1f}, WIP peak {a["wip_peak"]:.2f}, '
              f'mean {a["wip_mean"]:.2f}, p95 {a["wip_p95"]:.2f}, buffer peak {a["buffer_peak"]:.2f}; '
              f'checks {json.dumps(checks)}; {elapsed:.0f} s', flush=True)


def summarize(root=OUT_ROOT):
    sys.argv = [sys.argv[0]]
    from stats_tests import rank_biserial, wilcoxon_paired
    rows, missing, check_rows = [], [], []
    for set_name in SETS:
        res = {}
        for m in ALL_METHODS:
            p = os.path.join(root, set_name, f'wip_{m}.json')
            if os.path.exists(p):
                with open(p) as f:
                    res[m] = json.load(f)
        if not res:
            missing.append(set_name)
            continue
        n_mod = int(set_name.split('x', 1)[0])
        inst = None
        per = {}
        for m, d in res.items():
            names = [r['instance'] for r in d['per_instance']]
            inst = inst or names
            assert names == inst, f'{set_name} {m}: order mismatch'
            per[m] = {k: np.array([r[k] for r in d['per_instance']], dtype=float) for k in SCALARS}
            # repeat the checks from the stored files
            chk = run_checks(set_name, m, d['per_instance'])
            check_rows.append({'set': set_name, 'method': m, **chk})
            rows.append({'set': set_name, 'n_modules': n_mod, 'method': m,
                         'n_orders': len(names),
                         'mean': {k: float(per[m][k].mean()) for k in SCALARS},
                         'max': {k: float(per[m][k].max()) for k in SCALARS}})
        seeds = [m for m in SLAB_MODELS if m in per]
        if seeds:
            sm = {k: np.mean([per[m][k] for m in seeds], axis=0) for k in SCALARS}
            per['SLAB seed mean'] = sm
            rows.append({'set': set_name, 'n_modules': n_mod, 'method': 'SLAB seed mean',
                         'seeds': seeds, 'n_orders': len(inst),
                         'mean': {k: float(sm[k].mean()) for k in SCALARS},
                         'max': {k: float(sm[k].max()) for k in SCALARS}})
            # paired comparison of the SLAB seed mean with each rule
            for rule in RULES:
                if rule not in per:
                    continue
                for k in ('wip_peak', 'wip_p95', 'buffer_peak', 'wip_mean', 'flow_mean'):
                    d = sm[k] - per[rule][k]
                    rows[-1].setdefault('vs_rule', {}).setdefault(rule, {})[k] = {
                        'mean_diff': float(d.mean()),
                        'n_lower': int((d < 0).sum()), 'n_equal': int((d == 0).sum()),
                        'n_higher': int((d > 0).sum()),
                        'wilcoxon_p': wilcoxon_paired(sm[k], per[rule][k]),
                        'rank_biserial_r': rank_biserial(sm[k], per[rule][k])}
    out = {'rows': rows, 'checks': check_rows, 'missing': missing}
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, 'wip_peaks_summary.json'), 'w') as f:
        json.dump(out, f, indent=1)

    L = ['# Work in progress and lag-buffer peaks of the makespan schedules', '',
         '- Nominal test sets (all orders). SLAB: greedy decoding of seeds 301, 302, 303 on the CPU with '
         'one torch thread; rules with eval_ppvc tie seeds (50 + order index). Schedules validated '
         'against the true waits.',
         '- WIP(t): modules between the start of their first and the completion of their last operation. '
         'Peak = max over [0, makespan]; P95 = 95th percentile over time; mean = time average '
         '(= sum of flow times / makespan). Buffer: modules in a post-operation wait at the same instant.',
         '- Values are means over orders; "max" columns are the largest value over the orders.', '',
         '| modules | method | orders | makespan | mean flow | WIP mean | WIP P95 | WIP peak | WIP peak (max) | '
         'buffer mean | buffer peak | buffer peak (max) |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        a, mx = r['mean'], r['max']
        L.append(f'| {r["n_modules"]} | {r["method"]} | {r["n_orders"]} | {a["makespan"]:.1f} | '
                 f'{a["flow_mean"]:.1f} | {a["wip_mean"]:.2f} | {a["wip_p95"]:.2f} | {a["wip_peak"]:.2f} | '
                 f'{mx["wip_peak"]:.0f} | {a["buffer_mean"]:.2f} | {a["buffer_peak"]:.2f} | '
                 f'{mx["buffer_peak"]:.0f} |')
    L += ['', '## SLAB seed mean against each rule (per-order differences, SLAB minus rule)', '',
          '| modules | rule | measure | mean diff | lower/equal/higher | p | r |',
          '|---|---|---|---|---|---|---|']
    def fp(p):
        return f'{p:.1e}' if p < 1e-3 else f'{p:.3f}'
    for r in rows:
        for rule, ks in r.get('vs_rule', {}).items():
            for k, v in ks.items():
                L.append(f'| {r["n_modules"]} | {rule} | {k} | {v["mean_diff"]:+.2f} | '
                         f'{v["n_lower"]}/{v["n_equal"]}/{v["n_higher"]} | '
                         f'{fp(v["wilcoxon_p"])} | {v["rank_biserial_r"]:.2f} |')
    L += ['', '## Checks against the stored files', '',
          '| set | method | by-product file: makespan mismatches, max abs diff WIP mean, flow mean | '
          'buffer file: makespan / peak / per-type mismatches |', '|---|---|---|---|']
    for c in check_rows:
        b = c.get('byproducts_hstar')
        u = c.get('buffers_occupancy')
        bs = (f'{b["makespan_mismatches"]}/{b["orders"]}, {b["wip_mean_max_abs_diff"]:.1e}, '
              f'{b["flow_mean_max_abs_diff"]:.1e}') if b else 'no stored file'
        us = (f'{u["makespan_mismatches"]} / {u["peak_mismatches"]} / {u["peak_by_type_mismatches"]} '
              f'of {u["orders"]}') if u else 'no stored file'
        L.append(f'| {c["set"]} | {c["method"]} | {bs} | {us} |')
    if missing:
        L += ['', 'Missing sets: ' + ', '.join(missing)]
    with open(os.path.join(root, 'wip_peaks_summary.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    ap = argparse.ArgumentParser(description='WIP and buffer peaks of SLAB, SPT and LLF schedules')
    ap.add_argument('--set')
    ap.add_argument('--methods', nargs='+', default=list(ALL_METHODS), choices=ALL_METHODS)
    ap.add_argument('--seed_test', type=int, default=50)
    ap.add_argument('--max_instances', type=int, default=0)
    ap.add_argument('--save_dir', default=None)
    ap.add_argument('--summarize', action='store_true')
    args = ap.parse_args()
    if args.summarize:
        summarize()
        return
    if not args.set:
        ap.error('--set or --summarize is required')
    if args.max_instances > 0 and args.save_dir is None:
        ap.error('--max_instances needs --save_dir')
    if args.set not in SETS and args.save_dir is None:
        ap.error(f'{args.set} is not one of {SETS}; use --save_dir for other sets')
    evaluate_set(args.set, args.methods, args.seed_test, args.max_instances, args.save_dir)


if __name__ == '__main__':
    main()
