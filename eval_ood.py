"""
eval_ood.py
===========
Out-of-distribution evaluation of the SLAB policy and the SPT and LLF rules on
the paired structural-shift test sets written by make_ood_testsets.py, plus
the nominal sets and the existing x1.5 wait sets.

Every schedule comes from the validated rollouts of eval_ppvc.py (greedy
decoding for the policy, eval_ppvc's tie seeds seed_test + order index for the
rules), run on the CPU with one torch thread, and is re-validated against the
waits of its own set with schedule_validator before it is counted. Greedy
decoding is deterministic for a fixed device and thread count, so these
CPU makespans can differ from GPU-evaluated makespans of the same checkpoint
on some orders. The run time is recorded for bookkeeping only and is not a
timing measurement.

METHODS
  SLAB-s301, SLAB-s302, SLAB-s303   trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s30{1,2,3}.pth
  SPT, LLF                          eval_ppvc dispatching rules

PER-ORDER MEASURES
  makespan     h
  queue Q      sum over consecutive operations of s_{i,j+1} - c_ij - l_ij (h):
               time a module waits for a station after its wait has elapsed
  wait W       sum of the waits l_ij between consecutive operations (h)
  sigma        Q / (Q + W), the share of module idle time between operations
               that is spent queuing (eval_byproducts queue_wait and lag_hold)

OUTPUT
  test_results/PPVC/ood/<set>/ood_<method>.json   one file per method and set
  test_results/PPVC/ood/ood_summary.json and ood_summary.md   (--summarize)
  Existing files are never overwritten: a method whose file exists is skipped.

CLI (run from the repository root)
  python eval_ood.py --set 50x25+ppvc-mixed-lagx2 [--methods SLAB-s301 SPT]
  python eval_ood.py --summarize
  Pin to one core with one thread per runtime, for example
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 9 python eval_ood.py ...
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np

OUT_ROOT = os.path.join('test_results', 'PPVC', 'ood')
SLAB_MODELS = {f'SLAB-s{s}': f'10x25+ppvc-mixed+a1lagx05-2k-s{s}' for s in (301, 302, 303)}
RULES = ('SPT', 'LLF')
ALL_METHODS = tuple(SLAB_MODELS) + RULES
SIZES = (10, 50)
# (key, label, set-name template, number of orders)
CONDITIONS = [
    ('nominal', 'nominal (waits x1, 25 stations, spread +-20%)', '{n}x25+ppvc-mixed'),
    ('lagx0.5', 'waits x0.5', '{n}x25+ppvc-mixed-lagx0.5-ood'),
    ('lagx1.5', 'waits x1.5', '{n}x25+ppvc-mixed-lagx1.5'),
    ('lagx2', 'waits x2', '{n}x25+ppvc-mixed-lagx2'),
    ('lagx3', 'waits x3', '{n}x25+ppvc-mixed-lagx3'),
    ('st20', '20 stations', '{n}x20+ppvc-mixed-ood'),
    ('st30', '30 stations', '{n}x30+ppvc-mixed-ood'),
    ('st35', '35 stations', '{n}x35+ppvc-mixed-ood'),
    ('ptdev0.4', 'duration spread +-40%', '{n}x25+ppvc-mixed-ptdev0.4'),
]
N_ORDERS = {10: 100, 50: 50}
PROVENANCE_MODULES = ('eval_ppvc', 'params', 'fjsp_env_same_op_nums',
                      'common_utils', 'model.main_model', 'model.PPO',
                      'model.sub_layers', 'model.attention_layer',
                      'schedule_validator', 'eval_byproducts',
                      'ppvc_instance_generator', 'data_utils')


def all_sets():
    return [(key, n, tpl.format(n=n)) for key, _l, tpl in CONDITIONS for n in SIZES]


def sha256_of(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def interpreter():
    """sys.executable with the home directory written as ~ (no user path)."""
    home = os.path.expanduser('~')
    exe = sys.executable
    return '~' + exe[len(home):] if home and exe.startswith(home + os.sep) else exe


def provenance():
    import torch
    # paths are relative to the source root (no machine-local paths); a frozen
    # copy of a commit carries a COMMIT file, the working tree does not
    src = os.path.dirname(os.path.abspath(sys.modules['eval_ppvc'].__file__))
    mods = {}
    for name in PROVENANCE_MODULES:
        m = sys.modules.get(name)
        if m is not None and getattr(m, '__file__', None):
            mods[name] = {'path': os.path.relpath(os.path.abspath(m.__file__), src),
                          'sha256': sha256_of(m.__file__)}
    commit_file = os.path.join(src, 'COMMIT')
    return {
        'modules': mods,
        'source': ('frozen copy of a repository commit' if os.path.exists(commit_file)
                   else 'working tree'),
        'source_commit': (open(commit_file).read().strip()
                          if os.path.exists(commit_file) else None),
        'affinity': sorted(os.sched_getaffinity(0)),
        'torch_threads': torch.get_num_threads(),
        'torch_interop_threads': torch.get_num_interop_threads(),
        'env_threads': {k: os.environ.get(k) for k in
                        ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                         'OPENBLAS_NUM_THREADS', 'CUDA_VISIBLE_DEVICES')},
        'device': 'cpu',
        'python_executable': interpreter(),
        'python': platform.python_version(), 'torch': torch.__version__,
        'numpy': np.__version__,
    }


def queue_and_wait(eb, jl, pt, meta, asg, ct):
    r = eb.compute_byproducts(jl, pt, meta['time_lag'], asg, ct,
                              meta['mch_type'], np.zeros(len(jl)))
    q, w = r['queue_wait'], r['lag_hold']
    return q, w, (q / (q + w) if q + w > 0 else float('nan'))


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
    assert ep.ROUTE_VARIANT == 'fixed', f'{set_name}: fixed routes expected'
    out_dir = save_dir or os.path.join(OUT_ROOT, set_name)
    os.makedirs(out_dir, exist_ok=True)
    instances = [load_instance(s) for s in stems]
    names = [os.path.basename(s) for s in stems]
    with open(os.path.join(data_path, 'dataset_meta.json')) as f:
        ds_meta = json.load(f)

    # the three SLAB snapshots must share every architecture flag
    snaps = {}
    for m in methods:
        if m in SLAB_MODELS:
            with open(f'./train_log/PPVC/config_{SLAB_MODELS[m]}.json') as f:
                s = json.load(f)
            snaps[m] = {k: s.get(k) for k in ep.ARCH_KEYS}
    if len(snaps) > 1:
        first = next(iter(snaps.values()))
        assert all(v == first for v in snaps.values()), 'SLAB snapshots differ in ARCH_KEYS'

    written = []
    for method in methods:
        path = os.path.join(out_dir, f'ood_{method}.json')
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
            q, w, sigma = queue_and_wait(eb, jl, pt, meta, asg, ct)
            rows.append({'instance': stem, 'makespan': float(ms),
                         'queue_h': float(q), 'wait_h': float(w),
                         'sigma': float(sigma)})
        elapsed = time.time() - t0
        out = {
            'dataset': set_name, 'method': method, 'n_instances': len(rows),
            'model': model, 'checkpoint': ckpt, 'checkpoint_sha256': ckpt_sha,
            'decoding': 'greedy' if model else None,
            'seed_test': seed_test,
            'tie_seed': 'seed_test + order index (eval_ppvc rules)' if not model else None,
            'validated': True,
            'validation': 'schedule_validator.validate_schedule against the time_lag of this set; '
                          'validator makespan equals the environment makespan',
            'dataset_meta': ds_meta,
            'definitions': {
                'queue_h': 'sum over consecutive operations of successor start - predecessor '
                           'completion - wait (h)',
                'wait_h': 'sum of the waits between consecutive operations (h)',
                'sigma': 'queue_h / (queue_h + wait_h)'},
            'provenance': provenance(),
            'started': started, 'elapsed_s': elapsed,
            'elapsed_note': 'bookkeeping only, not a timing measurement',
            'mean_makespan': float(np.mean([r['makespan'] for r in rows])),
            'mean_sigma': float(np.mean([r['sigma'] for r in rows])),
            'per_instance': rows,
        }
        with open(path, 'w') as f:
            json.dump(out, f, indent=1)
        written.append(path)
        print(f'  {set_name} {method}: mean makespan {out["mean_makespan"]:.2f} h, '
              f'sigma {out["mean_sigma"]:.3f}, {len(rows)} orders, {elapsed:.0f} s', flush=True)
    return written


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def load_method(set_name, method, root=OUT_ROOT):
    path = os.path.join(root, set_name, f'ood_{method}.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def summarize(root=OUT_ROOT):
    sys.argv = [sys.argv[0]]
    from stats_tests import holm, rank_biserial, wilcoxon_paired
    rows, missing = [], []
    nominal = {}
    for key, label, tpl in CONDITIONS:
        for n in SIZES:
            set_name = tpl.format(n=n)
            res = {m: load_method(set_name, m, root) for m in ALL_METHODS}
            if any(v is None for v in res.values()):
                missing.append((set_name, [m for m, v in res.items() if v is None]))
                continue
            inst = [r['instance'] for r in res['SPT']['per_instance']]
            for m, v in res.items():
                assert [r['instance'] for r in v['per_instance']] == inst, f'{set_name} {m}: order mismatch'
            ms = {m: np.array([r['makespan'] for r in v['per_instance']]) for m, v in res.items()}
            slab = np.mean([ms[m] for m in SLAB_MODELS], axis=0)
            means = {m: float(ms[m].mean()) for m in ALL_METHODS}
            better = min(RULES, key=lambda r: means[r])
            rule = ms[better]
            d = slab - rule
            row = {
                'condition': key, 'label': label, 'n_modules': n, 'set': set_name,
                'n_orders': len(inst), 'mean_makespan': means,
                'slab_seed_mean': float(slab.mean()),
                'better_rule': better,
                'gain_pct': float(100.0 * (rule.mean() - slab.mean()) / rule.mean()),
                'gain_pct_per_seed': {m: float(100.0 * (rule.mean() - ms[m].mean()) / rule.mean())
                                      for m in SLAB_MODELS},
                'mean_per_order_gain_pct': float(np.mean(100.0 * (rule - slab) / rule)),
                'n_slab_shorter': int((d < 0).sum()), 'n_equal': int((d == 0).sum()),
                'n_slab_longer': int((d > 0).sum()),
                'wilcoxon_p': wilcoxon_paired(slab, rule),
                'rank_biserial_r': rank_biserial(slab, rule),
                'r_sign': 'r < 0: SLAB seed mean shorter than the rule',
                'sigma_spt': float(np.mean([r['sigma'] for r in res['SPT']['per_instance']])),
                'sigma_spt_pooled': float(sum(r['queue_h'] for r in res['SPT']['per_instance'])
                                          / sum(r['queue_h'] + r['wait_h'] for r in res['SPT']['per_instance'])),
                'sigma_slab_s301': float(np.mean([r['sigma'] for r in res['SLAB-s301']['per_instance']])),
                'per_order': {'instance': inst, 'slab_seed_mean': slab.tolist(),
                              **{m: ms[m].tolist() for m in ALL_METHODS}},
            }
            if key == 'nominal':
                nominal[n] = {m: ms[m] for m in ALL_METHODS}
                nominal[n]['SLAB'] = slab
            rows.append(row)
    # makespan relative to the nominal twin (paired by order)
    for row in rows:
        n = row['n_modules']
        if n not in nominal:
            continue
        rel = {}
        for m in ALL_METHODS + ('SLAB',):
            cur = np.array(row['per_order']['slab_seed_mean'] if m == 'SLAB' else row['per_order'][m])
            base = nominal[n][m][:len(cur)]
            rel[m] = float(100.0 * (cur.mean() / base.mean() - 1.0))
        row['change_vs_nominal_pct'] = rel
    adj, _rej = holm([r['wilcoxon_p'] for r in rows]) if rows else ([], [])
    for r, p in zip(rows, adj):
        r['wilcoxon_p_holm'] = float(p)

    out = {'definitions': {
        'gain_pct': '100 x (mean rule - mean SLAB seed mean) / mean rule; positive = SLAB shorter; '
                    'rule = the better of SPT and LLF by mean makespan on that set',
        'n_slab_shorter/equal/longer': 'orders on which the SLAB seed mean (mean over seeds 301-303) is '
                                       'shorter than, equal to or longer than that rule',
        'wilcoxon_p': 'two-sided Wilcoxon signed-rank test of SLAB seed mean - rule, zeros dropped '
                      '(stats_tests.wilcoxon_paired)',
        'wilcoxon_p_holm': 'Holm adjustment over all rows of this summary',
        'rank_biserial_r': 'matched-pairs rank-biserial correlation of SLAB seed mean - rule '
                           '(stats_tests.rank_biserial); r < 0 favours SLAB',
        'sigma_spt': 'mean over orders of Q / (Q + W) on the SPT schedule',
        'sigma_spt_pooled': 'sum Q / sum (Q + W) over the orders of the set, SPT schedule',
        'change_vs_nominal_pct': '100 x (mean makespan on the set / mean makespan on the nominal twins - 1)'},
        'rows': rows, 'missing': missing}
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, 'ood_summary.json'), 'w') as f:
        json.dump(out, f, indent=1)

    def fp(p):
        return f'{p:.1e}' if p < 1e-3 else f'{p:.3f}'
    L = ['# Out-of-distribution evaluation: SLAB policy against SPT and LLF', '',
         '- Paired test sets: every shifted order has the generator seed of its nominal twin and '
         'differs only in the shifted parameter (make_ood_testsets.py).',
         '- SLAB: greedy decoding of seeds 301, 302, 303 on the CPU with one torch thread; rules with '
         'eval_ppvc tie seeds (50 + order index). Every schedule is validated against the waits of its set.',
         '- Gain: 100 x (rule - SLAB seed mean) / rule on mean makespans, positive = SLAB shorter; rule = '
         'the better of SPT and LLF on that set. W/T/L: orders on which the SLAB seed mean is shorter / '
         'equal / longer. p: two-sided Wilcoxon signed-rank (zeros dropped); p (Holm) over all rows. '
         'r: matched-pairs rank-biserial, r < 0 favours SLAB.',
         '- sigma: mean over orders of Q / (Q + W) on the SPT schedule (share of module idle time '
         'between operations spent queuing).', '',
         '| condition | modules | orders | SPT | LLF | SLAB s301 | s302 | s303 | SLAB mean | rule | '
         'gain % | gain % per seed | W/T/L | p | p (Holm) | r | sigma SPT |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        m = r['mean_makespan']
        g = r['gain_pct_per_seed']
        L.append(f'| {r["label"]} | {r["n_modules"]} | {r["n_orders"]} | {m["SPT"]:.1f} | {m["LLF"]:.1f} | '
                 f'{m["SLAB-s301"]:.1f} | {m["SLAB-s302"]:.1f} | {m["SLAB-s303"]:.1f} | '
                 f'{r["slab_seed_mean"]:.1f} | {r["better_rule"]} | {r["gain_pct"]:.2f} | '
                 f'{min(g.values()):.2f} to {max(g.values()):.2f} | '
                 f'{r["n_slab_shorter"]}/{r["n_equal"]}/{r["n_slab_longer"]} | {fp(r["wilcoxon_p"])} | '
                 f'{fp(r["wilcoxon_p_holm"])} | {r["rank_biserial_r"]:.2f} | {r["sigma_spt"]:.3f} |')
    L += ['', '## Mean makespan relative to the nominal twins (%)', '',
          '| condition | modules | SPT | LLF | SLAB seed mean |', '|---|---|---|---|---|']
    for r in rows:
        c = r.get('change_vs_nominal_pct')
        if c is None:
            continue
        L.append(f'| {r["label"]} | {r["n_modules"]} | {c["SPT"]:+.1f} | {c["LLF"]:+.1f} | {c["SLAB"]:+.1f} |')
    if missing:
        L += ['', '## Missing results', ''] + [f'- {s}: {", ".join(ms)}' for s, ms in missing]
    with open(os.path.join(root, 'ood_summary.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))
    return out


def main():
    ap = argparse.ArgumentParser(description='Out-of-distribution evaluation of SLAB, SPT and LLF')
    ap.add_argument('--set', help='test set under data/PPVC/')
    ap.add_argument('--methods', nargs='+', default=list(ALL_METHODS), choices=ALL_METHODS)
    ap.add_argument('--seed_test', type=int, default=50)
    ap.add_argument('--max_instances', type=int, default=0)
    ap.add_argument('--save_dir', default=None)
    ap.add_argument('--summarize', action='store_true')
    ap.add_argument('--list_sets', action='store_true')
    args = ap.parse_args()
    if args.list_sets:
        for _k, _n, s in all_sets():
            print(s)
        return
    if args.summarize:
        summarize()
        return
    if not args.set:
        ap.error('--set or --summarize is required')
    if args.max_instances > 0 and args.save_dir is None:
        ap.error('--max_instances needs --save_dir (partial runs must not write the full-set files)')
    known = {s for _k, _n, s in all_sets()}
    if args.set not in known and args.save_dir is None:
        ap.error(f'{args.set} is not one of the study sets; use --save_dir for other sets')
    n_mod = int(args.set.split('x', 1)[0])
    max_inst = args.max_instances or (N_ORDERS.get(n_mod, 0) if args.set in known else 0)
    evaluate_set(args.set, args.methods, args.seed_test, max_inst, args.save_dir)


if __name__ == '__main__':
    main()
