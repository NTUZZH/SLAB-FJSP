#!/usr/bin/env python3
"""analysis_cpsat_regimes.py -- CP-SAT incumbents at fixed solver times and
the best proven lower bounds, from the stored anytime logs (analysis only, no
solving).

For every size of the ladder and every CP-SAT mode of the paper (cold, from
the SPT schedule, and at 50 and 80 modules from the SLAB schedule of seed
301), the script reads the logged trajectory ([time, objective, bound] per
improving solution; an SPT- or SLAB-started run logs its hint at t = 0) and
records, per order, the incumbent makespan at the checkpoints
1, 5, 10, 30, 60, 300, 600 and 1800 s (a point logged after the limit is
clamped to the limit, as in analysis_cpsat_time_to_slab.py). A checkpoint
beyond a run's limit is not filled. A cold run without a solution at a
checkpoint has no incumbent there; its count is reported and the mean over
orders is reported only when every order has one.

Per order it also takes the best lower bound proven by any CP-SAT run of
that order (the maximum of the bounds logged in all its runs, including the
1800-s and 16-worker runs where they exist) and the best known makespan (the
minimum over all CP-SAT runs, the memetic algorithm and the three SLAB
seeds). It reports the mean ratio of each to the SLAB makespan (per-order
mean of the three seeds).

The SLAB comparison values and the CP-SAT records are read through
make_paper_macros (instance ids, validator feasibility and the CPU share of
every budget-limited run are checked there).

Writes test_results/PPVC/cpsat_regimes/cpsat_regimes.json and .md.
usage: python analysis_cpsat_regimes.py
"""
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.dont_write_bytecode = True
import make_paper_macros as M  # noqa: E402

OUT = os.path.join(REPO, 'test_results', 'PPVC', 'cpsat_regimes')
SIZES = (('Ten', '10x25+ppvc-mixed', 100), ('Twenty', '20x25+ppvc-mixed', 50),
         ('Thirty', '30x25+ppvc-mixed', 20), ('Fifty', '50x25+ppvc-mixed', 50),
         ('Eighty', '80x25+ppvc-mixed', 50))
CHECKPOINTS = (1, 5, 10, 30, 60, 300, 600, 1800)
# Time of one SLAB schedule on one CPU core (greedy, batch 1, seed 301), the
# operating point at which CP-SAT is also read ("matched" checkpoint). CP-SAT
# keeps its four workers, so it receives four times the cores at that time.
COST = os.path.join(REPO, os.environ.get('COST_TABLE',
                                          'paper_inputs/cost_table_v3.json'))


def slab_latency(ds):
    c = json.load(open(COST))['cpu'][M.HSTAR_MODEL][ds.split('+')[0]]
    assert c['dataset'] == ds, (c['dataset'], ds)
    return float(c['latency_s_per_instance_mean'])


def runs_of(S, ds, n):
    """(label, budget, n_orders, records) of every CP-SAT run of one size."""
    out = []
    lim = M.cp_budget(S)
    for mode in ('cold', 'warm-spt'):
        out.append((mode, lim, n, M.anytime(ds, mode, lim, mode, n=n)))
    if S in ('Fifty', 'Eighty'):
        out.append(('warm-policy', 600, n,
                    M.anytime(ds, 'warm-policy-halflag-s301', 600, 'warm-policy',
                              model=M.HSTAR_MODEL)))
        n20 = 20
        for mode in ('cold', 'warm-spt'):
            out.append((mode + '-1800', 1800, n20,
                        M.anytime(ds, mode, 1800, mode, n=n20)))
        if S == 'Eighty':
            out.append(('warm-spt-w16', 600, n20,
                        M.anytime(ds, 'warm-spt', 600, 'warm-spt', n=n20, workers=16)))
    return out


def incumbent_at(rec, t, limit):
    best = np.inf
    for p in rec['trajectory']:
        tt, o = min(float(p[0]), float(limit)), p[1]
        if o is None or tt > t + 1e-9:
            continue
        best = min(best, float(o))
    return best


def best_bound(rec):
    b = [float(p[2]) for p in rec['trajectory'] if len(p) > 2 and p[2] is not None]
    b += [float(p[1]) for p in rec.get('bound_trajectory', []) if p[1] is not None]
    if rec.get('bound') is not None:
        b.append(float(rec['bound']))
    return max(b) if b else -np.inf


def main():
    res = {}
    for S, ds, n in SIZES:
        slab = np.asarray(M.inst_of(M.HSTAR, ds), float)[:n]
        ma = np.asarray(M.memetic(ds)[:n] if not isinstance(M.memetic(ds), dict)
                        else M.memetic(ds)['makespan'][:n], float)
        lb = np.full(n, -np.inf)
        ub = np.minimum(slab.copy(), ma)
        for i in range(n):  # each SLAB seed alone also bounds the best known makespan
            for s in (301, 302, 303):
                ub[i] = min(ub[i], float(np.asarray(M.result(M.HSTAR.format(s=s), ds))[i]))
        rows = []
        for label, limit, nr, recs in runs_of(S, ds, n):
            per_t = {}
            for c in CHECKPOINTS + ('matched',):
                c_lab = c
                if c == 'matched':
                    c = slab_latency(ds)
                if c > limit:
                    continue
                v = np.array([incumbent_at(r, c, limit) for r in recs])
                have = np.isfinite(v)
                per_t[str(c_lab)] = {
                    'time_s': float(c),
                    'n_with_solution': int(have.sum()),
                    'mean_incumbent': float(v.mean()) if have.all() else None,
                    'mean_slab_same_orders': float(slab[:nr].mean()),
                    'n_slab_shorter': int(np.sum(slab[:nr] < v - 1e-9)),
                    'n_slab_longer': int(np.sum(slab[:nr] > v + 1e-9)),
                }
            obj = np.array([float(r['objective']) for r in recs])
            for i, r in enumerate(recs):
                lb[i] = max(lb[i], best_bound(r))
                ub[i] = min(ub[i], float(r['objective']))
            rows.append({'run': label, 'limit_s': limit, 'n': nr,
                         'source': M.any_rel(ds, {'warm-policy': 'warm-policy-halflag-s301'}.get(
                             label, label.split('-1800')[0].replace('-w16', '')), limit,
                             16 if label.endswith('w16') else 4),
                         'mean_final': float(obj.mean()),
                         'n_optimal': int(sum(r['status'] == 'OPTIMAL' for r in recs)),
                         'checkpoints': per_t})
        res[S] = {
            'dataset': ds, 'n': n, 'slab_latency_s': slab_latency(ds),
            'mean_slab': float(slab.mean()),
            'mean_best_bound': float(lb.mean()),
            'mean_best_known': float(ub.mean()),
            'mean_slab_over_bound': float(np.mean(slab / lb)),
            'mean_best_known_over_bound': float(np.mean(ub / lb)),
            'mean_slab_over_best_known': float(np.mean(slab / ub)),
            'runs': rows,
        }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'cpsat_regimes.json'), 'w') as f:
        json.dump({'script': 'analysis_cpsat_regimes.py', 'checkpoints_s': CHECKPOINTS,
                   'matched_source': os.path.relpath(COST, REPO),
                   'slab_sources': {ds: M.src(M.HSTAR, ds) for _, ds, _ in SIZES},
                   'sizes': res}, f, indent=1)
    L = ['# CP-SAT incumbent at fixed solver times and best proven bounds', '',
         'Mean incumbent makespan (h) over the orders of each run; n/N = orders with a solution '
         'at that time when not all have one. SLAB = per-order mean of three seeds on the same orders.', '']
    for S, d in res.items():
        L += [f"## {d['dataset']} (SLAB {d['mean_slab']:.1f} h; best CP-SAT bound {d['mean_best_bound']:.1f} h; "
              f"best known {d['mean_best_known']:.1f} h; SLAB/bound {d['mean_slab_over_bound']:.3f}; "
              f"SLAB/best known {d['mean_slab_over_best_known']:.3f})", '',
              f"SLAB time on one CPU core: {d['slab_latency_s']:.2f} s", '',
              '| run | n | ' + ' | '.join(f'{c} s' for c in CHECKPOINTS)
              + ' | at SLAB time | final | optimal |',
              '|---|---:|' + '---:|' * (len(CHECKPOINTS) + 3)]
        for r in d['runs']:
            cells = []
            for c in CHECKPOINTS + ('matched',):
                x = r['checkpoints'].get(str(c))
                if x is None:
                    cells.append('--')
                elif x['mean_incumbent'] is None:
                    cells.append(f"{x['n_with_solution']}/{r['n']} solved")
                else:
                    cells.append(f"{x['mean_incumbent']:.1f} (SLAB {x['mean_slab_same_orders']:.1f})")
            L.append(f"| {r['run']} | {r['n']} | " + ' | '.join(cells) +
                     f" | {r['mean_final']:.1f} | {r['n_optimal']} |")
        L.append('')
    with open(os.path.join(OUT, 'cpsat_regimes.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    os.chdir(REPO)
    main()
