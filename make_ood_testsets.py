"""
make_ood_testsets.py
====================
Write the paired out-of-distribution test sets of the structural-shift study
to data/PPVC/. Every shifted order is built from the same generator seed as
its twin in a nominal test set, so a shifted order differs from its twin only
in the shifted parameter and results can be compared order by order.

SOURCES (nominal twins)
  data/PPVC/10x25+ppvc-mixed   first 100 orders (seed0 50000)
  data/PPVC/50x25+ppvc-mixed   all 50 orders    (seed0 100000)

CONDITIONS
  wait scale      every post-operation wait multiplied by 0.5, 2 or 3 and
                  rounded with np.rint, as make_lagscaled_testsets.py does for
                  1.5 (and train.py --lag_scale for training orders). Sets
                  <src>-lagx0.5-ood, <src>-lagx2, <src>-lagx3.
  station count   20, 30 or 35 stations. The default counts A3 B2 C4 D4 E3
                  F3 G1 H2 Q3 (25) are scaled by total/25 and rounded by the
                  largest-remainder method with at least one station per type
                  (apportion below). Sets <n>x<total>+ppvc-mixed-ood.
  duration spread processing time of an operation on each compatible station
                  drawn as nominal x U(1 - 0.4, 1 + 0.4) in place of
                  U(1 - 0.2, 1 + 0.2): ppvc_instance_generator.PT_DEVIATION is
                  set to 0.4 before generation with the twin's seed. Sets
                  <src>-ptdev0.4.

PAIRING (verified for every order by this script)
  wait scale      routes, stations and processing times are copied from the
                  twin; only time_lag changes.
  duration spread the generator consumes one uniform draw per (operation,
                  station) whatever the bounds, so the same seed gives the
                  same classes, module weights, routes, nominal durations and
                  waits; only the per-station processing times change. The
                  script checks this by regenerating the twin's draws.
  station count   calling the generator with other station counts does NOT
                  give a paired order: it draws one uniform per compatible
                  station, so every later draw (nominal durations, waits,
                  weights of later modules) shifts. The script therefore
                  builds the order from the twin: classes, weights, routes,
                  nominal durations and waits are the twin's; a station that
                  exists in both factories (the first k stations of each type)
                  keeps the twin's processing time column; a station added to
                  a type gets a new draw nominal x U(0.8, 1.2) from a separate
                  stream seeded with [EXTRA_STREAM, seed] (MAX_EXTRA draws per
                  operation are always consumed, so the added stations of the
                  30-station factory are the first added stations of the
                  35-station factory); a station removed from a type drops its
                  column. The factories are nested type by type
                  (20 within 25 within 30 within 35).
  The nominal durations are not stored in an instance file. They are
  recovered with generator_draws(), a step-by-step mirror of
  ppvc_instance_generator that also returns its draws; the mirror is checked
  to reproduce the stored twin exactly before it is used.

CHECKS (per order; any failure stops the script)
  * the stored twin equals ppvc_instance_generator(seed0 + i) and the mirror;
  * the pairing above, field by field;
  * structure: 22 operations per module, compatible stations are exactly the
    stations of the operation's type, processing times >= 1 there and 0
    elsewhere, waits >= 0 and 0 on the last operation of every module;
  * no overwrite: an existing instance file must equal the order this script
    would write (it is then left untouched), and an existing dataset_meta.json
    must carry the same seed0 (the check of the other make_* scripts);
  * round trip: every written order is read back with load_instance and
    compared with the order in memory.

OUTPUT
  data/PPVC/<set>/instance_XXX.fjs + .meta.json, data/PPVC/<set>/dataset_meta.json
  test_results/PPVC/ood/make_ood_testsets_report.json (counts and checks)

usage: python make_ood_testsets.py [--dry_run]
"""
import argparse
import hashlib
import json
import os
import sys
from fractions import Fraction

import numpy as np

import ppvc_instance_generator as gen
from ppvc_instance_generator import (DEFAULT_FACTORY, OPS_PER_MODULE,
                                     ROUTES, ROUTING_CLASSES, STATION_TYPES,
                                     MODULE_WEIGHT_T, OP_TYPE_NAMES,
                                     build_factory, load_instance,
                                     ppvc_instance_generator, save_instance)

DATA_ROOT = os.path.join('data', 'PPVC')
REPORT_DIR = os.path.join('test_results', 'PPVC', 'ood')
SOURCES = [  # (nominal set, number of orders used)
    ('10x25+ppvc-mixed', 100),
    ('50x25+ppvc-mixed', 50),
]
LAG_SCALES = [(0.5, '-lagx0.5-ood'), (2.0, '-lagx2'), (3.0, '-lagx3')]
STATION_TOTALS = [20, 30, 35]
PT_DEVIATIONS = [(0.4, '-ptdev0.4')]
NOMINAL_PT_DEVIATION = 0.2
EXTRA_STREAM = 104729   # separates the added-station draws from the main stream
MAX_EXTRA = 4           # added-station draws consumed per operation
META_KEYS = ('op_type', 'op_station', 'time_lag', 'mch_type', 'routing_class',
             'module_weight_t', 'op_name', 'station_counts')


# ---------------------------------------------------------------------------
# Station counts
# ---------------------------------------------------------------------------

def apportion(base, total):
    """Largest-remainder (Hamilton) apportionment of `total` stations over the
    station types in proportion to `base`, with at least one station per type.

    Each type first gets max(1, floor(quota)); the remaining stations go to the
    types with the largest remainder quota - floor(quota) (a type raised to
    the minimum of one takes no further station). Ties in the remainder go to
    the type with the larger base count, then to the type listed first in
    STATION_TYPES. Exact rational arithmetic, so ties are exact.
    """
    types = [t for t in STATION_TYPES if base.get(t, 0) > 0]
    b = sum(base[t] for t in types)
    assert total >= len(types), 'fewer stations than station types'
    quota = {t: Fraction(base[t] * total, b) for t in types}
    alloc = {t: max(1, int(quota[t])) for t in types}   # int() floors a Fraction >= 0
    left = total - sum(alloc.values())
    assert left >= 0, f'minimum of one station per type exceeds {total}'
    cand = [t for t in types if alloc[t] < quota[t]]
    cand.sort(key=lambda t: (-(quota[t] - alloc[t]), -base[t],
                             STATION_TYPES.index(t)))
    assert left <= len(cand)
    for t in cand[:left]:
        alloc[t] += 1
    assert sum(alloc.values()) == total
    return {t: alloc[t] for t in STATION_TYPES if t in alloc}


# ---------------------------------------------------------------------------
# Generator mirror (returns the draws that an instance file does not store)
# ---------------------------------------------------------------------------

def generator_draws(n_modules, class_mix, station_counts, seed, pt_deviation):
    """Step-by-step mirror of ppvc_instance_generator (fixed routes).

    Makes the same calls on the same Generator in the same order, so it
    returns the same instance, plus the draws: nominal duration per operation
    and the standard uniform v in [0, 1) behind every per-station factor
    U(1 - dev, 1 + dev) = 1 - dev + 2 dev v (one row per operation, one entry
    per station of its type, in station order).
    """
    rng = np.random.default_rng(seed)
    station_counts = dict(station_counts)
    mch_type, type_ranges = build_factory(station_counts)
    if isinstance(class_mix, str):
        class_mix = gen.PRESET_MIXES[class_mix]
    classes = rng.choice(len(ROUTING_CLASSES), size=n_modules, p=list(class_mix))
    n_ops = OPS_PER_MODULE * n_modules
    op_pt = np.zeros((n_ops, len(mch_type)), dtype=int)
    op_type = np.zeros(n_ops, dtype=int)
    op_station = np.zeros(n_ops, dtype=int)
    time_lag = np.zeros(n_ops, dtype=int)
    nominal = np.zeros(n_ops, dtype=int)
    v = []
    weight = np.zeros(n_modules)
    names = []
    row = 0
    for j in range(n_modules):
        cls = ROUTING_CLASSES[classes[j]]
        shell = 'RC' if cls.startswith('RC') else 'Steel'
        w_lo, w_hi = MODULE_WEIGHT_T[shell]
        weight[j] = round(rng.uniform(w_lo, w_hi), 1)
        for (name, s_type, o_type, (d_lo, d_hi), (l_lo, l_hi)) in ROUTES[cls]:
            nom = rng.integers(d_lo, d_hi + 1)
            nominal[row] = nom
            vs = []
            for m in type_ranges[s_type]:
                u = rng.uniform(1 - pt_deviation, 1 + pt_deviation)
                vs.append((u - (1 - pt_deviation)) / (2 * pt_deviation))
                op_pt[row, m] = max(1, int(round(nom * u)))
            v.append(vs)
            op_type[row] = OP_TYPE_NAMES.index(o_type)
            op_station[row] = STATION_TYPES.index(s_type)
            time_lag[row] = rng.integers(l_lo, l_hi + 1) if l_hi > 0 else 0
            names.append(f'M{j}:{name}')
            row += 1
    jl = np.full(n_modules, OPS_PER_MODULE, dtype=int)
    meta = {'op_type': op_type, 'op_station': op_station, 'time_lag': time_lag,
            'mch_type': mch_type, 'routing_class': classes.astype(int),
            'module_weight_t': weight, 'op_name': names,
            'station_counts': station_counts}
    return (jl, op_pt, meta), {'nominal': nominal, 'v': v, 'classes': classes}


def generate_with_deviation(n_modules, class_mix, station_counts, seed, dev):
    """ppvc_instance_generator with PT_DEVIATION = dev for this call only."""
    saved = gen.PT_DEVIATION
    gen.PT_DEVIATION = dev
    try:
        return ppvc_instance_generator(n_modules=n_modules, class_mix=class_mix,
                                       station_counts=station_counts, seed=seed)
    finally:
        gen.PT_DEVIATION = saved


# ---------------------------------------------------------------------------
# Comparison and structural checks
# ---------------------------------------------------------------------------

def _eq(a, b):
    if isinstance(a, dict) or isinstance(b, dict):
        return {str(k): int(v) for k, v in dict(a).items()} == \
               {str(k): int(v) for k, v in dict(b).items()}
    if isinstance(a, list) and a and isinstance(a[0], str):
        return list(a) == list(b)
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and np.array_equal(a, b)


def diff_fields(x, y, keys=META_KEYS):
    """Names of the fields in which two instances differ."""
    (jl1, pt1, m1), (jl2, pt2, m2) = x, y
    out = []
    if not _eq(jl1, jl2):
        out.append('job_length')
    if not _eq(pt1, pt2):
        out.append('op_pt')
    for k in keys:
        if (k in m1) != (k in m2) or (k in m1 and not _eq(m1[k], m2[k])):
            out.append(k)
    extra = (set(m1) ^ set(m2)) - set(keys)
    out.extend(sorted(f'key:{k}' for k in extra))
    return out


def check_structure(inst, label):
    jl, pt, meta = inst
    n_mod = len(jl)
    assert (np.asarray(jl) == OPS_PER_MODULE).all(), f'{label}: route length'
    assert pt.shape[0] == OPS_PER_MODULE * n_mod, f'{label}: op count'
    mch_type = np.asarray(meta['mch_type'])
    assert pt.shape[1] == len(mch_type), f'{label}: station count'
    counts = {t: int((mch_type == i).sum()) for i, t in enumerate(STATION_TYPES)}
    sc = {t: int(c) for t, c in meta['station_counts'].items()}
    assert {t: c for t, c in counts.items() if c} == {t: c for t, c in sc.items() if c}, \
        f'{label}: station_counts does not match mch_type'
    compat_expected = (np.asarray(meta['op_station'])[:, None] == mch_type[None, :])
    assert np.array_equal(pt > 0, compat_expected), f'{label}: compatibility'
    assert (pt[compat_expected] >= 1).all(), f'{label}: processing time < 1'
    lag = np.asarray(meta['time_lag'])
    assert (lag >= 0).all(), f'{label}: negative wait'
    last = np.cumsum(jl) - 1
    assert (lag[last] == 0).all(), f'{label}: wait on a last operation'


# ---------------------------------------------------------------------------
# Shifted orders
# ---------------------------------------------------------------------------

def lag_scaled(src, scale):
    jl, pt, meta = src
    meta = dict(meta)
    meta['time_lag'] = np.rint(np.asarray(meta['time_lag'], dtype=float)
                               * scale).astype(int)
    return jl, pt.copy(), meta


def station_shifted(src, nominal, seed, new_counts):
    """Order on another factory, paired with `src` (see PAIRING)."""
    jl, pt_src, meta = src
    src_counts = {t: int(c) for t, c in meta['station_counts'].items()}
    _, src_ranges = build_factory(src_counts)
    new_mch_type, new_ranges = build_factory(new_counts)
    xrng = np.random.default_rng([EXTRA_STREAM, int(seed)])
    op_station = np.asarray(meta['op_station'])
    pt = np.zeros((pt_src.shape[0], len(new_mch_type)), dtype=int)
    n_shared = n_added = 0
    for row in range(pt_src.shape[0]):
        t = STATION_TYPES[op_station[row]]
        v = xrng.random(MAX_EXTRA)   # always consumed: nested added stations
        k_old, k_new = len(src_ranges[t]), len(new_ranges[t])
        assert k_new - k_old <= MAX_EXTRA
        for k, m_new in enumerate(new_ranges[t]):
            if k < k_old:
                pt[row, m_new] = pt_src[row, src_ranges[t][k]]
                n_shared += 1
            else:
                u = 1 - NOMINAL_PT_DEVIATION + 2 * NOMINAL_PT_DEVIATION * v[k - k_old]
                pt[row, m_new] = max(1, int(round(nominal[row] * u)))
                n_added += 1
    new_meta = dict(meta)
    new_meta['mch_type'] = new_mch_type
    new_meta['station_counts'] = dict(new_counts)
    return (jl, pt, new_meta), {'shared_cells': n_shared, 'added_cells': n_added}


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def sha256_of(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def write_set(name, orders, ds_meta, dry_run=False):
    """Write orders [(stem, instance)] to data/PPVC/<name>/. Never overwrites:
    an existing file must equal the order in memory."""
    out_dir = os.path.join(DATA_ROOT, name)
    meta_path = os.path.join(out_dir, 'dataset_meta.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            old = json.load(f)
        assert old['seed0'] == ds_meta['seed0'], f'{out_dir}: stale seed0 {old["seed0"]}'
    n_new = n_same = 0
    if not dry_run:
        os.makedirs(out_dir, exist_ok=True)
    for stem, inst in orders:
        path = os.path.join(out_dir, stem)
        if os.path.exists(path + '.fjs'):
            d = diff_fields(load_instance(path), inst)
            assert not d, f'{path}: existing file differs in {d}; not overwritten'
            n_same += 1
            continue
        if dry_run:
            continue
        save_instance(path, *inst)
        back = load_instance(path)
        d = diff_fields(back, inst)
        assert not d, f'{path}: round trip differs in {d}'
        n_new += 1
    if not dry_run:
        with open(meta_path, 'w') as f:
            json.dump(ds_meta, f, indent=1)
    return out_dir, n_new, n_same


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--dry_run', action='store_true',
                    help='run every check, write nothing')
    args = ap.parse_args()
    sys.argv = [sys.argv[0]]   # params.py (imported via data_utils) parses argv at import
    gen_sha = sha256_of(gen.__file__)
    script = os.path.basename(__file__)
    home = os.path.expanduser('~')
    exe = sys.executable
    exe = '~' + exe[len(home):] if exe.startswith(home + os.sep) else exe
    report = {'generator_sha256': gen_sha, 'python_executable': exe,
              'numpy': np.__version__, 'sets': {}}
    for t in STATION_TOTALS:
        print(f'stations {t}: {apportion(DEFAULT_FACTORY, t)}')
    for src_name, n_orders in SOURCES:
        src_dir = os.path.join(DATA_ROOT, src_name)
        with open(os.path.join(src_dir, 'dataset_meta.json')) as f:
            src_meta = json.load(f)
        seed0, n_mod, mix = src_meta['seed0'], src_meta['n_modules'], src_meta['ppvc_mix']
        assert src_meta.get('ppvc_factory', 'default') == 'default'
        stems = [f'instance_{i:03d}' for i in range(n_orders)]
        # 1) twins: stored file == generator == mirror; recover nominal durations
        twins, nominals, vs = [], [], []
        for i, stem in enumerate(stems):
            seed = seed0 + i
            stored = load_instance(os.path.join(src_dir, stem))
            regen = ppvc_instance_generator(n_modules=n_mod, class_mix=mix,
                                            station_counts=DEFAULT_FACTORY, seed=seed)
            mirror, draws = generator_draws(n_mod, mix, DEFAULT_FACTORY, seed,
                                            NOMINAL_PT_DEVIATION)
            assert not diff_fields(stored, regen), f'{src_name}/{stem}: stored != generator {diff_fields(stored, regen)}'
            assert not diff_fields(stored, mirror), f'{src_name}/{stem}: stored != mirror {diff_fields(stored, mirror)}'
            check_structure(stored, f'{src_name}/{stem}')
            twins.append(stored)
            nominals.append(draws['nominal'])
            vs.append(draws['v'])
        print(f'{src_name}: {n_orders} twins equal the generator and the mirror')
        base_meta = {'seed0': seed0, 'n_instances': n_orders, 'n_modules': n_mod,
                     'ppvc_mix': mix, 'source_set': src_name,
                     'source_orders': f'instance_000 to instance_{n_orders - 1:03d}',
                     'paired_with_source': True, 'route_variant_note': 'fixed routes',
                     'generator': 'ppvc_instance_generator.py',
                     'generator_sha256': gen_sha, 'written_by': script}

        # 2) wait scale
        for scale, suffix in LAG_SCALES:
            name = src_name + suffix
            orders = []
            for stem, tw in zip(stems, twins):
                inst = lag_scaled(tw, scale)
                d = diff_fields(inst, tw)
                assert d == ['time_lag'] or (scale == 1.0 and not d), f'{name}/{stem}: {d}'
                assert np.array_equal(inst[2]['time_lag'],
                                      np.rint(tw[2]['time_lag'] * scale).astype(int))
                assert np.array_equal(inst[2]['time_lag'] > 0, tw[2]['time_lag'] > 0)
                check_structure(inst, f'{name}/{stem}')
                orders.append((stem, inst))
            lags = np.concatenate([o[1][2]['time_lag'] for o in orders])
            ds = dict(base_meta, condition='wait scale', lag_scale=scale,
                      ppvc_factory='default', station_counts=dict(DEFAULT_FACTORY),
                      pt_deviation=NOMINAL_PT_DEVIATION,
                      lag_rounding='np.rint (round half to even), as train.py --lag_scale',
                      pairing='routes, stations and processing times copied from the source order; only time_lag differs',
                      purpose=f'out-of-distribution test set: {src_name} with every wait x{scale:g} (rounded)')
            out, n_new, n_same = write_set(name, orders, ds, args.dry_run)
            nz = lags[lags > 0]
            report['sets'][name] = {'orders': n_orders, 'new': n_new, 'existing_equal': n_same,
                                    'nonzero_wait_range_h': [int(nz.min()), int(nz.max())]}
            print(f'{out}: {n_new} new, {n_same} existing and equal; nonzero waits {nz.min()}-{nz.max()} h')

        # 3) station count
        for total in STATION_TOTALS:
            counts = apportion(DEFAULT_FACTORY, total)
            name = f'{n_mod}x{total}+ppvc-mixed-ood'
            orders, cells = [], {'shared_cells': 0, 'added_cells': 0}
            n_direct_paired = n_direct_same_class = 0
            for i, (stem, tw) in enumerate(zip(stems, twins)):
                seed = seed0 + i
                inst, c = station_shifted(tw, nominals[i], seed, counts)
                for k in cells:
                    cells[k] += c[k]
                d = diff_fields(inst, tw)
                assert set(d) <= {'op_pt', 'mch_type', 'station_counts'}, f'{name}/{stem}: {d}'
                # pairing: shared station columns equal the twin's
                _, r_src = build_factory(DEFAULT_FACTORY)
                _, r_new = build_factory(counts)
                for t in STATION_TYPES:
                    for k in range(min(len(r_src[t]), len(r_new[t]))):
                        assert np.array_equal(inst[1][:, r_new[t][k]], tw[1][:, r_src[t][k]])
                    for k in range(len(r_src[t]), len(r_new[t])):
                        rows = np.nonzero(np.asarray(tw[2]['op_station']) == STATION_TYPES.index(t))[0]
                        p = inst[1][rows, r_new[t][k]]
                        nom = nominals[i][rows]
                        lo = np.maximum(1, nom * (1 - NOMINAL_PT_DEVIATION) - 0.5 - 1e-9)
                        hi = np.maximum(1, nom * (1 + NOMINAL_PT_DEVIATION) + 0.5 + 1e-9)
                        assert ((p >= np.floor(lo)) & (p <= hi)).all(), f'{name}/{stem}: added station out of band'
                check_structure(inst, f'{name}/{stem}')
                # what a direct generator call would give (documents the shift)
                direct = ppvc_instance_generator(n_modules=n_mod, class_mix=mix,
                                                 station_counts=counts, seed=seed)
                n_direct_same_class += int(np.array_equal(direct[2]['routing_class'], tw[2]['routing_class']))
                n_direct_paired += int(np.array_equal(direct[2]['time_lag'], tw[2]['time_lag'])
                                       and np.array_equal(direct[2]['module_weight_t'], tw[2]['module_weight_t']))
                orders.append((stem, inst))
            ds = dict(base_meta, condition='station count', n_stations=total,
                      ppvc_factory=f'scaled-{total}', station_counts=counts,
                      station_count_rule='default counts x total/25, largest remainder, at least one per type; '
                                         'remainder ties to the larger default count, then STATION_TYPES order',
                      pt_deviation=NOMINAL_PT_DEVIATION, extra_station_stream=[EXTRA_STREAM, 'seed'],
                      extra_draws_per_operation=MAX_EXTRA,
                      pairing='classes, module weights, routes, nominal durations and waits of the source order; '
                              'the first k stations of each type keep the source processing times; added '
                              'stations draw nominal x U(0.8, 1.2) from the separate stream; removed stations '
                              'drop their column',
                      purpose=f'out-of-distribution test set: {src_name} on a {total}-station factory')
            out, n_new, n_same = write_set(name, orders, ds, args.dry_run)
            report['sets'][name] = {'orders': n_orders, 'new': n_new, 'existing_equal': n_same,
                                    'station_counts': counts, **cells,
                                    'direct_generator_same_classes': n_direct_same_class,
                                    'direct_generator_same_waits_and_weights': n_direct_paired}
            print(f'{out}: {n_new} new, {n_same} existing and equal; {counts}; '
                  f'cells shared {cells["shared_cells"]}, added {cells["added_cells"]}; '
                  f'direct generator call: classes equal on {n_direct_same_class}/{n_orders}, '
                  f'waits and weights equal on {n_direct_paired}/{n_orders}')

        # 4) duration spread
        for dev, suffix in PT_DEVIATIONS:
            name = src_name + suffix
            orders = []
            rel_dev = []
            for i, (stem, tw) in enumerate(zip(stems, twins)):
                seed = seed0 + i
                inst = generate_with_deviation(n_mod, mix, DEFAULT_FACTORY, seed, dev)
                mirror, draws = generator_draws(n_mod, mix, DEFAULT_FACTORY, seed, dev)
                assert not diff_fields(inst, mirror), f'{name}/{stem}: generator != mirror'
                d = diff_fields(inst, tw)
                assert set(d) <= {'op_pt'}, f'{name}/{stem}: {d}'
                assert np.array_equal(draws['nominal'], nominals[i]), f'{name}/{stem}: nominal durations differ'
                assert all(np.allclose(a, b, rtol=0, atol=1e-9) for a, b in zip(draws['v'], vs[i])), \
                    f'{name}/{stem}: per-station draws differ'
                assert np.array_equal(inst[1] > 0, tw[1] > 0)
                check_structure(inst, f'{name}/{stem}')
                rows, cols = np.nonzero(inst[1] > 0)
                rel_dev.append(np.abs(inst[1][rows, cols] / nominals[i][rows] - 1).mean())
                orders.append((stem, inst))
            ds = dict(base_meta, condition='duration spread', pt_deviation=dev,
                      ppvc_factory='default', station_counts=dict(DEFAULT_FACTORY),
                      pairing='same seed: classes, module weights, routes, nominal durations, waits and the '
                              'standard uniform draw of every (operation, station) equal the source order; '
                              f'processing time = max(1, round(nominal x (1 - {dev} + {2 * dev} v)))',
                      purpose=f'out-of-distribution test set: {src_name} with processing-time spread +-{dev:g}')
            out, n_new, n_same = write_set(name, orders, ds, args.dry_run)
            rel_src = []
            for i, tw in enumerate(twins):
                rows, cols = np.nonzero(tw[1] > 0)
                rel_src.append(np.abs(tw[1][rows, cols] / nominals[i][rows] - 1).mean())
            report['sets'][name] = {'orders': n_orders, 'new': n_new, 'existing_equal': n_same,
                                    'mean_abs_rel_dev_shifted': float(np.mean(rel_dev)),
                                    'mean_abs_rel_dev_source': float(np.mean(rel_src))}
            print(f'{out}: {n_new} new, {n_same} existing and equal; mean |p/nominal - 1| '
                  f'{np.mean(rel_dev):.3f} (source {np.mean(rel_src):.3f})')
    if not args.dry_run:
        os.makedirs(REPORT_DIR, exist_ok=True)
        path = os.path.join(REPORT_DIR, 'make_ood_testsets_report.json')
        if os.path.exists(path):
            with open(path) as f:
                old = json.load(f)
            assert {k: {kk: vv for kk, vv in v.items() if kk not in ('new', 'existing_equal')}
                    for k, v in old['sets'].items()} == \
                   {k: {kk: vv for kk, vv in v.items() if kk not in ('new', 'existing_equal')}
                    for k, v in report['sets'].items()}, 'ood_testsets_report.json differs'
        else:
            with open(path, 'w') as f:
                json.dump(report, f, indent=1)
    print('all checks passed')


if __name__ == '__main__':
    main()
