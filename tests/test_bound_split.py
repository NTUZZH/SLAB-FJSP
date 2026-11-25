"""
test_bound_split.py
===================
The completion-time lower bound of the fixed-route environment enters the
reward (array op_ct_lb_rw) and the operation features (array op_ct_lb).
--bound_lag_free switches both to the lag-free bound; --bound_lag_free_reward
and --bound_lag_free_features switch one each. CPU only.

  a. All flags off: every state tensor, the raw feature arrays, the rewards and
     the bound are bit-identical to the environment of commit 07923569 (the
     last commit before the split), on 10-module orders under fixed random
     action sequences, batched and single, before and after env.reset(); the
     two bound arrays are equal at every step.
  b. --bound_lag_free True is bit-identical to (reward True, features True),
     and both equal the environment of 07923569 with bound_lag_free=True.
  c. Reward only: the features of the default run and the rewards of the
     lag-free run. Features only: the converse. Same action sequences; the
     dynamics and every non-bound state tensor are unchanged in all variants.
  d. Telescoping, in every variant: the sum of the rewards times the time
     scale equals the initial reward bound times the scale minus the makespan.
  e. The flags parse (default False); train.py passes them to the training
     and validation environments; eval_ppvc.py applies them from a snapshot.

Run (params.py parses sys.argv at import):
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c <core> \\
      python -c "import sys,pytest; sys.argv=['pytest']; sys.exit(pytest.main(['-q','tests/test_bound_split.py']))"
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import functools
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import types

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
PY = sys.executable
CHILD_ENV = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                 CUDA_VISIBLE_DEVICES='')
BASE_COMMIT = '07923569f32a094c36951503b521fd726d43a00e'

import fjsp_env_same_op_nums as NEW  # noqa: E402
from ppvc_instance_generator import (load_instance, ppvc_instance_generator,  # noqa: E402
                                     DEFAULT_FACTORY)

STATE_FIELDS = ('fea_j_tensor', 'op_mask_tensor', 'fea_m_tensor', 'mch_mask_tensor',
                'dynamic_pair_mask_tensor', 'comp_idx_tensor', 'candidate_tensor',
                'fea_pairs_tensor', 'op_type_tensor', 'mch_type_tensor')
FEATURE_FIELDS = ('fea_j_tensor',)   # the only state tensor that carries the bound

VARIANTS = {   # name: (module tag, env kwargs)
    'base_default': ('base', {}),
    'base_lagfree': ('base', {'bound_lag_free': True}),
    'default': ('new', {}),
    'default_explicit': ('new', {'bound_lag_free': False, 'bound_lag_free_reward': False,
                                 'bound_lag_free_features': False}),
    'lagfree': ('new', {'bound_lag_free': True}),
    'both': ('new', {'bound_lag_free_reward': True, 'bound_lag_free_features': True}),
    'reward_only': ('new', {'bound_lag_free_reward': True}),
    'features_only': ('new', {'bound_lag_free_features': True}),
}
NEW_VARIANTS = [k for k, (m, _) in VARIANTS.items() if m == 'new']


@functools.lru_cache(maxsize=1)
def base_module():
    """The environment module of BASE_COMMIT, loaded from git."""
    try:
        src = subprocess.run(['git', 'show', f'{BASE_COMMIT}:fjsp_env_same_op_nums.py'],
                             cwd=REPO, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        pytest.skip(f'pre-split environment not available from git ({e})')
    mod = types.ModuleType('fjsp_env_same_op_nums_base')
    mod.__file__ = f'fjsp_env_same_op_nums.py@{BASE_COMMIT[:8]}'
    exec(compile(src, mod.__file__, 'exec'), mod.__dict__)
    return mod


def _hash(x):
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    x = np.ascontiguousarray(np.asarray(x))
    m = hashlib.sha256()
    m.update(str(x.dtype).encode())
    m.update(str(x.shape).encode())
    m.update(x.tobytes())
    return m.hexdigest()


def _state_hashes(state):
    return {f: (None if getattr(state, f) is None else _hash(getattr(state, f)))
            for f in STATE_FIELDS}


def _play(env, state, rng=None, actions=None):
    """Roll to the end with random legal actions (rng) or a given sequence.
    Legal pairs are read from the state tensor, as the policy reads them."""
    rec = {'a': [], 'r': [], 'lb': [np.array(env.op_ct_lb)],
           'lb_rw': [np.array(getattr(env, 'op_ct_lb_rw', env.op_ct_lb))],
           'state': [_state_hashes(state)], 'raw': [],
           'init_quality': np.array(env.init_quality),
           'scale': float(env.pt_upper_bound - env.pt_lower_bound + 1e-8)}
    E = env.number_of_envs
    t = 0
    while True:
        legal = ~state.dynamic_pair_mask_tensor.cpu().numpy().reshape(E, -1)
        if actions is None:
            acts = np.array([int(rng.choice(np.where(legal[e])[0])) for e in range(E)])
        else:
            acts = np.asarray(actions[t])
            assert legal[np.arange(E), acts].all(), 'replayed action is not legal'
        state, r, done = env.step(acts)
        t += 1
        rec['a'].append(acts)
        rec['r'].append(np.array(r))
        rec['lb'].append(np.array(env.op_ct_lb))
        rec['lb_rw'].append(np.array(getattr(env, 'op_ct_lb_rw', env.op_ct_lb)))
        rec['state'].append(_state_hashes(state))
        rec['raw'].append({'fea_j': _hash(env.fea_j), 'fea_m': _hash(env.fea_m),
                           'fea_pairs': _hash(env.fea_pairs),
                           'max_endTime': _hash(env.max_endTime)})
        if done.all():
            break
    rec['makespan'] = np.array(env.current_makespan)
    rec['true_op_ct'] = np.array(env.true_op_ct)
    return rec


def _batch_testset(idx):
    out = ([], [], [], [], [])
    for i in idx:
        jl, pt, meta = load_instance(os.path.join(REPO, 'data', 'PPVC', '10x25+ppvc-mixed',
                                                  f'instance_{i:03d}'))
        for lst, v in zip(out, (jl, pt, np.asarray(meta['time_lag'], dtype=float),
                                meta['op_type'], meta['mch_type'])):
            lst.append(v)
    return out


def _batch_generated(seeds, lag_scale):
    """Training-style orders: generator draws, waits scaled and rounded as train.py."""
    out = ([], [], [], [], [])
    for s in seeds:
        jl, pt, meta = ppvc_instance_generator(n_modules=10, class_mix='mixed',
                                               station_counts=DEFAULT_FACTORY, seed=s)
        lag = np.rint(np.asarray(meta['time_lag'], dtype=float) * lag_scale)
        for lst, v in zip(out, (jl, pt, lag, meta['op_type'], meta['mch_type'])):
            lst.append(v)
    return out


BATCHES = {
    'test000-003': lambda: _batch_testset([0, 1, 2, 3]),          # batch of 4, true waits
    'test004': lambda: _batch_testset([4]),                        # batch of 1 (evaluation)
    'gen-halfwait': lambda: _batch_generated([11, 12, 13], 0.5),   # training-style
}
# (use_lag_features, type indices passed): the A1/SLAB recipe and the A3 recipe
RECIPES = {'A1': (False, False), 'A3': (True, True)}
CASES = [(b, r) for b in BATCHES for r in RECIPES]


@functools.lru_cache(maxsize=None)
def run_case(batch, recipe):
    """All variants of one case on one action sequence (drawn in the default
    run of the new module, replayed everywhere else), before and after reset."""
    jls, pts, lags, ots, mts = BATCHES[batch]()
    use_lag, with_types = RECIPES[recipe]
    J, M = jls[0].shape[0], pts[0].shape[1]
    kw_data = dict(time_lag_list=lags)
    if with_types:
        kw_data.update(op_type_list=ots, mch_type_list=mts)
    seqs = None
    out = {}
    for name in ['default'] + [v for v in VARIANTS if v != 'default']:
        tag, kw = VARIANTS[name]
        mod = NEW if tag == 'new' else base_module()
        env = mod.FJSPEnvForSameOpNums(J, M, use_lag_features=use_lag, **kw)
        state = env.set_initial_data(jls, pts, **kw_data)
        if seqs is None:
            first = _play(env, state, rng=np.random.RandomState(1234))
            state = env.reset()
            second = _play(env, state, rng=np.random.RandomState(99))
            seqs = (first['a'], second['a'])
        else:
            first = _play(env, state, actions=seqs[0])
            state = env.reset()
            second = _play(env, state, actions=seqs[1])
        out[name] = (first, second, env)
    return out


def _same_rewards(x, y):
    return all(a.dtype == b.dtype and np.array_equal(a, b) for a, b in zip(x['r'], y['r'])) \
        and len(x['r']) == len(y['r'])


def _same_arrays(x, y, key):
    return len(x[key]) == len(y[key]) and all(
        a.dtype == b.dtype and np.array_equal(a, b) for a, b in zip(x[key], y[key]))


def _same_state(x, y, fields):
    return len(x['state']) == len(y['state']) and all(
        sx[f] == sy[f] for sx, sy in zip(x['state'], y['state']) for f in fields)


def _same_dynamics(x, y):
    return (np.array_equal(x['makespan'], y['makespan'])
            and np.array_equal(x['true_op_ct'], y['true_op_ct'])
            and all(np.array_equal(a, b) for a, b in zip(x['a'], y['a'])))


NON_BOUND_FIELDS = tuple(f for f in STATE_FIELDS if f not in FEATURE_FIELDS)


# ---------------------------------------------------------------------------
# a. all flags off == the pre-split environment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('batch,recipe', CASES)
def test_a_default_bit_identical_to_pre_split(batch, recipe):
    res = run_case(batch, recipe)
    for name in ('default', 'default_explicit'):
        for k in (0, 1):   # before / after reset
            x, ref = res[name][k], res['base_default'][k]
            assert _same_rewards(x, ref), f'{name}: rewards differ'
            assert _same_arrays(x, ref, 'lb'), f'{name}: bound differs'
            assert _same_state(x, ref, STATE_FIELDS), f'{name}: a state tensor differs'
            assert x['raw'] == ref['raw'], f'{name}: a raw feature array differs'
            assert np.array_equal(x['init_quality'], ref['init_quality'])
            assert _same_dynamics(x, ref)
            # the two bound arrays are equal at every step
            assert _same_arrays({'v': x['lb']}, {'v': x['lb_rw']}, 'v'), \
                f'{name}: reward and feature bound arrays differ'
        env = res[name][2]
        assert env.lf_reward is False and env.lf_feat is False


# ---------------------------------------------------------------------------
# b. --bound_lag_free == both split flags == the pre-split lag-free env
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('batch,recipe', CASES)
def test_b_lag_free_equals_both_flags(batch, recipe):
    res = run_case(batch, recipe)
    for k in (0, 1):
        x, y, ref = res['lagfree'][k], res['both'][k], res['base_lagfree'][k]
        for other, label in ((y, 'both flags'), (ref, 'pre-split lag-free env')):
            assert _same_rewards(x, other), f'rewards differ from {label}'
            assert _same_arrays(x, other, 'lb'), f'feature bound differs from {label}'
            assert _same_state(x, other, STATE_FIELDS), f'state differs from {label}'
            assert x['raw'] == other['raw'], f'raw features differ from {label}'
            assert np.array_equal(x['init_quality'], other['init_quality'])
            assert _same_dynamics(x, other)
        assert _same_arrays(x, y, 'lb_rw')
        assert _same_arrays({'v': x['lb']}, {'v': x['lb_rw']}, 'v')
    for name in ('lagfree', 'both'):
        env = res[name][2]
        assert env.lf_reward is True and env.lf_feat is True


# ---------------------------------------------------------------------------
# c. one bound at a time
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('batch,recipe', CASES)
def test_c_reward_only_and_features_only(batch, recipe):
    res = run_case(batch, recipe)
    for k in (0, 1):
        d, f = res['default'][k], res['lagfree'][k]
        ro, fo = res['reward_only'][k], res['features_only'][k]
        # reward only: features of the default run, rewards of the lag-free run
        assert _same_state(ro, d, FEATURE_FIELDS) and _same_arrays(ro, d, 'lb')
        assert [r['fea_j'] for r in ro['raw']] == [r['fea_j'] for r in d['raw']]
        assert _same_rewards(ro, f) and _same_arrays(ro, f, 'lb_rw')
        assert np.array_equal(ro['init_quality'], f['init_quality'])
        # features only: features of the lag-free run, rewards of the default run
        assert _same_state(fo, f, FEATURE_FIELDS) and _same_arrays(fo, f, 'lb')
        assert [r['fea_j'] for r in fo['raw']] == [r['fea_j'] for r in f['raw']]
        assert _same_rewards(fo, d) and _same_arrays(fo, d, 'lb_rw')
        assert np.array_equal(fo['init_quality'], d['init_quality'])
        # everything that does not carry the bound is unchanged in every variant
        for name in NEW_VARIANTS:
            x = res[name][k]
            assert _same_state(x, d, NON_BOUND_FIELDS), f'{name}: non-bound state differs'
            assert [(r['fea_m'], r['fea_pairs']) for r in x['raw']] == \
                [(r['fea_m'], r['fea_pairs']) for r in d['raw']]
            assert _same_dynamics(x, d), f'{name}: schedule differs'
        # the split is not vacuous: these orders have waits, so the bounds differ
        assert not _same_rewards(d, f) and not _same_state(d, f, FEATURE_FIELDS)
        assert (f['init_quality'] <= d['init_quality']).all() \
            and (f['init_quality'] < d['init_quality']).any()
    ro_env, fo_env = res['reward_only'][2], res['features_only'][2]
    assert (ro_env.lf_reward, ro_env.lf_feat) == (True, False)
    assert (fo_env.lf_reward, fo_env.lf_feat) == (False, True)


# ---------------------------------------------------------------------------
# d. telescoping return
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('batch,recipe', CASES)
def test_d_telescoping(batch, recipe):
    res = run_case(batch, recipe)
    for name in NEW_VARIANTS:
        for k in (0, 1):
            x = res[name][k]
            ret = np.sum(np.stack(x['r']), axis=0) * x['scale']
            expect = x['init_quality'] * x['scale'] - x['makespan']
            assert np.allclose(ret, expect, rtol=0, atol=1e-6), \
                f'{name}: return {ret} != initial bound - makespan {expect}'
            # at the end both bound arrays hold the realized completion times
            assert np.allclose(x['lb_rw'][-1].max(axis=1) * x['scale'], x['makespan'],
                               rtol=0, atol=1e-6)
            assert np.allclose(x['lb'][-1].max(axis=1) * x['scale'], x['makespan'],
                               rtol=0, atol=1e-6)
            # the reward bound never exceeds the makespan it bounds
            assert (x['init_quality'] * x['scale'] <= x['makespan'] + 1e-6).all()


# ---------------------------------------------------------------------------
# e. flags, training and evaluation wiring
# ---------------------------------------------------------------------------

def test_e_flags_parse():
    from params import parser
    d = parser.parse_args([])
    assert d.bound_lag_free is False and d.bound_lag_free_reward is False \
        and d.bound_lag_free_features is False
    a = parser.parse_args(['--bound_lag_free_reward', 'True',
                           '--bound_lag_free_features', 'false'])
    assert a.bound_lag_free_reward is True and a.bound_lag_free_features is False
    with pytest.raises(SystemExit):
        parser.parse_args(['--bound_lag_free_features', 'maybe'])


TRAIN_FLAGS = ['--data_source', 'PPVC', '--device', 'cpu', '--ppvc_factory', 'default',
               '--ppvc_mix', 'mixed', '--max_updates', '2000', '--vali_size', '4',
               '--ppvc_vali_sizes', '10,20', '--n_j', '10', '--use_lag_features', 'False',
               '--use_type_embedding', 'False', '--seed_train', '301',
               '--model_suffix', 'boundsplit-test']

TRAINER_PROBE = r'''
import json, sys
sys.argv = ['train.py'] + json.loads(sys.argv[1])
import train as T
orig = T.Trainer.make_ppvc_dataset
# validation sets in memory only, as preflight_train.py does
T.Trainer.make_ppvc_dataset = lambda self, n, seed0, save_dir=None, n_modules=None: \
    orig(self, n, seed0, save_dir=None, n_modules=n_modules)
t = T.Trainer(T.configs)
envs = [t.env] + [e for _, e in getattr(t, 'vali_envs', [])]
print(json.dumps([[e.lf_reward, e.lf_feat] for e in envs]))
'''


@pytest.mark.parametrize('extra,expect', [
    ([], [False, False]),
    (['--bound_lag_free_reward', 'True'], [True, False]),
    (['--bound_lag_free_features', 'True'], [False, True]),
    (['--bound_lag_free', 'True'], [True, True]),
])
def test_e_train_passes_flags(extra, expect):
    out = subprocess.run([PY, '-B', '-c', TRAINER_PROBE, json.dumps(TRAIN_FLAGS + extra)],
                         cwd=REPO, env=CHILD_ENV, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, (out.stdout + out.stderr)[-2000:]
    flags = json.loads(out.stdout.strip().splitlines()[-1])
    assert len(flags) == 3, flags          # training env + validation envs at 10 and 20
    assert all(f == expect for f in flags), flags


def test_e_train_rejects_variable_routes():
    out = subprocess.run([PY, '-B', '-c', TRAINER_PROBE,
                          json.dumps(TRAIN_FLAGS + ['--ppvc_route_variant', 'variable',
                                                    '--bound_lag_free_features', 'True'])],
                         cwd=REPO, env=CHILD_ENV, capture_output=True, text=True, timeout=600)
    assert out.returncode != 0 and 'not wired' in out.stderr


EVAL_PROBE = r'''
import json, os, sys
sys.argv = [sys.argv[0]]
repo, tmp = os.environ['REPO'], os.environ['TMPD']
sys.path.insert(0, repo)
import eval_ppvc as ev
for k in ('bound_lag_free', 'bound_lag_free_reward', 'bound_lag_free_features'):
    assert k in ev.ARCH_KEYS, k
assert ev._model_bound_kwargs() == {'bound_lag_free': False, 'bound_lag_free_reward': False,
                                    'bound_lag_free_features': False}
e = ev._make_env(10, 25, False)
assert (e.lf_reward, e.lf_feat) == (False, False)
e = ev._make_env(10, 25, False, True)
assert e.bound_lag_free is True and (e.lf_reward, e.lf_feat) == (True, True)
e = ev._make_env(10, 25, False, bound_lag_free_features=True)
assert (e.lf_reward, e.lf_feat) == (False, True)
# a snapshot with the features-only flag (read from a copy in a temp dir)
os.chdir(tmp)
snap = ev.load_train_config('10x25+ppvc-mixed+a1lffeat-2k-s301')
assert snap['bound_lag_free_features'] is True
kw = ev._model_bound_kwargs()
assert kw == {'bound_lag_free': False, 'bound_lag_free_reward': False,
              'bound_lag_free_features': True}, kw
assert ev._model_bound_lag_free() is False
from ppvc_instance_generator import load_instance
jl, pt, meta = load_instance(os.path.join(repo, 'data', 'PPVC', '10x25+ppvc-mixed', 'instance_000'))
env, _ = ev._new_env(jl, pt, meta, False, False, **kw)
assert (env.lf_reward, env.lf_feat) == (False, True)
print('eval_ppvc ok')
'''


def test_e_eval_applies_flags_from_snapshot():
    src = os.path.join(REPO, 'train_log', 'PPVC', 'config_10x25+ppvc-mixed+a1-2k-s301.json')
    if not os.path.exists(src):
        pytest.skip('comparator snapshot absent')
    with open(src) as f:
        snap = json.load(f)
    snap.update(model_suffix='a1lffeat-2k-s301', bound_lag_free_features=True,
                bound_lag_free_reward=False)
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, 'train_log', 'PPVC'))
        with open(os.path.join(tmp, 'train_log', 'PPVC',
                               'config_10x25+ppvc-mixed+a1lffeat-2k-s301.json'), 'w') as f:
            json.dump(snap, f)
        env = dict(CHILD_ENV, REPO=REPO, TMPD=tmp)
        out = subprocess.run([PY, '-B', '-c', EVAL_PROBE], cwd=REPO, env=env,
                             capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, (out.stdout + out.stderr)[-2000:]
    assert 'eval_ppvc ok' in out.stdout
