#!/usr/bin/env python3
"""
ANALYSIS 19 -- Target variable at the anchor: ten seeds per architecture,
three attribution methods, with a different meter as the prediction target.

Why: the extension proposed repeating the analysis with another meter as the
target. The portfolio-wide version (373 chilled-water buildings) was deferred
with supervisor agreement. This is the anchor-only version, run to the same
protocol as Analyses 5, 17 and 18, so the method ordering can be checked
under a second target.

TARGET CHOICE (decided from the data, 2026-10-13)
  Chilled water is NOT usable at the anchor under the chronological split.
  The anchor's chilled-water meter is zero for 70% of 2016 and for 99% of
  December. The test partition (2016-11-25 onward) has a mean of 0.0, so
  MAE% is undefined and the model would learn to predict zero. The script
  refuses any target whose test-partition mean is near zero.
  Gas is the default. It has no zero runs, a strong seasonal cycle and a
  non-zero test partition. Hot water is offered as an option but correlates
  at 0.98 with gas at this building and is zero June-August.

INPUT DESIGN (kept as close to the electricity runs as possible)
  Same weather and calendar features. The target meter is removed from the
  inputs, since the target's own history is never an input in this work.
  Electricity is NOT added as an input. Every other meter covariate the
  anchor carries is kept. With gas as the target the schema is 14 columns,
  11 after collapse. NOTE hot water stays an input and correlates at 0.98
  with gas, so a hot-water-led ranking is expected (prediction P3).

PROTOCOL (matches Analyses 5, 17, 18)
  Training: Analysis 5 cell 6 builders exec'd verbatim; LSTM batch 32, TFT
    batch 64 (TFT_CONFIG); Adam lr 0.001 for the LSTM; MAX_EPOCHS 300,
    PATIENCE 15, early stopping on val loss with best-weight restore; seeds
    2016-2025. Models are saved under analysis19_models/ only.
  Permutation: Analysis 5 cell 19 -- all test sequences, 5 repeats, rng
    seeded stable_seed(ANCHOR) + seed, canonical permutation_importance.
  SHAP: Analysis 17 -- GradientExplainer, background X_train[:200],
    explained X_test[:200], nsamples 200, rseed 0 (rseed 1 for a ceiling).
  IG: Analysis 18 -- baseline = training mean (1,48,F), 200 steps, batch 25,
    explained X_test[:200].
  Accuracy: test-partition MAE% (denominator = test-partition mean), plus a
    24-hour persistence baseline on the same test sequences for context.

PREDICTIONS, registered 2026-10-13 before any gas-target model is trained.
  P1  The method ordering holds under the new target on both architectures:
      median across-seed agreement for permutation importance is below that
      of both IG and SHAP, within the LSTM and within the TFT.
  P2  Changing the target costs more agreement than changing the seed:
      for each gradient method and architecture, the median agreement
      between electricity-target and gas-target rankings (same features,
      100 seed pairs) is below the median across-seed agreement within the
      gas target.
  P3  Hot water is rank-1 under SHAP in at least 8 of 10 runs per
      architecture.
  P4  Every model beats 24-hour persistence on test MAE%.
The summary prints PASS/FAIL. A FAIL is a result, not a bug. P1-P4 refer
to --target gas; for any other target they are reported but not tested.

Inputs are READ ONLY. Outputs go to NEW files only:
  analysis19_models/{target}_{arch}_seed{seed}.keras
  results/analysis19_{target}_predictive.csv        (incremental, resumable)
  results/analysis19_{target}_attribution.csv       (incremental, resumable)
  results/analysis19_{target}_summary.txt
  results/analysis19_{target}_PROVENANCE.txt

Usage (WSL, fyp_env, GPU), from the Phase2 folder:
  python a19_target_anchor.py --phase2 "$P/Phase2" --data "$P/Data"
Optional: --target gas|hotwater   --seeds 2016   --epochs 2 --steps 10
          --nsamples 20 --out_suffix _smoke   (for a quick smoke test)
Safe to stop and restart: finished models and vectors are skipped.
"""
import os, sys, json, time, hashlib, argparse, platform, itertools, gc
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--phase2', default='/home/pie/fyp/phase2')
ap.add_argument('--data', default='/home/pie/fyp/ZEIT4500 - FYP/Data')
ap.add_argument('--target', default='gas', choices=['gas', 'hotwater', 'chilledwater'])
ap.add_argument('--seeds', type=int, nargs='*', default=list(range(2016, 2026)))
ap.add_argument('--epochs', type=int, default=300)
ap.add_argument('--patience', type=int, default=15)
ap.add_argument('--steps', type=int, default=200)
ap.add_argument('--ig_batch', type=int, default=25)
ap.add_argument('--nsamples', type=int, default=200)
ap.add_argument('--n_explain', type=int, default=200)
ap.add_argument('--out_suffix', default='')
args = ap.parse_args()

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
import shap
from tensorflow.keras import layers

PHASE2 = args.phase2
RESULTS = os.path.join(PHASE2, 'results')
MODELS = os.path.join(PHASE2, 'analysis19_models')
NB = os.path.join(PHASE2, 'Analysis5_TenSeed_Confirmation.ipynb')
ANCHOR = 'Bobcat_education_Alissa'
TGT = args.target
SFX = args.out_suffix
TAG = f'analysis19_{TGT}{SFX}'
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def rho(a, b):
    ra, rb = pd.Series(a).rank().values, pd.Series(b).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])

# ---- data ------------------------------------------------------------------
sys.path.insert(0, PHASE2)
from fyp_pipeline import prepare_building, split_sequences, WINDOW_SIZE

wx_p = os.path.join(args.data, 'weather.csv')
weather_all = pd.read_csv(wx_p, parse_dates=['timestamp'])
METER_FILES = {'gas': 'gas.csv', 'hotwater': 'hotwater.csv', 'chilledwater': 'chilledwater.csv',
               'steam': 'steam.csv', 'solar': 'solar.csv', 'water': 'water.csv',
               'irrigation': 'irrigation.csv'}
meter_dfs, input_files = {}, [wx_p]
for name, fn in METER_FILES.items():
    p = os.path.join(args.data, fn)
    if os.path.exists(p):
        meter_dfs[name] = pd.read_csv(p, parse_dates=['timestamp'])
        input_files.append(p)

target_df = meter_dfs[TGT]
inputs_meters = {k: v for k, v in meter_dfs.items() if k != TGT}   # target never an input
# prepare_building names the target column 'electricity' internally; it is the TGT meter here.
X_seq, y_seq, scaler_y, feature_cols, raw_y_mean = prepare_building(
    ANCHOR, target_df, weather_all, inputs_meters)
X_train, y_train, X_val, y_val, X_test, y_test = split_sequences(
    X_seq, y_seq, train_frac=0.80, val_frac=0.10)
assert X_test.shape[0] == 874 and X_train.shape[0] == 6988, (X_train.shape, X_test.shape)
assert TGT not in feature_cols
N_F = X_train.shape[-1]
print('target', TGT, '| schema', len(feature_cols), feature_cols)

y_test_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
y_train_true = scaler_y.inverse_transform(y_train.reshape(-1, 1)).ravel()
y_test_mean = float(np.abs(y_test_true).mean())
zero_frac_test = float((y_test_true == 0).mean())
print(f'test-partition mean {y_test_mean:.3f}  train mean {y_train_true.mean():.3f}  '
      f'test zero fraction {zero_frac_test:.3f}')
if y_test_mean < 0.01 * abs(y_train_true.mean()) or zero_frac_test > 0.5:
    sys.exit(f'REFUSED: {TGT} at the anchor has a near-zero test partition '
             f'(mean {y_test_mean:.3f}, {zero_frac_test:.0%} zeros). MAE% is undefined.')

# 24-hour persistence on the same test sequences: y_seq[i] is row i+48, lag-24 is y_seq[i-24]
n_tr_va = len(X_train) + len(X_val)
lag_idx = np.arange(n_tr_va, len(y_seq)) - 24
persist = scaler_y.inverse_transform(y_seq[lag_idx].reshape(-1, 1)).ravel()
persist_mae_pct = 100.0 * np.mean(np.abs(persist - y_test_true)) / y_test_mean
print(f'24h persistence MAE% on test: {persist_mae_pct:.2f}')

# ---- builders and canonical helpers: Analysis 5 cell 6 verbatim -------------
nb = json.load(open(NB, encoding='utf-8'))
cell6 = ''.join(nb['cells'][6]['source'])
assert 'def build_tft' in cell6 and 'def permutation_importance' in cell6, 'cell 6 moved?'
ns = dict(tf=tf, np=np, layers=layers, hashlib=hashlib, WINDOW_SIZE=WINDOW_SIZE,
          N_ANCHOR_FEATURES=N_F, ANCHOR_FEATURE_SCHEMA=list(feature_cols))
exec(cell6, ns)
LABELS, collapse_vec = ns['COLLAPSED_LABELS'], ns['collapse_vec']
TFT_CONFIG = ns['TFT_CONFIG']
CUSTOM = {k: ns[k] for k in ('GRN', 'VariableSelectionNetwork', 'InterpretableMultiHeadAttention')}

# ---- training (Analysis 5 cell 9 logic) ------------------------------------
PRED_CSV = os.path.join(RESULTS, f'{TAG}_predictive.csv')
ATTR_CSV = os.path.join(RESULTS, f'{TAG}_attribution.csv')

def model_path(arch, seed):
    return os.path.join(MODELS, f'{TGT}{SFX}_{arch}_seed{seed}.keras')

def mae_pct(m):
    pred = scaler_y.inverse_transform(m.predict(X_test, verbose=0).reshape(-1, 1)).ravel()
    return 100.0 * np.mean(np.abs(pred - y_test_true)) / y_test_mean

def train_one(arch, seed):
    t0 = time.time()
    if arch == 'lstm':
        m = ns['build_lstm'](N_F, seed); bs = 32
    else:
        m = ns['build_tft'](N_F, seed, TFT_CONFIG); bs = TFT_CONFIG['batch_size']
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience,
                                          restore_best_weights=True)
    h = m.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=args.epochs,
              batch_size=bs, callbacks=[es], verbose=0)
    p = model_path(arch, seed)
    m.save(p)
    r = dict(target=TGT, arch=arch, seed=seed, mae_pct=mae_pct(m),
             persistence_mae_pct=persist_mae_pct, n_epochs_trained=len(h.history['loss']),
             elapsed_s=round(time.time() - t0, 1), weights_sha256=sha(p), status='ok')
    del m; gc.collect(); tf.keras.backend.clear_session()
    return r

def load(arch, seed):
    p = model_path(arch, seed)
    if arch == 'lstm':
        return tf.keras.models.load_model(p, compile=False)
    try:
        return tf.keras.models.load_model(p, custom_objects=CUSTOM, compile=False)
    except Exception:
        m = ns['build_tft'](N_F, seed, TFT_CONFIG); m.load_weights(p); return m

done = set()
if os.path.exists(PRED_CSV):
    prior = pd.read_csv(PRED_CSV); done = set(zip(prior.arch, prior.seed))
for arch in ['lstm', 'tft']:
    for seed in args.seeds:
        if (arch, seed) in done:
            continue
        print(f'[train {arch} {seed}] ...', flush=True)
        r = train_one(arch, seed)
        print(f'   MAE% {r["mae_pct"]:.2f} (persistence {persist_mae_pct:.2f})  '
              f'epochs {r["n_epochs_trained"]}  {r["elapsed_s"]}s', flush=True)
        pd.DataFrame([r]).to_csv(PRED_CSV, mode='a', header=not os.path.exists(PRED_CSV), index=False)

# ---- attribution: permutation (A5), SHAP (A17), IG (A18) --------------------
bg = X_train[:200].astype(np.float32)
Xe = X_test[:args.n_explain].astype(np.float32)
ig_base = X_train.mean(axis=0, keepdims=True).astype(np.float32)

def perm_vec(m, name, seed):
    infer = ns['_get_infer_fn'](name, m)
    rng = np.random.default_rng(ns['stable_seed'](ANCHOR) + seed)
    X_use, y_use = X_test, y_test_true.reshape(-1, 1)
    d = ns['permutation_importance'](infer, X_use, y_use, scaler_y, N_F, rng, n_repeats=5)
    return collapse_vec(d)

def shap_vec(m, rseed):
    e = shap.GradientExplainer(m, bg)
    sv = e.shap_values(Xe, nsamples=args.nsamples, rseed=rseed)
    sv = np.asarray(sv[0] if isinstance(sv, list) else sv).reshape(Xe.shape)
    return collapse_vec(np.abs(sv).mean(axis=(0, 1)))

def ig_vec(m, X):
    total = np.zeros(X.shape[1:], dtype=np.float64); n = 0
    for s in range(0, len(X), args.ig_batch):
        xb = X[s:s + args.ig_batch]
        bl = np.repeat(ig_base, len(xb), axis=0); diff = xb - bl
        acc = np.zeros_like(xb, dtype=np.float64)
        for k in range(1, args.steps + 1):
            pt = tf.convert_to_tensor(bl + (k / args.steps) * diff)
            with tf.GradientTape() as tape:
                tape.watch(pt); pred = m(pt, training=False)
            acc += tape.gradient(pred, pt).numpy()
        total += np.sum(np.abs(diff * (acc / args.steps)), axis=0); n += len(xb)
        del acc; gc.collect()
    return collapse_vec(np.mean(total / n, axis=0))

adone = set()
if os.path.exists(ATTR_CSV):
    pa = pd.read_csv(ATTR_CSV); adone = set(zip(pa.arch, pa.seed))
for arch in ['lstm', 'tft']:
    for seed in args.seeds:
        if (arch, seed) in adone:
            continue
        t0 = time.time()
        m = load(arch, seed)
        pv = perm_vec(m, f'{TGT}_{arch}_{seed}', seed)
        s0 = shap_vec(m, 0); s1 = shap_vec(m, 1)
        iv = ig_vec(m, Xe)
        row = dict(target=TGT, arch=arch, seed=seed, mae_pct_reloaded=mae_pct(m),
                   shap_ceiling=rho(s0, s1), elapsed_s=round(time.time() - t0, 1))
        for tag, v in (('perm', pv), ('shap', s0), ('shapr1', s1), ('ig', iv)):
            row.update({f'{tag}_{l}': x for l, x in zip(LABELS, v)})
        print(f'[attr {arch} {seed}] rank1 perm {LABELS[int(np.argmax(pv))]} | '
              f'shap {LABELS[int(np.argmax(s0))]} | ig {LABELS[int(np.argmax(iv))]}  '
              f'shap ceiling {row["shap_ceiling"]:.4f}  {row["elapsed_s"]}s', flush=True)
        pd.DataFrame([row]).to_csv(ATTR_CSV, mode='a', header=not os.path.exists(ATTR_CSV), index=False)
        del m; tf.keras.backend.clear_session(); gc.collect()

# ---- summary ----------------------------------------------------------------
P = pd.read_csv(PRED_CSV); A = pd.read_csv(ATTR_CSV)
P = P[P.seed.isin(args.seeds)]; A = A[A.seed.isin(args.seeds)]
med = lambda x: float(np.median(x)) if len(x) else np.nan
def vec(arch, seed, tag):
    r = A[(A.arch == arch) & (A.seed == seed)]
    return r[[f'{tag}_{l}' for l in LABELS]].values[0].astype(float)
def pairs(tag):
    L = [s for s in args.seeds if ((A.arch == 'lstm') & (A.seed == s)).any()]
    T = [s for s in args.seeds if ((A.arch == 'tft') & (A.seed == s)).any()]
    wl = [rho(vec('lstm', a, tag), vec('lstm', b, tag)) for a, b in itertools.combinations(L, 2)]
    wt = [rho(vec('tft', a, tag), vec('tft', b, tag)) for a, b in itertools.combinations(T, 2)]
    ac = [rho(vec('lstm', a, tag), vec('tft', b, tag)) for a in L for b in T]
    return wl, wt, ac
R = {t: pairs(t) for t in ('perm', 'ig', 'shap')}

lines = [f'ANALYSIS 19 -- target = {TGT} at the anchor, ten seeds per architecture, three methods',
         f'schema {N_F} inputs ({TGT} removed, electricity not an input); collapsed {len(LABELS)}: {LABELS}',
         f'test-partition mean {y_test_mean:.2f}; 24h persistence MAE% {persist_mae_pct:.2f}']
for arch in ['lstm', 'tft']:
    v = P[P.arch == arch].mae_pct
    lines.append(f'{arch.upper()} test MAE%: mean {v.mean():.2f} sd {v.std(ddof=1):.2f} '
                 f'min {v.min():.2f} max {v.max():.2f}  (beats persistence in {int((v < persist_mae_pct).sum())}/{len(v)})')
lines.append(f'reload check: max |MAE% reloaded - trained| = '
             f'{np.max(np.abs(A.set_index(["arch","seed"]).mae_pct_reloaded - P.set_index(["arch","seed"]).mae_pct)):.4f} pp')
lines.append(f'SHAP ceiling (rseed 0 vs 1) median: LSTM {med(A[A.arch=="lstm"].shap_ceiling):.4f}  '
             f'TFT {med(A[A.arch=="tft"].shap_ceiling):.4f}')
for t in ('perm', 'ig', 'shap'):
    rk = {a: pd.Series([LABELS[int(np.argmax(vec(a, s, t)))] for s in A[A.arch == a].seed]).value_counts().to_dict()
          for a in ['lstm', 'tft']}
    wl, wt, ac = R[t]
    lines.append(f'{t.upper():5s} within-LSTM {med(wl):.4f}  within-TFT {med(wt):.4f}  across {med(ac):.4f}   rank-1 {rk}')
for arch in ['lstm', 'tft']:
    sa = A[A.arch == arch].seed
    lines.append(f'{arch.upper()} same model: IG~SHAP {med([rho(vec(arch,s,"ig"),vec(arch,s,"shap")) for s in sa]):.4f}  '
                 f'IG~perm {med([rho(vec(arch,s,"ig"),vec(arch,s,"perm")) for s in sa]):.4f}  '
                 f'SHAP~perm {med([rho(vec(arch,s,"shap"),vec(arch,s,"perm")) for s in sa]):.4f}')

# cross-target: electricity-target vectors from Analyses 5/17/18 on the shared collapsed features
src = {'perm': ('analysis5_tenseed_attribution.csv', lambda a: f'{a}_'),
       'shap': ('analysis17_shap_tenseed.csv', lambda a: 'shap_'),
       'ig': ('analysis18_ig_tenseed.csv', lambda a: 'ig_')}
xt = {}
for t, (fn, pref) in src.items():
    p = os.path.join(RESULTS, fn)
    if not os.path.exists(p):
        lines.append(f'cross-target {t}: {fn} missing, skipped'); continue
    E = pd.read_csv(p)
    for arch in ['lstm', 'tft']:
        vals = []
        for se in range(2016, 2026):
            er = E[(E.arch == arch) & (E.seed == se)]
            if not len(er): continue
            ev = np.array([er[f'{pref(arch)}{l}'].values[0] for l in LABELS], dtype=float)
            for sg in A[A.arch == arch].seed:
                vals.append(rho(ev, vec(arch, sg, t)))
        xt[(t, arch)] = med(vals)
    lines.append(f'cross-target {t.upper():5s} (electricity vs {TGT}, same features): '
                 f'LSTM {xt.get((t,"lstm"), np.nan):.4f}  TFT {xt.get((t,"tft"), np.nan):.4f}')

full = len(args.seeds) == 10 and TGT == 'gas' and SFX == ''
pf = lambda b: 'PASS' if b else 'FAIL'
p1 = all(med(R['perm'][i]) < min(med(R['ig'][i]), med(R['shap'][i])) for i in (0, 1))
p2 = all(xt.get((t, a), 1) < med(R[t][i]) for t in ('ig', 'shap') for i, a in ((0, 'lstm'), (1, 'tft')))
p3 = all(sum(LABELS[int(np.argmax(vec(a, s, 'shap')))] == 'hotwater' for s in A[A.arch == a].seed) >= 8
         for a in ['lstm', 'tft'])
p4 = bool((P.mae_pct < persist_mae_pct).all())
lines += ['', 'PREDICTIONS (registered 2026-10-13, before the run)' +
          ('' if full else '  [NOT the reported test: subset, smoke or non-gas run]'),
          f'  P1 method ordering holds (perm below IG and SHAP, both archs) : {pf(p1)}',
          f'  P2 target change costs more than seed change (IG, SHAP)       : {pf(p2)}',
          f'  P3 hot water rank-1 under SHAP in >= 8/10 per arch            : {pf(p3)}',
          f'  P4 every model beats 24h persistence                          : {pf(p4)}']
open(os.path.join(RESULTS, f'{TAG}_summary.txt'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))

with open(os.path.join(RESULTS, f'{TAG}_PROVENANCE.txt'), 'w') as f:
    f.write(f'run (UTC) : {pd.Timestamp.now(tz="UTC").isoformat()}\n')
    f.write(f'script    : {os.path.basename(__file__)} sha256 {sha(__file__)}\n')
    f.write(f'args      : {vars(args)}\n')
    f.write(f'python {platform.python_version()} tensorflow {tf.__version__} shap {shap.__version__} '
            f'numpy {np.__version__} pandas {pd.__version__}  GPUs {len(gpus)}\n')
    f.write(f'notebook  : {os.path.basename(NB)} sha256 {sha(NB)} (cell 6 exec\'d)\n')
    f.write(f'pipeline  : fyp_pipeline.py sha256 {sha(os.path.join(PHASE2, "fyp_pipeline.py"))}\n')
    for p in input_files:
        f.write(f'input     : {os.path.basename(p)} sha256 {sha(p)}\n')
    for _, r in P.iterrows():
        f.write(f'model     : {TGT}{SFX}_{r.arch}_seed{r.seed}.keras sha256 {r.weights_sha256}\n')
print('done')
