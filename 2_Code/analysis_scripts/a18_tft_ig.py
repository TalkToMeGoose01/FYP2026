#!/usr/bin/env python3
"""
ANALYSIS 18 -- Integrated Gradients on both architectures, on the ten-seed
anchor models of Analysis 5.

Why: Analysis 17 gave the TFT permutation importance and SHAP at the anchor.
The thesis's method ordering (Table tab:methodeffect: permutation far less
reproducible than either gradient method, the two gradient methods close) is
measured on the LSTM only. IG on the same twenty models completes the three
methods on the second architecture. It trains nothing.

Protocol (matches the retraining rung, a6_sameseed_rung.py, and the portfolio
IG of FYP_Phase2_TFT_FINAL.py):
  anchor Bobcat_education_Alissa, 874-sequence test partition,
  IG_STEPS = 200 Riemann steps, batches of 25 sequences,
  baseline = mean of the anchor training partition, shape (1, 48, 15),
  mean |IG| over samples and timesteps, sin/cos collapse to 12 features.
  Primary set A = X_test[:200], the same sequences Analysis 17 explained, so
  IG pairs directly with the Analysis 17 SHAP vectors.

Ceiling. IG has no random state, so two runs on set A agree by construction
and are not a ceiling. Following the thesis convention (two independent
subsamples), the ceiling is IG on set A against IG on set B = X_test[200:400],
a disjoint set. This is a SAMPLE ceiling, not the explainer-random-state
ceiling of Analysis 17, and must be described that way. A determinism check
(set A twice) is run on the first model of each architecture only, to confirm
GPU arithmetic does not move the ranking.

PREDICTIONS, registered 2026-10-12 before any IG was run on these models:
  P1  Median IG agreement within the TFT across seeds exceeds 0.85.
  P2  Median IG-vs-SHAP agreement on the same TFT exceeds the SHAP-vs-
      permutation figure on the same TFT (0.671).
  P3  Changing method costs more than changing architecture under IG. Tested
      as: median IG-vs-permutation agreement on the same model is below the
      median IG agreement across architectures, for both architectures.
  P4  Hour of Day is rank-1 under IG in at least 9 of 10 runs per architecture.
The summary prints PASS/FAIL against each. A FAIL is a result, not a bug.

Inputs are READ ONLY. Outputs go to NEW files only:
  results/analysis18_ig_tenseed.csv
  results/analysis18_summary.txt
  results/analysis18_PROVENANCE.txt

Usage (WSL, fyp_env, GPU):
  python a18_tft_ig.py \
      --phase2 /home/pie/fyp/phase2 \
      --data "/home/pie/fyp/ZEIT4500 - FYP/Data"
Optional: --seeds 2016 2017   (subset, for a quick check)
          --steps 200         (Riemann steps; keep at 200 for the reported run)
"""
import os, sys, json, time, hashlib, argparse, platform, itertools, gc
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--phase2', default='/home/pie/fyp/phase2')
ap.add_argument('--data', default='/home/pie/fyp/ZEIT4500 - FYP/Data')
ap.add_argument('--seeds', type=int, nargs='*', default=list(range(2016, 2026)))
ap.add_argument('--steps', type=int, default=200)
ap.add_argument('--batch', type=int, default=25)
ap.add_argument('--n_explain', type=int, default=200)
ap.add_argument('--out_suffix', default='')
args = ap.parse_args()

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
from tensorflow.keras import layers

PHASE2 = args.phase2
RESULTS = os.path.join(PHASE2, 'results')
MODELS = os.path.join(PHASE2, 'analysis5_models')
NB = os.path.join(PHASE2, 'Analysis5_TenSeed_Confirmation.ipynb')
ANCHOR = 'Bobcat_education_Alissa'
SFX = args.out_suffix
os.makedirs(RESULTS, exist_ok=True)

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

# ---- pipeline + data (identical calls to Analysis 5 cell 4 and Analysis 17) --
sys.path.insert(0, PHASE2)
from fyp_pipeline import prepare_building, split_sequences, WINDOW_SIZE

elec_p = os.path.join(args.data, 'electricity.csv')
wx_p = os.path.join(args.data, 'weather.csv')
electricity = pd.read_csv(elec_p, parse_dates=['timestamp'])
weather_all = pd.read_csv(wx_p, parse_dates=['timestamp'])
METER_FILES = {'gas': 'gas.csv', 'hotwater': 'hotwater.csv', 'chilledwater': 'chilledwater.csv',
               'steam': 'steam.csv', 'solar': 'solar.csv', 'water': 'water.csv',
               'irrigation': 'irrigation.csv'}
meter_dfs, input_files = {}, [elec_p, wx_p]
for name, fn in METER_FILES.items():
    p = os.path.join(args.data, fn)
    if os.path.exists(p):
        meter_dfs[name] = pd.read_csv(p, parse_dates=['timestamp'])
        input_files.append(p)

X_seq, y_seq, scaler_y, feature_cols, raw_y_mean = prepare_building(
    ANCHOR, electricity, weather_all, meter_dfs)
X_train, y_train, X_val, y_val, X_test, y_test = split_sequences(
    X_seq, y_seq, train_frac=0.80, val_frac=0.10)
assert X_test.shape[0] == 874 and X_train.shape[0] == 6988, (X_train.shape, X_test.shape)
assert X_test.shape[0] >= 2 * args.n_explain, 'set B needs a disjoint second block'
N_ANCHOR_FEATURES = X_train.shape[-1]
ANCHOR_FEATURE_SCHEMA = list(feature_cols)
print('schema', ANCHOR_FEATURE_SCHEMA)

# ---- model builders: exec Analysis 5 cell 6 verbatim ------------------------
nb = json.load(open(NB, encoding='utf-8'))
cell6 = ''.join(nb['cells'][6]['source'])
assert 'def build_tft' in cell6 and 'build_collapse_index' in cell6, 'cell 6 moved?'
ns = dict(tf=tf, np=np, layers=layers, hashlib=hashlib, WINDOW_SIZE=WINDOW_SIZE,
          N_ANCHOR_FEATURES=N_ANCHOR_FEATURES, ANCHOR_FEATURE_SCHEMA=ANCHOR_FEATURE_SCHEMA)
exec(cell6, ns)
COLLAPSED_LABELS, collapse_vec = ns['COLLAPSED_LABELS'], ns['collapse_vec']
TFT_CONFIG = ns['TFT_CONFIG']
CUSTOM = {k: ns[k] for k in ('GRN', 'VariableSelectionNetwork', 'InterpretableMultiHeadAttention')}

def load(arch, seed):
    p = os.path.join(MODELS, f'{arch}_seed{seed}.keras')
    how = 'load_model'
    if arch == 'lstm':
        m = tf.keras.models.load_model(p, compile=False)
    else:
        try:
            m = tf.keras.models.load_model(p, custom_objects=CUSTOM, compile=False)
        except Exception:
            m = ns['build_tft'](N_ANCHOR_FEATURES, seed, TFT_CONFIG)
            m.load_weights(p)
            how = 'rebuild+load_weights'
    return m, p, how

y_test_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
def mae_pct(m):
    pred = scaler_y.inverse_transform(m.predict(X_test, verbose=0).reshape(-1, 1)).ravel()
    return 100.0 * np.mean(np.abs(pred - y_test_true)) / np.mean(y_test_true)

pred_csv = os.path.join(RESULTS, 'analysis5_tenseed_predictive.csv')
recorded = pd.read_csv(pred_csv).set_index(['arch', 'seed'])['mae_pct'].to_dict() \
    if os.path.exists(pred_csv) else {}
GATE_TOL_PP = 0.05          # same gate as Analysis 17; 20/20 matched exactly on the GPU

baseline = X_train.mean(axis=0, keepdims=True).astype(np.float32)     # (1, 48, 15)
XA = X_test[:args.n_explain].astype(np.float32)                        # = Analysis 17 set
XB = X_test[args.n_explain:2 * args.n_explain].astype(np.float32)      # disjoint ceiling set

def ig_vec(m, X):
    """mean |IG| over samples and timesteps, collapsed to 12. Same arithmetic as
    integrated_gradients() in a6_sameseed_rung.py."""
    total = np.zeros(X.shape[1:], dtype=np.float64)
    n = 0
    for s in range(0, len(X), args.batch):
        xb = X[s:s + args.batch]
        bl = np.repeat(baseline, len(xb), axis=0)
        diff = xb - bl
        acc = np.zeros_like(xb, dtype=np.float64)
        for k in range(1, args.steps + 1):
            pt = tf.convert_to_tensor(bl + (k / args.steps) * diff)
            with tf.GradientTape() as tape:
                tape.watch(pt)
                pred = m(pt, training=False)
            acc += tape.gradient(pred, pt).numpy()
        total += np.sum(np.abs(diff * (acc / args.steps)), axis=0)
        n += len(xb)
        del acc
        gc.collect()
    return collapse_vec(np.mean(total / n, axis=0))

def rho(a, b):
    ra, rb = pd.Series(a).rank().values, pd.Series(b).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])

# ---- comparison vectors already on record (read only) -----------------------
shap_csv = os.path.join(RESULTS, 'analysis17_shap_tenseed.csv')
perm_csv = os.path.join(RESULTS, 'analysis5_tenseed_attribution.csv')
SH = pd.read_csv(shap_csv)
PM = pd.read_csv(perm_csv)
def shap_of(arch, seed):
    r = SH[(SH.arch == arch) & (SH.seed == seed)]
    return r[[f'shap_{l}' for l in COLLAPSED_LABELS]].values[0].astype(float) if len(r) else None
def perm_of(arch, seed):
    r = PM[(PM.arch == arch) & (PM.seed == seed)]
    return r[[f'{arch}_{l}' for l in COLLAPSED_LABELS]].values[0].astype(float) if len(r) else None

rows = []
for arch in ['lstm', 'tft']:
    for i, seed in enumerate(args.seeds):
        t0 = time.time()
        m, p, how = load(arch, seed)
        mp = mae_pct(m)
        rec = recorded.get((arch, seed), np.nan)
        gate = bool(np.isfinite(rec) and abs(mp - rec) < GATE_TOL_PP)
        vA = ig_vec(m, XA)
        vB = ig_vec(m, XB)
        det = rho(vA, ig_vec(m, XA)) if i == 0 else np.nan
        sv, pv = shap_of(arch, seed), perm_of(arch, seed)
        row = dict(arch=arch, seed=seed, load=how, mae_pct=mp, mae_pct_recorded=rec,
                   gate_mae_matches=gate, ceiling_rho_setA_vs_setB=rho(vA, vB),
                   determinism_rho_setA_twice=det,
                   rank1=COLLAPSED_LABELS[int(np.argmax(vA))],
                   rho_ig_vs_shap=rho(vA, sv) if sv is not None else np.nan,
                   rho_ig_vs_perm=rho(vA, pv) if pv is not None else np.nan,
                   rho_shap_vs_perm=rho(sv, pv) if (sv is not None and pv is not None) else np.nan,
                   elapsed_s=round(time.time() - t0, 1), model_sha256=sha(p))
        row.update({f'ig_{l}': x for l, x in zip(COLLAPSED_LABELS, vA)})
        row.update({f'igB_{l}': x for l, x in zip(COLLAPSED_LABELS, vB)})
        rows.append(row)
        print(f'[{arch} {seed}] {how}  MAE% {mp:.3f} (recorded {rec:.3f}, gate {gate})  '
              f'ceilingAB {row["ceiling_rho_setA_vs_setB"]:.4f}  det {det:.4f}  '
              f'rank1 {row["rank1"]}  ig~shap {row["rho_ig_vs_shap"]:.3f}  '
              f'ig~perm {row["rho_ig_vs_perm"]:.3f}  {row["elapsed_s"]}s', flush=True)
        del m; tf.keras.backend.clear_session(); gc.collect()

R = pd.DataFrame(rows)
R.to_csv(os.path.join(RESULTS, f'analysis18_ig_tenseed{SFX}.csv'), index=False)

# ---- summary: same pair structure as Analysis 17 and the permutation run ----
cols = [f'ig_{l}' for l in COLLAPSED_LABELS]
V = {(r.arch, r.seed): r[cols].values.astype(float) for _, r in R.iterrows()}
lst = [k for k in V if k[0] == 'lstm']; tft = [k for k in V if k[0] == 'tft']
within_l = [rho(V[a], V[b]) for a, b in itertools.combinations(lst, 2)]
within_t = [rho(V[a], V[b]) for a, b in itertools.combinations(tft, 2)]
across = [rho(V[a], V[b]) for a in lst for b in tft]
med = lambda x: float(np.median(x)) if len(x) else np.nan
L_, T_ = R[R.arch == 'lstm'], R[R.arch == 'tft']

# sanity: recompute the Analysis 17 figure the thesis quotes
shp_perm_t = med(T_.rho_shap_vs_perm.dropna())
shp_perm_l = med(L_.rho_shap_vs_perm.dropna())

p1 = med(within_t) > 0.85
p2 = med(T_.rho_ig_vs_shap) > 0.671
p3 = (med(L_.rho_ig_vs_perm) < med(across)) and (med(T_.rho_ig_vs_perm) < med(across))
p4 = (L_.rank1 == 'Hour of Day').sum() >= 9 and (T_.rank1 == 'Hour of Day').sum() >= 9
pf = lambda b: 'PASS' if b else 'FAIL'
full = len(args.seeds) == 10

lines = [
    'ANALYSIS 18 -- Integrated Gradients on ten-seed anchor models, both architectures',
    f'models: {len(lst)} LSTM, {len(tft)} TFT; set A = X_test[:{args.n_explain}], '
    f'set B = X_test[{args.n_explain}:{2*args.n_explain}]; steps {args.steps}; baseline = train mean',
    f'MAE gate (reloaded MAE% == Analysis 5 record within {GATE_TOL_PP} pp): '
    f'{int(R.gate_mae_matches.sum())}/{len(R)} pass',
    f'IG sample ceiling (set A vs set B) median: LSTM {med(L_.ceiling_rho_setA_vs_setB):.4f}  '
    f'TFT {med(T_.ceiling_rho_setA_vs_setB):.4f}  (min {R.ceiling_rho_setA_vs_setB.min():.4f})',
    f'determinism (set A twice, first model per arch): {R.determinism_rho_setA_twice.dropna().round(4).tolist()}',
    f'rank-1 LSTM: {L_.rank1.value_counts().to_dict()}',
    f'rank-1 TFT : {T_.rank1.value_counts().to_dict()}',
    f'median Spearman within-LSTM {med(within_l):.4f} (n={len(within_l)})  '
    f'within-TFT {med(within_t):.4f} (n={len(within_t)})  across {med(across):.4f} (n={len(across)})',
    'Compare: permutation 0.783 / 0.762 / 0.615 ; SHAP 0.958 / 0.909 / 0.874 (Analysis 17).',
    f'same model, median: IG~SHAP  LSTM {med(L_.rho_ig_vs_shap):.4f}  TFT {med(T_.rho_ig_vs_shap):.4f}',
    f'same model, median: IG~perm  LSTM {med(L_.rho_ig_vs_perm):.4f}  TFT {med(T_.rho_ig_vs_perm):.4f}',
    f'check, SHAP~perm recomputed (thesis quotes 0.783 / 0.671): LSTM {shp_perm_l:.4f}  TFT {shp_perm_t:.4f}',
    '',
    'PREDICTIONS (registered before the run)' + ('' if full else '  [SUBSET RUN - not the reported test]'),
    f'  P1 within-TFT IG > 0.85                         : {pf(p1)}  ({med(within_t):.4f})',
    f'  P2 IG~SHAP on TFT > SHAP~perm on TFT (0.671)    : {pf(p2)}  ({med(T_.rho_ig_vs_shap):.4f})',
    f'  P3 IG~perm (both archs) < IG across-arch        : {pf(p3)}  '
    f'(LSTM {med(L_.rho_ig_vs_perm):.4f}, TFT {med(T_.rho_ig_vs_perm):.4f} vs {med(across):.4f})',
    f'  P4 Hour of Day rank-1 in >= 9/10 per arch       : {pf(p4)}',
]
open(os.path.join(RESULTS, f'analysis18_summary{SFX}.txt'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))

with open(os.path.join(RESULTS, f'analysis18_PROVENANCE{SFX}.txt'), 'w') as f:
    f.write(f'run (UTC) : {pd.Timestamp.now(tz="UTC").isoformat()}\n')
    f.write(f'script    : {os.path.basename(__file__)} sha256 {sha(__file__)}\n')
    f.write(f'python {platform.python_version()} tensorflow {tf.__version__} '
            f'numpy {np.__version__} pandas {pd.__version__}  GPUs {len(gpus)}\n')
    f.write(f'notebook  : {os.path.basename(NB)} sha256 {sha(NB)} (cell 6 exec\'d for builders)\n')
    f.write(f'pipeline  : fyp_pipeline.py sha256 {sha(os.path.join(PHASE2, "fyp_pipeline.py"))}\n')
    f.write(f'compare   : analysis17_shap_tenseed.csv sha256 {sha(shap_csv)}\n')
    f.write(f'compare   : analysis5_tenseed_attribution.csv sha256 {sha(perm_csv)}\n')
    for p in input_files:
        f.write(f'input     : {os.path.basename(p)} sha256 {sha(p)}\n')
    for _, r in R.iterrows():
        f.write(f'model     : {r.arch}_seed{r.seed}.keras sha256 {r.model_sha256}\n')
print('done')
