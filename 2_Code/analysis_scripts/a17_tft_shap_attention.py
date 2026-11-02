#!/usr/bin/env python3
"""
ANALYSIS 17 -- SHAP GradientExplainer on both architectures, plus TFT native
weights, on the ten-seed anchor models of Analysis 5.

Why: the agreed extension deliverable for Architecture Invariance specifies
"SHAP GradientExplainer applied for a like-for-like attribution comparison and
the model's native attention weights reported alongside". The thesis's
architecture comparison used permutation importance only. This script closes
that gap on the models that already exist. It trains nothing.

Protocol (matches Analysis 7 / the A7 confirmation run):
  anchor Bobcat_education_Alissa, 874-sequence test partition,
  background = X_train[:200], explained = X_test[:200],
  mean |SHAP| over samples and timesteps, sin/cos collapse to 12 features.
  Each model is explained twice with different GradientExplainer random
  states (rseed 0 and 1) so a per-model SHAP estimation ceiling is available.
  TFT native outputs: VSN weights (mean over samples and timesteps, collapsed
  the same way) and interpretable attention (head-averaged, row of the final
  query timestep, mean over samples -> weight on each of the 48 lags).

Inputs are READ ONLY. Outputs go to NEW files only:
  results/analysis17_shap_tenseed.csv
  results/analysis17_tft_native.csv
  results/analysis17_tft_attention_by_lag.csv
  results/analysis17_summary.txt
  results/analysis17_PROVENANCE.txt

Usage (WSL, fyp_env, GPU):
  python a17_tft_shap_attention.py \
      --phase2 /home/pie/fyp/phase2 \
      --data "/home/pie/fyp/ZEIT4500 - FYP/Data"
Optional: --seeds 2016 2017   (subset, for a quick check)
          --nsamples 200      (GradientExplainer samples per explanation)
"""
import os, sys, json, time, hashlib, argparse, platform, itertools
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--phase2', default='/home/pie/fyp/phase2')
ap.add_argument('--data', default='/home/pie/fyp/ZEIT4500 - FYP/Data')
ap.add_argument('--seeds', type=int, nargs='*', default=list(range(2016, 2026)))
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

# ---- pipeline + data (identical calls to Analysis 5 cell 4) ----------------
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

# recorded Analysis 5 accuracy, used as a gate before any attribution is trusted
pred_csv = os.path.join(RESULTS, 'analysis5_tenseed_predictive.csv')
recorded = pd.read_csv(pred_csv).set_index(['arch', 'seed'])['mae_pct'].to_dict() \
    if os.path.exists(pred_csv) else {}

# Gate tolerance. A sandbox CPU test (2026-10-12) reproduced the LSTM records to
# 0.001 pp but the TFT records only to 0.02-0.03 pp, identically under both load
# paths, which points at CPU vs GPU float arithmetic, not the weights. On the
# original GPU the TFT should match to 0.01 pp; anything over 0.05 pp means the
# wrong file or a drifted pipeline, so stop and check before using the output.
GATE_TOL_PP = 0.05

bg = X_train[:200].astype(np.float32)
Xe = X_test[:args.n_explain].astype(np.float32)

def shap_vec(m, rseed):
    e = shap.GradientExplainer(m, bg)
    sv = e.shap_values(Xe, nsamples=args.nsamples, rseed=rseed)
    sv = np.asarray(sv[0] if isinstance(sv, list) else sv)
    sv = sv.reshape(Xe.shape[0], Xe.shape[1], Xe.shape[2])
    return collapse_vec(np.abs(sv).mean(axis=(0, 1)))

def rho(a, b):
    ra, rb = pd.Series(a).rank().values, pd.Series(b).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])

shap_rows, native_rows, lag_rows = [], [], []
for arch in ['lstm', 'tft']:
    for seed in args.seeds:
        t0 = time.time()
        m, p, how = load(arch, seed)
        mp = mae_pct(m)
        rec = recorded.get((arch, seed), np.nan)
        gate = bool(np.isfinite(rec) and abs(mp - rec) < GATE_TOL_PP)
        v0 = shap_vec(m, 0)
        v1 = shap_vec(m, 1)
        row = dict(arch=arch, seed=seed, load=how, mae_pct=mp, mae_pct_recorded=rec,
                   gate_mae_matches=gate, ceiling_rho=rho(v0, v1),
                   rank1=COLLAPSED_LABELS[int(np.argmax(v0))],
                   elapsed_s=round(time.time() - t0, 1), model_sha256=sha(p))
        row.update({f'shap_{l}': x for l, x in zip(COLLAPSED_LABELS, v0)})
        row.update({f'shap_r1_{l}': x for l, x in zip(COLLAPSED_LABELS, v1)})
        shap_rows.append(row)
        if arch == 'tft':
            vsn = next(l for l in m.layers if isinstance(l, CUSTOM['VariableSelectionNetwork']))
            att = next(l for l in m.layers if isinstance(l, CUSTOM['InterpretableMultiHeadAttention']))
            am = tf.keras.Model(m.inputs, [m.outputs[0], vsn.output[1], att.output[1]])
            _, w, a = am.predict(Xe, verbose=0)
            wv = collapse_vec(np.asarray(w).mean(axis=(0, 1)))
            nr = dict(seed=seed, vsn_rank1=COLLAPSED_LABELS[int(np.argmax(wv))],
                      rho_vsn_vs_shap=rho(wv, v0))
            nr.update({f'vsn_{l}': x for l, x in zip(COLLAPSED_LABELS, wv)})
            native_rows.append(nr)
            last = np.asarray(a)[:, -1, :].mean(axis=0)          # weight of final query on each lag
            for lag, wt in enumerate(last[::-1]):                 # lag 0 = most recent hour
                lag_rows.append(dict(seed=seed, lag_hours=lag, attention=float(wt)))
        print(f'[{arch} {seed}] {how}  MAE% {mp:.3f} (recorded {rec:.3f}, gate {gate})  '
              f'ceiling {row["ceiling_rho"]:.4f}  rank1 {row["rank1"]}  {row["elapsed_s"]}s', flush=True)
        del m; tf.keras.backend.clear_session()

S = pd.DataFrame(shap_rows); S.to_csv(os.path.join(RESULTS, f'analysis17_shap_tenseed{SFX}.csv'), index=False)
N = pd.DataFrame(native_rows); N.to_csv(os.path.join(RESULTS, f'analysis17_tft_native{SFX}.csv'), index=False)
L = pd.DataFrame(lag_rows); L.to_csv(os.path.join(RESULTS, f'analysis17_tft_attention_by_lag{SFX}.csv'), index=False)

# ---- summary: same pair structure as the permutation ten-seed comparison ----
cols = [f'shap_{l}' for l in COLLAPSED_LABELS]
V = {(r.arch, r.seed): r[cols].values.astype(float) for _, r in S.iterrows()}
lst = [k for k in V if k[0] == 'lstm']; tft = [k for k in V if k[0] == 'tft']
within_l = [rho(V[a], V[b]) for a, b in itertools.combinations(lst, 2)]
within_t = [rho(V[a], V[b]) for a, b in itertools.combinations(tft, 2)]
across = [rho(V[a], V[b]) for a in lst for b in tft]
lines = [
    'ANALYSIS 17 -- SHAP on ten-seed anchor models, both architectures',
    f'models: {len(lst)} LSTM, {len(tft)} TFT; explained {len(Xe)} test sequences; nsamples {args.nsamples}',
    f'MAE gate (reloaded MAE% == Analysis 5 record within {GATE_TOL_PP} pp): {int(S.gate_mae_matches.sum())}/{len(S)} pass',
    f'SHAP ceiling (rseed 0 vs 1) median: LSTM {np.median(S[S.arch=="lstm"].ceiling_rho):.4f}  TFT {np.median(S[S.arch=="tft"].ceiling_rho):.4f}',
    f'rank-1 LSTM: {S[S.arch=="lstm"].rank1.value_counts().to_dict()}',
    f'rank-1 TFT : {S[S.arch=="tft"].rank1.value_counts().to_dict()}',
    f'median Spearman within-LSTM {np.median(within_l):.4f} (n={len(within_l)})  '
    f'within-TFT {np.median(within_t):.4f} (n={len(within_t)})  across {np.median(across):.4f} (n={len(across)})',
    'Compare: permutation ten-seed medians 0.783 / 0.762 / 0.615 (thesis Sec IV-E).',
]
if len(N):
    lines += [f'TFT VSN rank-1: {N.vsn_rank1.value_counts().to_dict()}',
              f'median Spearman VSN weights vs SHAP (same TFT): {np.median(N.rho_vsn_vs_shap):.4f}',
              'attention by lag: see analysis17_tft_attention_by_lag.csv (lag 0 = most recent hour)']
open(os.path.join(RESULTS, f'analysis17_summary{SFX}.txt'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))

with open(os.path.join(RESULTS, f'analysis17_PROVENANCE{SFX}.txt'), 'w') as f:
    f.write(f'run (UTC) : {pd.Timestamp.utcnow().isoformat()}\n')
    f.write(f'script    : {os.path.basename(__file__)} sha256 {sha(__file__)}\n')
    f.write(f'python {platform.python_version()} tensorflow {tf.__version__} shap {shap.__version__} '
            f'numpy {np.__version__} pandas {pd.__version__}  GPUs {len(gpus)}\n')
    f.write(f'notebook  : {os.path.basename(NB)} sha256 {sha(NB)} (cell 6 exec\'d for builders)\n')
    f.write(f'pipeline  : fyp_pipeline.py sha256 {sha(os.path.join(PHASE2, "fyp_pipeline.py"))}\n')
    for p in input_files:
        f.write(f'input     : {os.path.basename(p)} sha256 {sha(p)}\n')
    for _, r in S.iterrows():
        f.write(f'model     : {r.arch}_seed{r.seed}.keras sha256 {r.model_sha256}\n')
print('done')
