#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os, random, joblib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED']         = '2016'

import ast, warnings
warnings.filterwarnings('ignore', message=".*does not support masking.*")
warnings.filterwarnings('ignore', message='The structure of `inputs` doesn')


# ── Reproducibility ───────────────────────────────────────────────────
import numpy as np
SEED = 2016
random.seed(SEED)
np.random.seed(SEED)

# ── Core imports ──────────────────────────────────────────────────────
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except RuntimeError as e:
        print("memory_growth must be set before GPU init:", e)
print("GPU memory growth enabled:", [g.name for g in gpus])

import gc
import itertools, time
import hashlib
import pandas as pd
import matplotlib.pyplot as plt
import shap
import shutil
import inspect

from pathlib import Path
from tensorflow.keras import layers
from tensorflow.keras.layers import (Input, Dense, LSTM, Dropout,
    MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Add)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import wilcoxon, spearmanr, kruskal
from itertools import combinations

tf.random.set_seed(SEED)

# ── fyp_pipeline ──────────────────────────────────────────────────────
if 'fyp_pipeline' in sys.modules:
    del sys.modules['fyp_pipeline']
sys.path.insert(0, '/home/pie/fyp/phase2')
from fyp_pipeline import prepare_building, align_to_schema, split_sequences, WINDOW_SIZE

# ── Verification ──────────────────────────────────────────────────────
print(sys.executable)
print(f"TF: {tf.__version__}  SHAP: {shap.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")
print(f"SEED: {SEED}  WINDOW_SIZE: {WINDOW_SIZE}")
print("All imports OK")


# In[6]:


# ── Paths ─────────────────────────────────────────────────────────────
PHASE1_DIR = '/home/pie/fyp/ZEIT4500 - FYP'
PHASE2_DIR = '/home/pie/fyp/phase2'
DATA_DIR   = os.path.join(PHASE1_DIR, 'Data')

# ── Load anchor model ─────────────────────────────────────────────────
lstm_anchor = load_model(os.path.join(PHASE1_DIR, 'multisite_anchor_best.keras'))
print(f"Anchor model loaded — params: {lstm_anchor.count_params()}")

# ── Load anchor attribution reference ─────────────────────────────────
anchor_attr = pd.read_csv(os.path.join(PHASE1_DIR, 'anchor_attribution_values.csv'))
print(f"Anchor attribution loaded — {len(anchor_attr)} features")
print(anchor_attr[['feature','shap_mean','perm_delta_kwh']].to_string(index=False))


# ## Pre-registered validity threshold — TFT vs LSTM-VCA
# 
# **Registered:** 2026-07-14, before any `model.fit()` call on the full TFT.
# 
# **Threshold:** The TFT is accepted as a valid architecture-invariance comparison against
# LSTM-VCA if its test-partition MAE% (of mean consumption) is **≤ 26%** — 1.5× the anchor
# LSTM's 17.3% test MAE%.
# 
# **Rationale:** 1.5× allows for the TFT being a data-hungrier architecture trained on the
# same single-building-year dataset (8,712 sequences) without letting an undertrained or
# poorly-suited model masquerade as a valid comparison point.
# 
# **Handling of both outcomes (fixed now, not after the result is seen):**
# - **≤ 26%** — proceed with the SHAP GradientExplainer / native VSN & attention comparison
#   as planned. This is the primary Analysis 1 result.
# - **> 26%** — the divergence itself is reported as the Analysis 1 finding (attribution
#   stability may be architecture-dependent, not a universal property of the data). Analyses
#   2 (cross-meter) and 3 (Integrated Gradients) carry the primary robustness argument instead.
#   No architecture beyond LSTM and TFT (e.g. a Physics-Informed NN) is pursued within the
#   current project scope — that would be a separate, larger modelling effort out of scope for
#   the October 2026 submission.
# 
# Same seed (2016), same chronological 80/10/10 split, same early stopping (patience=15,
# restore_best_weights=True) as LSTM-VCA. Batch size tuned independently for the TFT rather
# than assumed equal to the LSTM's.

# ## Hyperparameter tuning log
# 
# Every configuration in the grid search is logged here (or to a CSV alongside the notebook),
# not just the winning run — required for the reproducibility package regardless of outcome.
# 
# | d_model | n_heads | dropout | lstm_units | batch_size | val_RMSE | val_MAE% | epoch_time_s | best_epoch | notes |
# |---|---|---|---|---|---|---|---|---|---|
# | | | | | | | | | | |
# 
# Grid: `d_model` ∈ {16, 32, 64} · `n_heads` ∈ {2, 4} · `dropout` ∈ {0.1, 0.2, 0.3} ·
# `lstm_units` ∈ {(64,32), (32,16)} · `batch_size` ∈ {64, 256}
# 
# Winning config is carried forward to the held-out test partition and evaluated against the
# 26% pre-registered threshold above.

# ## Gated Residual Network

# In[7]:


class GRN(layers.Layer):
    """Gated Residual Network — TFT's core building block (Lim et al. 2021, eq. 2-5)."""
    def __init__(self, units, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense1 = layers.Dense(units, activation='elu')
        self.dense2 = layers.Dense(units)
        self.gate_dense = layers.Dense(units * 2)  # for GLU
        self.dropout = layers.Dropout(dropout)
        self.layernorm = layers.LayerNormalization()
        self.skip_proj = None  # built lazily if input dim != units

    def build(self, input_shape):
        in_dim = input_shape[-1]
        if in_dim != self.units:
            self.skip_proj = layers.Dense(self.units)
        super().build(input_shape)

    def call(self, x, context=None, training=None):
        skip = self.skip_proj(x) if self.skip_proj is not None else x
        h = self.dense1(tf.concat([x, context], axis=-1) if context is not None else x)
        h = self.dense2(h)
        h = self.dropout(h, training=training)
        gate_in = self.gate_dense(h)
        a, b = tf.split(gate_in, 2, axis=-1)
        gated = a * tf.sigmoid(b)  # GLU
        return self.layernorm(skip + gated)


# ## Variable Selection Network

# In[8]:


class VariableSelectionNetwork(layers.Layer):
    """Per-timestep feature weighting. Applied via TimeDistributed by the caller,
    or directly since it already operates on the last axis of a (batch, time, features) tensor."""
    def __init__(self, n_features, units, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.units = units
        # one single-variable GRN per feature (input dim 1 -> units)
        self.feature_grns = [GRN(units, dropout) for _ in range(n_features)]
        # weighting GRN sees all features flattened -> softmax over n_features
        self.weight_grn = GRN(n_features, dropout)
        self.softmax = layers.Softmax(axis=-1)

    def call(self, x, training=None):
        # x: (batch, time, n_features)
        flat = x  # already flattened on last axis
        weights = self.weight_grn(flat, training=training)   # (batch, time, n_features)
        weights = self.softmax(weights)
        transformed = []
        for j in range(self.n_features):
            feat_j = x[..., j:j+1]                            # (batch, time, 1)
            transformed.append(self.feature_grns[j](feat_j, training=training))  # (batch, time, units)
        stacked = tf.stack(transformed, axis=-1)               # (batch, time, units, n_features)
        weights_exp = tf.expand_dims(weights, axis=-2)         # (batch, time, 1, n_features)
        selected = tf.reduce_sum(stacked * weights_exp, axis=-1)  # (batch, time, units)
        return selected, weights  # return weights for your native-attribution logging


# ## Interpretable multi-head attention (shared value projection)

# In[9]:


class InterpretableMultiHeadAttention(layers.Layer):
    """TFT attention: separate Q/K per head, but V is SHARED across heads
    and averaged post-attention — this is what makes it interpretable."""
    def __init__(self, d_model, n_heads, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_dense = layers.Dense(d_model)
        self.k_dense = layers.Dense(d_model)
        self.v_dense = layers.Dense(self.d_head)   # shared across heads — key difference
        self.out_dense = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout)

    def call(self, x, mask=None, training=None):
        B, T = tf.shape(x)[0], tf.shape(x)[1]
        q = tf.reshape(self.q_dense(x), (B, T, self.n_heads, self.d_head))
        k = tf.reshape(self.k_dense(x), (B, T, self.n_heads, self.d_head))
        v = self.v_dense(x)  # (B, T, d_head) — one shared value set

        q = tf.transpose(q, [0, 2, 1, 3])  # (B, heads, T, d_head)
        k = tf.transpose(k, [0, 2, 1, 3])
        scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(self.d_head, tf.float32))
        if mask is not None:
            scores += (1.0 - mask) * -1e9
        attn = tf.nn.softmax(scores, axis=-1)          # (B, heads, T, T)
        attn = self.dropout(attn, training=training)
        attn_mean = tf.reduce_mean(attn, axis=1)        # average over heads -> the interpretable weights
        out = tf.matmul(attn_mean, v)                   # (B, T, d_head), same V for every head
        return self.out_dense(out), attn_mean


# ## Causal mask - matches the LSTM's forward-only information flow

# In[10]:


def causal_mask(T):
    m = tf.linalg.band_part(tf.ones((T, T)), -1, 0)
    return m[tf.newaxis, tf.newaxis, :, :]  # broadcastable over (B, heads, T, T)


# ## Assemble the model

# In[11]:


def build_tft(seq_len=48, n_features=15, d_model=32, n_heads=4, lstm_units=(64, 32), dropout=0.2):
    inputs = layers.Input(shape=(seq_len, n_features))

    vsn = VariableSelectionNetwork(n_features, d_model, dropout)
    selected, vsn_weights = vsn(inputs)  # (B, T, d_model)

    x = layers.LSTM(lstm_units[0], return_sequences=True, dropout=dropout)(selected)
    x = layers.LSTM(lstm_units[1], return_sequences=True, dropout=dropout)(x)

    attn_layer = InterpretableMultiHeadAttention(d_model=lstm_units[1], n_heads=n_heads, dropout=dropout)
    attn_out, attn_weights = attn_layer(x, mask=causal_mask(seq_len))

    grn_out = GRN(lstm_units[1], dropout)(attn_out)
    final_step = grn_out[:, -1, :]   # take last timestep, mirrors LSTM's return_sequences=False output
    output = layers.Dense(1, activation='linear')(final_step)

    model = tf.keras.Model(inputs, output)
    attribution_model = tf.keras.Model(inputs, [output, vsn_weights, attn_weights])
    return model, attribution_model


# In[12]:


# ── Paths ─────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PHASE1_DIR, 'Data')   # capital D — confirmed via find

# ── Electricity (all sites) ───────────────────────────────────────────
elec_path   = os.path.join(DATA_DIR, 'electricity.csv')
electricity = pd.read_csv(elec_path, parse_dates=['timestamp'])
print(f"Electricity   : {electricity.shape}")

# ── Weather (all sites) ───────────────────────────────────────────────
wx_path     = os.path.join(DATA_DIR, 'weather.csv')
weather_all = pd.read_csv(wx_path, parse_dates=['timestamp'])
print(f"Weather       : {weather_all.shape}")

# ── Meter covariates ────────────────────────────────────────────────────
METER_FILES = {
    'gas': 'gas.csv', 'hotwater': 'hotwater.csv', 'chilledwater': 'chilledwater.csv',
    'steam': 'steam.csv', 'solar': 'solar.csv', 'water': 'water.csv', 'irrigation': 'irrigation.csv',
}
meter_dfs = {}
for name, fname in METER_FILES.items():
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path):
        meter_dfs[name] = pd.read_csv(path, parse_dates=['timestamp'])
print(f"Meters loaded : {sorted(meter_dfs.keys())}")

# ── Build TFT training sequences for the anchor building ────────────────
BUILDING_ID = 'Bobcat_education_Alissa'

X_seq, y_seq, scaler_y, feature_cols, raw_y_mean = prepare_building(
    BUILDING_ID, electricity, weather_all, meter_dfs
)
print(f"Feature schema ({len(feature_cols)}): {feature_cols}")
print(f"Raw mean consumption: {raw_y_mean:.2f} kWh")

X_train, y_train, X_val, y_val, X_test, y_test = split_sequences(
    X_seq, y_seq, train_frac=0.80, val_frac=0.10
)
print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")


# In[13]:


# =====================================================================
# STEP 5 — CELL: Building discovery & quality filter  (port of Phase 1 §4)
# =====================================================================
# Ported verbatim from FYP_MultiSite_BDGP2_SHAP_CPU §4. Constants taken
# from the Phase 1 globals cell (lines 487-494 of the decoded export).
# Expected output: 1,323 -> 1,294 buildings.

ANCHOR_BUILDING = 'Bobcat_education_Alissa'
YEAR            = 2016
MISSING_THRESH  = 0.05   # drop buildings with > 5% missing electricity
MIN_MEAN_KWH    = 1.0    # drop near-zero buildings (noise floor)

# ── Metadata (canonical source of building type) ──────────────────────
meta_path = os.path.join(DATA_DIR, 'metadata.csv')
metadata  = pd.read_csv(meta_path) if os.path.exists(meta_path) else None
print(f"Metadata: {'loaded ' + str(metadata.shape) if metadata is not None else 'NOT FOUND'}")

# ── Filter to 2016 ────────────────────────────────────────────────────
elec_2016     = electricity[electricity['timestamp'].dt.year == YEAR].copy()
building_cols = [c for c in elec_2016.columns if c != 'timestamp']

# ── Missing data per building ─────────────────────────────────────────
missing_pct     = elec_2016[building_cols].isnull().mean()
clean_buildings = missing_pct[missing_pct <= MISSING_THRESH].index.tolist()
print(f"Buildings with <= {MISSING_THRESH*100:.0f}% missing : {len(clean_buildings)}")

# ── Near-zero mean filter ─────────────────────────────────────────────
mean_kwh        = elec_2016[clean_buildings].mean()
clean_buildings = mean_kwh[mean_kwh >= MIN_MEAN_KWH].index.tolist()
print(f"After min-mean filter (>= {MIN_MEAN_KWH} kWh)  : {len(clean_buildings)}")

# ── Summary table (site + type) ───────────────────────────────────────
from fyp_pipeline import infer_site

building_summary = pd.DataFrame({
    'building'   : clean_buildings,
    'site'       : [infer_site(b) for b in clean_buildings],
    'missing_pct': [missing_pct[b] * 100 for b in clean_buildings],
    'mean_kwh'   : [mean_kwh[b] for b in clean_buildings],
})

if metadata is not None and 'primaryspaceusage' in metadata.columns:
    key = 'building_id' if 'building_id' in metadata.columns else metadata.columns[0]
    meta_lookup = metadata.set_index(key)['primaryspaceusage'].to_dict()
    building_summary['type'] = building_summary['building'].map(meta_lookup).fillna('unknown')
else:
    building_summary['type'] = 'unknown'

# ── Anchor first, then the rest (matches Phase 1 ordering exactly) ────
others          = [b for b in clean_buildings if b != ANCHOR_BUILDING]
clean_buildings = [ANCHOR_BUILDING] + others

print(f"\nTotal buildings   : {len(clean_buildings)}")
print(f"Sites represented : {len(set(building_summary['site']))}")
print(f"\nType distribution:\n{building_summary['type'].value_counts().to_string()}")

assert len(clean_buildings) == 1294, (
    f"Expected 1294 buildings (Phase 1 result), got {len(clean_buildings)}. "
    "Investigate before proceeding — Gate 2 must run on the same population."
)
print("\n✓ Matches Phase 1: 1,294 buildings")


# ── Anchor feature schema ────────────────────────────────────────────
# Phase 1: ANCHOR_FEATURE_SCHEMA = anchor_features.copy()
# `feature_cols` from cell 18 IS the anchor's feature list. Lock it.
ANCHOR_FEATURE_SCHEMA = feature_cols.copy()
N_ANCHOR_FEATURES     = len(ANCHOR_FEATURE_SCHEMA)

print(f"\nAnchor feature schema ({N_ANCHOR_FEATURES} features):")
for i, f in enumerate(ANCHOR_FEATURE_SCHEMA):
    print(f"  [{i:2d}] {f}")

assert N_ANCHOR_FEATURES == 15, f"Expected 15 features, got {N_ANCHOR_FEATURES}"


# ## Note on sequence count: 8,736 sequences (2016, 48h window)
# 
# n=8,736 (366 days × 24h − 48h window, 2016 leap year). This is the sequence count used for
# both the anchor LSTM refit and the TFT in this analysis, generated fresh from the Phase 1
# CSVs (confirmed byte-identical via diff, 2026-07-14) via the current fyp_pipeline.py. Both
# models share this identical data slice, scaler, and split.

# In[14]:


# ── Multi-run LSTM refit (uncertainty quantification) ───────────────────
N_RUNS = 3
lstm_run_results = []

for run in range(N_RUNS):
    tf.random.set_seed(SEED)
    lstm_inputs = Input(shape=(WINDOW_SIZE, len(feature_cols)))
    x = LSTM(64, return_sequences=True, dropout=0.2)(lstm_inputs)
    x = LSTM(32, return_sequences=False, dropout=0.2)(x)
    lstm_output = Dense(1, activation='linear')(x)
    lstm_refit = Model(lstm_inputs, lstm_output)
    lstm_refit.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    es = EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss')
    hist = lstm_refit.fit(X_train, y_train, validation_data=(X_val, y_val),
                           epochs=300, batch_size=32, callbacks=[es], verbose=0)

    pred_kwh = scaler_y.inverse_transform(lstm_refit.predict(X_test, verbose=0)).flatten()
    true_kwh = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    mae_pct = 100 * mean_absolute_error(true_kwh, pred_kwh) / true_kwh.mean()

    lstm_run_results.append(mae_pct)
    print(f"Run {run+1}/{N_RUNS}: test MAE% = {mae_pct:.2f}%  (stopped epoch {len(hist.history['loss'])})")
    if run == 0:
        lstm_swept_mae_pct = mae_pct          # ← ADD: record the swept instance
        lstm_refit.save(os.path.join(PHASE2_DIR, 'lstm_refit_bobcat.keras'))
        joblib.dump(scaler_y, os.path.join(PHASE2_DIR, 'scaler_y_bobcat.pkl'))
        joblib.dump(feature_cols, os.path.join(PHASE2_DIR, 'feature_cols_bobcat.pkl'))
lstm_mean = np.mean(lstm_run_results)
lstm_std  = np.std(lstm_run_results)
print(f"\nLSTM refit test MAE%: {lstm_mean:.2f}% ± {lstm_std:.2f}%  "
      f"(n={N_RUNS}, range {min(lstm_run_results):.1f}-{max(lstm_run_results):.1f}%)")
print(f"Swept instance (run 1): {lstm_swept_mae_pct:.2f}%  ← this is the model used for the sweep")
lstm_refit = load_model(os.path.join(PHASE2_DIR, 'lstm_refit_bobcat.keras'))


# ## Extended grid + top-K confirmation protocol
# 
# **Registered:** 2026-07-14, before the extended grid search runs.
# 
# **Context:** the truncated 32-config grid's winning configuration (`d_model=64, n_heads=2,
# dropout=0.2, lstm_units=(32,16), batch_size=64`) showed an ~8-point gap between its
# single-run validation score (val_MAE%=12.1%) and its 3-run confirmed test mean
# (19.98% ± 1.05%), consistent with selection bias ("winner's curse") from taking `argmin`
# over 32 noisy single-run draws. Best-of-3 across the full extended grid would not correct
# this efficiently — it spends most of the added compute reconfirming configs that were never
# competitive. The protocol below targets the actual bias instead.
# 
# **Grid:** extended to the full pre-registered space — `d_model` ∈ {16, 32, 64} ·
# `n_heads` ∈ {2, 4} · `dropout` ∈ {0.1, 0.2, 0.3} · `lstm_units` ∈ {(64,32), (32,16)} ·
# `batch_size` ∈ {64, 256} — completing the originally-specified hyperparameter space rather
# than re-litigating the already-confirmed result below.
# 
# **Protocol:**
# 1. Single-run screening across the full extended grid, ranked by val_RMSE (as before).
# 2. The **top 5 configs by val_RMSE** proceed to 3-run test-set confirmation (same protocol
#    as the truncated-grid winner: seed=2016, same data/split/scaler, `EarlyStopping`
#    patience=15, epochs cap=300).
# 3. The **final Analysis 1 TFT candidate is the config with the best confirmed 3-run test
#    MAE% mean** among those 5 — not necessarily the single-run screening winner. If a
#    different config among the top 5 out-performs the original winner once confirmed, that
#    is itself reported as direct empirical evidence of the winner's curse in this pipeline.
# 
# **Prior confirmed result (truncated grid, stands as primary Analysis 1 finding regardless
# of this extension's outcome):** TFT test MAE% = 19.98% ± 1.05% (n=3) vs LSTM refit mean
# 17.08% ± 2.14% (n=3). Threshold 1.5× LSTM mean = 25.62%. **PASS.**
# 
# **Estimated runtime:** ~8h screening (72 configs × ~400s/config) + ~1.7h confirmation
# (5 configs × 3 runs) ≈ 9.5–10h total.

# ## A Note on the Extended Grid:
# 
# The extended grid log is included as tft_extended_grid_log.csv; this cell resumes from it if present, or regenerates all 36 configs from scratch on a clean checkout. Results are denominated by the full-year mean (raw_y_mean); config ranking is invariant to this choice.

# In[15]:


csv_path = os.path.join(PHASE2_DIR, 'tft_extended_grid_log.csv')
existing_df = pd.read_csv(csv_path)
existing_df['lstm_units'] = existing_df['lstm_units'].apply(ast.literal_eval)
results = existing_df.to_dict('records')
n_done = len(results)
print(f"Resuming — {n_done} configs already completed, loaded from {csv_path}")

grid = list(itertools.product(
    [16, 32, 64], [2, 4], [0.1, 0.2, 0.3], [(64, 32), (32, 16)], [64, 256],
))

for i, (d_model, n_heads, dropout, lstm_units, batch_size) in enumerate(grid[n_done:], start=n_done):
    tf.random.set_seed(SEED)
    model, attribution_model = build_tft(
        seq_len=WINDOW_SIZE, n_features=15, d_model=d_model,
        n_heads=n_heads, lstm_units=lstm_units, dropout=dropout
    )
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    es = EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss')
    t0 = time.time()
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=300, batch_size=batch_size, callbacks=[es], verbose=0
    )
    epoch_time = (time.time() - t0) / len(history.history['loss'])

    val_pred_kwh = scaler_y.inverse_transform(model.predict(X_val, verbose=0)).flatten()
    y_val_kwh    = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
    val_rmse    = np.sqrt(mean_squared_error(y_val_kwh, val_pred_kwh))
    val_mae_pct = 100 * mean_absolute_error(y_val_kwh, val_pred_kwh) / raw_y_mean

    row = dict(d_model=d_model, n_heads=n_heads, dropout=dropout,
               lstm_units=lstm_units, batch_size=batch_size,
               val_rmse=val_rmse, val_mae_pct=val_mae_pct,
               epoch_time_s=epoch_time, best_epoch=len(history.history['loss']))
    results.append(row)
    pd.DataFrame(results).to_csv(csv_path, index=False)

    print(f"[{i+1}/{len(grid)}] d_model={d_model} heads={n_heads} drop={dropout} "
          f"lstm={lstm_units} bs={batch_size} -> val_RMSE={val_rmse:.3f} val_MAE%={val_mae_pct:.1f}")

results_df = pd.DataFrame(results).sort_values('val_rmse')
top5 = results_df.head(5).to_dict('records')
print(f"\nScreening complete. Top 5 configs by val_RMSE:")
for r in top5:
    print(r)


# In[16]:


# ── Top-5 confirmation: 3 runs each ──────────────────────────────────────
N_RUNS = 3
confirmation_results = []

for rank, cfg in enumerate(top5):
    arch_kwargs = dict(d_model=cfg['d_model'], n_heads=cfg['n_heads'],
                        dropout=cfg['dropout'], lstm_units=tuple(cfg['lstm_units']))
    run_maes = []
    for run in range(N_RUNS):
        tf.random.set_seed(SEED)
        model, _ = build_tft(seq_len=WINDOW_SIZE, n_features=15, **arch_kwargs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        es = EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss')
        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=300, batch_size=cfg['batch_size'], callbacks=[es], verbose=0)

        pred_kwh = scaler_y.inverse_transform(model.predict(X_test, verbose=0)).flatten()
        true_kwh = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        mae_pct = 100 * mean_absolute_error(true_kwh, pred_kwh) / raw_y_mean
        run_maes.append(mae_pct)

    mean_mae, std_mae = np.mean(run_maes), np.std(run_maes)
    confirmation_results.append(dict(rank=rank+1, **arch_kwargs, batch_size=cfg['batch_size'],
                                      screening_val_rmse=cfg['val_rmse'],
                                      confirmed_mean=mean_mae, confirmed_std=std_mae, runs=run_maes))
    pd.DataFrame(confirmation_results).to_csv(os.path.join(PHASE2_DIR, 'tft_top5_confirmation_log.csv'), index=False)
    print(f"Rank {rank+1} (screening val_RMSE={cfg['val_rmse']:.3f}): "
          f"confirmed test MAE% = {mean_mae:.2f}% ± {std_mae:.2f}%  (runs: {run_maes})")

winner = min(confirmation_results, key=lambda r: r['confirmed_mean'])
print(f"\nBest confirmed config: {winner}")
print(f"Compare vs original truncated-grid confirmed result: 19.98% ± 1.05%")


# In[17]:


# =====================================================================
# STEP 6 — CELL: Rebuild, train and SAVE the selected TFT (Rank 2)
# =====================================================================
# Selected config: rank 2 of the top-5 confirmation.
#
# HONESTY NOTE:
#   Ranks 1-4 confirmed at 19.21±1.18, 19.06±1.45, 19.09±0.91,
#   19.53±1.51 — a 0.47pp spread across means against a within-config
#   sigma of 0.91-1.51pp. The selection is NOT statistically meaningful;
#   rank 2 was taken as the lowest confirmed mean and any of the top
#   four would serve equally.
#
# PROVENANCE NOTE:
#   This is a FRESH training run in this session — a fourth draw from
#   the same distribution as the three confirmation runs. Given the
#   ±1.45pp spread it may land anywhere in ~17-21%. That is expected.
#   Report this instance's own anchor MAE% alongside the n=3 mean ± std,
#   and state that the swept model is one draw from that distribution.


TFT_BEST_CONFIG = {
    'd_model':    64,
    'n_heads':    4,
    'dropout':    0.2,
    'lstm_units': (64, 32),
    'batch_size': 64,
}
print(f"Training selected TFT config: {TFT_BEST_CONFIG}")

tf.random.set_seed(SEED)
tft_best, tft_attribution = build_tft(
    seq_len=WINDOW_SIZE,
    n_features=N_ANCHOR_FEATURES,
    d_model=TFT_BEST_CONFIG['d_model'],
    n_heads=TFT_BEST_CONFIG['n_heads'],
    lstm_units=TFT_BEST_CONFIG['lstm_units'],
    dropout=TFT_BEST_CONFIG['dropout'],
)
tft_best.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

es = EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss')
hist = tft_best.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=300,
    batch_size=TFT_BEST_CONFIG['batch_size'],
    callbacks=[es],
    verbose=0,
)

# ── This instance's own anchor test MAE% (same convention as cell 20) ──
pred_kwh = scaler_y.inverse_transform(tft_best.predict(X_test, verbose=0)).flatten()
true_kwh = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
tft_swept_mae_pct = 100 * mean_absolute_error(true_kwh, pred_kwh) / true_kwh.mean()

print(f"\nSwept TFT instance test MAE% : {tft_swept_mae_pct:.2f}%")
print(f"  (n=3 confirmation was       : 19.06% ± 1.45%)")
print(f"  stopped at epoch            : {len(hist.history['loss'])}")

tft_best.save(os.path.join(PHASE2_DIR, 'tft_best_bobcat.keras'))
tft_attribution.save(os.path.join(PHASE2_DIR, 'tft_attribution_bobcat.keras'))
joblib.dump(TFT_BEST_CONFIG, os.path.join(PHASE2_DIR, 'tft_best_config.pkl'))
print(f"\n✓ Saved -> {os.path.join(PHASE2_DIR, 'tft_best_bobcat.keras')}")

# ── Gate 1 check, reported explicitly (pre-registration only earns
#    credibility if the comparison is actually reported) ──────────────
GATE1_THRESHOLD = 1.5 * 17.3   # = 25.95%
print(f"\nGate 1: TFT {tft_swept_mae_pct:.2f}% vs threshold {GATE1_THRESHOLD:.2f}% "
      f"-> {'PASS' if tft_swept_mae_pct <= GATE1_THRESHOLD else 'FAIL'}")


# In[18]:


# =====================================================================
# PHASE 2 -- CELL A (v3): Per-building sweep, BOTH models, ONE prep
# =====================================================================
# Corrected against the real fyp_pipeline.py signatures:
#   prepare_building(building_name, electricity_df, weather_df,
#                    meter_dfs_dict=None, year=YEAR, window=WINDOW_SIZE,
#                    meter_miss_thresh=METER_MISS_THRESH)
#     -> (X_seq, y_seq, scaler_y, feature_cols, raw_y_mean)
#   align_to_schema(X_seq, feature_cols_used, schema)   # 3 args
#
# OPTIMISATION: prepare_building() is the expensive step (pandas merges,
# scaler fit, 8,736-iteration window loop in Python). Calling it once per
# building and scoring BOTH models against the same prepared array halves
# the cost of the sweep. It also guarantees both models see byte-identical
# inputs, which is a stronger comparison control than two separate sweeps.
#
# METRIC CONVENTION -- matches Phase 1 multisite loop exactly:
#   y_pred_b  = scaler_y_b.inverse_transform(y_pred_s)
#   y_true_b  = scaler_y_b.inverse_transform(y_seq_b.reshape(-1, 1))
#   mae_pct_b = mae_b / y_true_b.mean() * 100
# Per-building scaler, per-building denominator. raw_y_mean (61.79) is
# ANCHOR-SCOPED ONLY and is NOT used as a denominator here.
#
# NOTE on scaler semantics (document this in the report):
#   prepare_building() fits scaler_y on THE TARGET BUILDING's own first
#   80%. The anchor-trained model's scaled outputs are therefore mapped
#   back through a scale the model never saw in training. This is the
#   Phase 1 convention and is retained for comparability, but it is a
#   methodological choice, not a neutral one -- it is the likely source
#   of the extreme right tail in the Phase 1 MAE% distribution
#   (max 4,345.6%) for buildings whose scale diverges from the anchor's.
# ---------------------------------------------------------------------

OUT_DIR = Path(PHASE2_DIR) / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SWEEP_CSV = OUT_DIR / 'per_building_both_models.csv'


def score_one_building(building, models, schema,
                       electricity, weather_all, meter_dfs):
    """
    Prepare ONE building once, score EVERY model in `models` against it.

    models : dict[str, keras.Model]  e.g. {'lstm': lstm_refit, 'tft': tft_best}
    schema : list[str]               the locked 15-feature anchor schema

    Returns a single flat dict (one CSV row per building, one column
    block per model).
    """
    row = {'building': building}

    try:
        X_seq_b, y_seq_b, scaler_y_b, feat_cols_b, raw_mean = prepare_building(
            building, electricity, weather_all, meter_dfs
        )
    except Exception as e:
        row['status'] = f'prep_failed: {type(e).__name__}: {e}'
        return row

    if X_seq_b is None or len(X_seq_b) == 0:
        row['status'] = 'no_sequences'
        return row

    X_aligned = align_to_schema(X_seq_b, feat_cols_b, schema)

    # Inverse-transform truth ONCE -- identical for every model.
    y_true_b   = scaler_y_b.inverse_transform(y_seq_b.reshape(-1, 1))
    denom_b    = float(y_true_b.mean())

    row.update({
        'status':       'ok',
        'n_seq':        int(len(X_aligned)),
        'n_feat_used':  len(feat_cols_b),
        'raw_mean_kwh': float(raw_mean),   # descriptive only, NOT the denom
        'seq_mean_kwh': denom_b,           # the actual denominator
    })

    # Guard: a near-zero denominator makes MAE% meaningless, not just large.
    if abs(denom_b) < 1e-6:
        row['status'] = 'zero_denominator'
        return row

    for name, model in models.items():
        try:
            y_pred_s = model.predict(X_aligned, batch_size=256, verbose=0)
            y_pred_b = scaler_y_b.inverse_transform(y_pred_s.reshape(-1, 1))
            mae_b    = mean_absolute_error(y_true_b, y_pred_b)
            row[f'mae_kwh_{name}'] = float(mae_b)
            row[f'mae_pct_{name}'] = float(mae_b / denom_b * 100)
        except Exception as e:
            row[f'mae_kwh_{name}'] = np.nan
            row[f'mae_pct_{name}'] = np.nan
            row[f'err_{name}']     = f'{type(e).__name__}: {e}'

    return row


def sweep_both(models, buildings, schema,
               electricity, weather_all, meter_dfs,
               out_path=SWEEP_CSV, resume=True):
    """
    Sweep all buildings, appending incrementally.

    Incremental append + resume: an interruption costs nothing. Same
    lesson as the grid search. Safe to restart after a crash or a
    kernel death.
    """
    done = set()
    if resume and out_path.exists():
        done = set(pd.read_csv(out_path)['building'])
        print(f'Resuming: {len(done)} buildings already scored.')

    todo  = [b for b in buildings if b not in done]
    first = not out_path.exists()

    for i, b in enumerate(todo):
        row = score_one_building(b, models, schema,
                                 electricity, weather_all, meter_dfs)
        pd.DataFrame([row]).to_csv(out_path, mode='a',
                                   header=first, index=False)
        first = False
        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(todo)}  ({b})')

    print(f'Sweep complete -> {out_path}')
    return pd.read_csv(out_path)


# --- Smoke test FIRST. Do not skip. ----------------------------------
# This is where an align_to_schema / n_features mismatch surfaces,
# before it costs a 1,294-building run.
#
models = {'lstm': lstm_refit, 'tft': tft_best}
test_row = score_one_building(clean_buildings[0], models,
                               ANCHOR_FEATURE_SCHEMA,
                               electricity, weather_all, meter_dfs)
print(test_row)
#Expect: status='ok', n_seq ~8,700, both mae_pct_* present and plausible.

# --- Then the full sweep ---------------------------------------------
sweep_df = sweep_both(models, clean_buildings, ANCHOR_FEATURE_SCHEMA,
                       electricity, weather_all, meter_dfs)

# --- Sanity check against Phase 1 BEFORE moving on -------------------
# Phase 1 LSTM cross-building: median 40.2%, mean 74.2%, min 2.0%,
# max 4,345.6% (n=1,294). If the LSTM column here diverges materially
# from that, something has changed between the two pipelines and you
# want to know NOW, not after building Gate 2 on top of it.
#
ok = sweep_df[sweep_df.status == 'ok']
print(ok['mae_pct_lstm'].describe().round(1))
print(ok['mae_pct_tft'].describe().round(1))
# ---------------------------------------------------------------------


# In[19]:


# =====================================================================
# STEP 7 — CELL: Smoke test — ONE building, BOTH models.
# =====================================================================
# This is where an align_to_schema / n_features mismatch surfaces,
# before it costs a 1,294-building run.

models = {'lstm': lstm_refit, 'tft': tft_best}

# Anchor first — sanity check. Full-year MAE% here (LSTM ~8%, TFT ~11%)
# is lower than the 17.3% test-partition figure because the full year
# includes the training period the model fits well. We only need
# status='ok', n_seq ~8,700, and finite mae_pct — not a specific value.
print("── Smoke test 1: anchor ──")
print(score_one_building(clean_buildings[0], models, ANCHOR_FEATURE_SCHEMA,
                         electricity, weather_all, meter_dfs))

# A foreign building — this is the real test of schema alignment.
print("\n── Smoke test 2: foreign building ──")
print(score_one_building(clean_buildings[1], models, ANCHOR_FEATURE_SCHEMA,
                         electricity, weather_all, meter_dfs))

# Smoke test: confirms the sweep harness runs and align_to_schema handles
# a foreign building (here 12 native features padded to the 15-feature
# anchor schema) before committing to the full 1,276-building sweep.
# Success = status='ok' and finite mae_pct for both models.


# ## Cross-building predictive sweep
# 
# Scores both the LSTM and TFT across all quality-filtered buildings in a single pass, using one shared prepared input per building so the two architectures are compared on byte-identical data. Produces the per-building MAE% table that Gate 2 filters for predictive parity.

# In[20]:


sweep_df = sweep_both(models, clean_buildings, ANCHOR_FEATURE_SCHEMA,
                      electricity, weather_all, meter_dfs)


# In[21]:


# =====================================================================
# STEP 9 — CELL: Sanity check against Phase 1. DO NOT SKIP.
# =====================================================================
# If the LSTM column does not land near Phase 1's distribution, something
# has diverged between the pipelines. Find out NOW, not after Gate 2 is
# built on top of it.

ok = sweep_df[sweep_df.status == 'ok']
print(f"Scored OK: {len(ok)}/{len(sweep_df)}")
print(f"Failures : {sweep_df[sweep_df.status != 'ok'].status.value_counts().to_dict()}")

print("\n── LSTM cross-building MAE% ──")
print(ok['mae_pct_lstm'].describe().round(1).to_string())
print("\nPhase 1 reference: median 40.2, mean 74.2, min 2.0, max 4345.6 (n=1294)")

print("\n── TFT cross-building MAE% ──")
print(ok['mae_pct_tft'].describe().round(1).to_string())

_med = ok['mae_pct_lstm'].median()
if not (35 <= _med <= 45):
    print(f"\n⚠️  LSTM median {_med:.1f}% is outside 35-45%. Phase 1 got 40.2%. "
          "Investigate before running Gate 2.")
else:
    print(f"\n✓ LSTM median {_med:.1f}% consistent with Phase 1 (40.2%)")


# # =====================================================================
# # STEP 10 — MARKDOWN CELL: Gate 2 pre-registration
# # INSERT AS MARKDOWN, BEFORE Cell B. Order matters: the registration
# # must visibly precede the computation in notebook order.
# # =====================================================================
# 
# ## Gate 2 — pre-registered predictive parity filter
#  
# **Registered:** 2026-07-17, before any TFT:LSTM ratio was computed.
#  
# **Gate 1 (registered 2026-07-14) is unchanged.** It asks only whether the
# TFT is competently trained on the anchor: test MAE% ≤ 26% (1.5× the anchor
# LSTM's 17.3%). It is a *training adequacy* check.
#  
# **Gate 2 is added, and its motivation is external to this work.** Bian et
# al. (*Energy and AI*, vol. 25, p. 100818, published June 2026) report a
# Transformer reference moving from +91.0% to −43.9% relative to an LSTM
# baseline depending on the target city, and conclude there is no uniformly
# dominant architecture. This indicates that the operative failure mode in
# cross-domain deployment is **target dependence**, not undertraining — a
# failure mode that a single anchor-level threshold cannot detect by
# construction.
#  
# Gate 2 therefore restricts the attribution comparison to buildings where
# both models are predictively comparable *on that building*:
#  
#     ratio = mae_pct_tft / mae_pct_lstm
#  
# - **Primary band:** 0.80 ≤ ratio ≤ 1.25 (neither model >25% worse)
# - **Sensitivity band:** 0.667 ≤ ratio ≤ 1.50 (matches Gate 1's 1.5×)
#  
# Both are reported. Agreement across bands is a robustness result;
# divergence is itself reportable.
#  
# Ratio bands are used rather than a percentage-difference rule because
# ratios are symmetric under inversion: TFT/LSTM = 1.25 failing implies
# LSTM/TFT = 1.25 fails identically. A "±30%" rule implemented as
# `abs(ratio − 1) ≤ 0.3` yields 0.7–1.3, which is asymmetric in log space.
#  
# **Buildings failing Gate 2 are analysed separately, not discarded.**
# "Attribution diverges precisely where predictive parity breaks down" is
# the cross-architecture analogue of the Phase 1 decoupling finding.
#  
# *Both bands were fixed before any ratio was inspected. This gate was
# motivated by literature published after Gate 1 was registered, not by
# inspection of our own results.*

# In[22]:


# =====================================================================
# STEP 11 — CELL B: Gate 2  (revised for the single-table sweep output)
# =====================================================================
# NOTE: Cell A v3 emits ONE table with mae_pct_lstm and mae_pct_tft
# columns, so no merge is needed — compute the ratio directly.

BAND_PRIMARY     = (0.80,  1.25)
BAND_SENSITIVITY = (0.667, 1.50)

parity = sweep_df[sweep_df.status == 'ok'].copy()
parity = parity[parity['mae_pct_lstm'].notna() & parity['mae_pct_tft'].notna()]
parity['ratio'] = parity['mae_pct_tft'] / parity['mae_pct_lstm']

# Reuse Phase 1's Type labels so Phase 1 and Phase 2 strata are identical.
parity = parity.merge(
    building_summary[['building', 'type']].rename(columns={'type': 'Type'}),
    on='building', how='left'
)


def apply_gate2(df, band, label):
    lo, hi = band
    col = f'pass_{label}'
    df[col] = df['ratio'].between(lo, hi)

    n_pass, n_tot = int(df[col].sum()), len(df)
    print(f"\n{'='*60}")
    print(f"Gate 2 [{label}]   band {lo:.3f} – {hi:.2f}")
    print(f"{'='*60}")
    print(f"  retained             : {n_pass}/{n_tot} ({100*n_pass/n_tot:.1f}%)")
    print(f"  excluded, TFT worse  : {int((df.ratio > hi).sum())}")
    print(f"  excluded, TFT better : {int((df.ratio < lo).sum())}")

    # READ THIS BEFORE ANY ATTRIBUTION RESULT.
    # If retention correlates with building type, the surviving sample is
    # type-biased and the stratified comparison inherits that bias.
    by_type = df.groupby('Type')[col].agg(n='count', retained='sum')
    by_type['retained_pct'] = (100 * by_type.retained / by_type.n).round(1)
    print("\n  Retention by building type:")
    print(by_type.sort_values('n', ascending=False).to_string())
    return df


parity = apply_gate2(parity, BAND_PRIMARY,     'primary')
parity = apply_gate2(parity, BAND_SENSITIVITY, 'sensitivity')

parity.to_csv(os.path.join(PHASE2_DIR, 'results', 'gate2_parity.csv'), index=False)
print(f"\n✓ Saved -> results/gate2_parity.csv")

print(f"\nRatio distribution:\n{parity['ratio'].describe().round(3).to_string()}")


# In[23]:


# =====================================================================
# ANALYSIS 1: Load saved models + run permutation sweep
# =====================================================================
# Loads the EXACT swept weights from disk (lstm_refit_bobcat.keras =
# the 14.56% run 1; tft_best_bobcat.keras = the 18.58% instance). No
# retraining -> provenance preserved, and no kernel restart needed.
#
# PYTHONHASHSEED: rather than relying on a kernel-level env var (which
# needs a restart to take effect), we make per-building seeds explicit
# and independent of Python's string hash. That removes the restart
# requirement entirely and is MORE reproducible than hash(building).
# ---------------------------------------------------------------------

ATTR_DIR = Path(PHASE2_DIR) / 'results'
ATTR_DIR.mkdir(parents=True, exist_ok=True)
ATTR_CSV = ATTR_DIR / 'analysis1_perm_both_models.csv'

N_PERM_REPEATS = 5   # matches Phase 1

# ── Load the saved swept models (NOT the in-memory ones) ─────────────
# The TFT uses custom layers (GRN, VariableSelectionNetwork,
# InterpretableMultiHeadAttention) that lack get_config/from_config and
# are not registered for serialization, so load_model() cannot rebuild
# the architecture from the .keras config. We therefore rebuild the
# skeleton with build_tft() using the EXACT Rank-2 config, then pour the
# saved weights in with load_weights(). This preserves provenance (same
# 18.58% weights) without depending on Keras custom-object deserialization.
#
# Requires cells 9/11/13/17 (GRN, VSN, attention, build_tft) to have run
# in this kernel. They define the classes the skeleton is built from.
#
# NOTE (reproducibility package TODO, NOT tonight): add get_config() to
# the three custom layers and decorate them with
# @keras.saving.register_keras_serializable(), then re-save both models
# once, so load_model() works cleanly for anyone opening the package.

# LSTM is all built-in layers -> loads directly.
lstm_swept = load_model(os.path.join(PHASE2_DIR, 'lstm_refit_bobcat.keras'),
                        compile=False)

# ── TFT: try direct load first (Option 1), fall back to weights (Option 2)
TFT_PATH = os.path.join(PHASE2_DIR, 'tft_best_bobcat.keras')
try:
    custom = {
        'GRN': GRN,
        'VariableSelectionNetwork': VariableSelectionNetwork,
        'InterpretableMultiHeadAttention': InterpretableMultiHeadAttention,
    }
    tft_swept = load_model(TFT_PATH, custom_objects=custom, compile=False)
    print("TFT loaded via load_model + custom_objects (Option 1).")
except Exception as e:
    print(f"Option 1 failed ({type(e).__name__}); rebuilding skeleton + load_weights (Option 2).")
    # Rebuild the EXACT Rank-2 architecture, then load weights only.
    tft_swept, _ = build_tft(
        seq_len=WINDOW_SIZE,
        n_features=N_ANCHOR_FEATURES,     # = 15
        d_model=64, n_heads=4,
        lstm_units=(64, 32), dropout=0.2,
    )
    tft_swept.load_weights(TFT_PATH)
    print("TFT rebuilt and weights loaded (Option 2).")

attr_models = {'lstm': lstm_swept, 'tft': tft_swept}
print("\nLoaded swept models:")
print("  lstm_refit_bobcat.keras (run 1, 14.56%)")
print("  tft_best_bobcat.keras   (18.58% instance)")

# ── Provenance sanity check: confirm the loaded TFT reproduces 18.58% ─
# If load_weights silently loaded into a mismatched skeleton, the anchor
# MAE% would be wrong. This catches that before the 6+ hour sweep.
_Xa, _ya, _sy, _fc, _ = prepare_building(
    ANCHOR_BUILDING, electricity, weather_all, meter_dfs)
_Xal = align_to_schema(_Xa, _fc, ANCHOR_FEATURE_SCHEMA)
_pred = _sy.inverse_transform(tft_swept.predict(_Xal, verbose=0))
_true = _sy.inverse_transform(_ya.reshape(-1, 1))
_anchor_full_mae_pct = 100 * mean_absolute_error(_true, _pred) / _true.mean()
print(f"\nLoaded-TFT anchor MAE% (full-year): {_anchor_full_mae_pct:.2f}%")
print("  (sweep uses full-year like Phase 1; the 18.58% was TEST-partition,")
print("   so these differ — this check only confirms the weights loaded sanely,")
print("   i.e. the number should be low/plausible, not garbage.)")


def stable_seed(building, base=SEED):
    """Deterministic per-building seed, independent of PYTHONHASHSEED.
    md5 of the name -> no reliance on Python's randomised string hash,
    so it reproduces across sessions with NO kernel restart needed."""
    h = int(hashlib.md5(building.encode()).hexdigest()[:8], 16)
    return base + (h % 100000)


# =====================================================================
# ANALYSIS 1.1: Permutation importance (memory-growth + sequence cap)
# =====================================================================
# The 8192-batch OOM was not accumulation (tf.function fixed that) — the
# card genuinely can't hold a large forward pass of these models plus the
# two loaded models plus TF's pool. Two structural fixes:
#   (1) GPU memory growth (above) — TF takes only what it needs.
#   (2) MAX_SEQ cap — bound the sequences per building. Permutation
#       importance is a RANKING; a few-thousand-sequence estimate of the
#       ranking is stable. This bounds BOTH memory and runtime and is a
#       principled, documentable choice — not a workaround.
# ---------------------------------------------------------------------



MAX_SEQ       = 3000    # sequences per building for permutation (cap)
PREDICT_BATCH = 3000    # one pass fits comfortably now; == MAX_SEQ

_infer_fns = {}

def _get_infer_fn(name, model):
    if name not in _infer_fns:
        @tf.function(reduce_retracing=True,
                     input_signature=[tf.TensorSpec(
                         shape=[None, WINDOW_SIZE, N_ANCHOR_FEATURES],
                         dtype=tf.float32)])
        def _fn(x):
            return model(x, training=False)
        _infer_fns[name] = _fn
    return _infer_fns[name]


def _predict_fast(infer_fn, X):
    outs = []
    for s in range(0, len(X), PREDICT_BATCH):
        chunk = tf.convert_to_tensor(X[s:s+PREDICT_BATCH], dtype=tf.float32)
        outs.append(infer_fn(chunk).numpy())
    return np.concatenate(outs, axis=0)


def permutation_importance(infer_fn, X_aligned, y_true_kwh, scaler_y_b,
                           n_features, rng, n_repeats=N_PERM_REPEATS):
    baseline = mean_absolute_error(
        y_true_kwh, scaler_y_b.inverse_transform(_predict_fast(infer_fn, X_aligned)))
    deltas = np.zeros(n_features)
    for fi in range(n_features):
        d = []
        for _ in range(n_repeats):
            X_p = X_aligned.copy()
            perm = rng.permutation(len(X_p))
            X_p[:, :, fi] = X_p[perm, :, fi]
            p_mae = mean_absolute_error(
                y_true_kwh, scaler_y_b.inverse_transform(_predict_fast(infer_fn, X_p)))
            d.append(p_mae - baseline)
        deltas[fi] = np.mean(d)
    return deltas


def attribute_one_building(building, models, schema,
                           electricity, weather_all, meter_dfs):
    row = {'building': building}
    try:
        X_seq_b, y_seq_b, scaler_y_b, feat_cols_b, _ = prepare_building(
            building, electricity, weather_all, meter_dfs)
    except Exception as e:
        row['status'] = f'prep_failed: {type(e).__name__}'
        return row
    if X_seq_b is None or len(X_seq_b) == 0:
        row['status'] = 'no_sequences'
        return row

    X_aligned  = align_to_schema(X_seq_b, feat_cols_b, schema)
    y_true_kwh = scaler_y_b.inverse_transform(y_seq_b.reshape(-1, 1))

    # ── SEQUENCE CAP ─────────────────────────────────────────────────
    # Deterministic subsample per building (seeded), so it reproduces.
    # Record how many sequences were actually used for transparency.
    n_full = len(X_aligned)
    if n_full > MAX_SEQ:
        cap_rng = np.random.default_rng(stable_seed(building))
        idx = np.sort(cap_rng.choice(n_full, MAX_SEQ, replace=False))
        X_aligned  = X_aligned[idx]
        y_true_kwh = y_true_kwh[idx]
    row['status']   = 'ok'
    row['n_seq_full'] = n_full
    row['n_seq_used'] = len(X_aligned)

    for name, model in models.items():
        infer_fn = _get_infer_fn(name, model)
        rng = np.random.default_rng(stable_seed(building))
        deltas = permutation_importance(
            infer_fn, X_aligned, y_true_kwh, scaler_y_b, len(schema), rng)
        for fi, feat in enumerate(schema):
            row[f'perm_{name}_{feat}'] = deltas[fi]

    del X_seq_b, y_seq_b, X_aligned, y_true_kwh
    gc.collect()
    return row

# ── DOCUMENTATION NOTE for the report ────────────────────────────────
# "Permutation importance was computed on a deterministic per-building
#  subsample of MAX_SEQ=3000 sequences (or all sequences where fewer
#  were available), seeded per building for reproducibility. Permutation
#  importance is a rank statistic and is stable under subsampling at this
#  scale; n_seq_full and n_seq_used are recorded per building."
#
# OPTIONAL VALIDATION (do once, on ~10 buildings): run with MAX_SEQ=3000
# and MAX_SEQ=8000 and confirm the collapsed top-1 ranking is unchanged.
# That demonstrates the cap does not alter the finding. Cheap, and it
# pre-empts "did subsampling change your result?" -> answer: "we checked."


def run_attribution_sweep(models, buildings, schema,
                          electricity, weather_all, meter_dfs,
                          out_path=ATTR_CSV, resume=True):
    done = set()
    if resume and out_path.exists():
        done = set(pd.read_csv(out_path)['building'])
        print(f"Resuming: {len(done)} buildings already attributed.")
    todo  = [b for b in buildings if b not in done]
    first = not out_path.exists()
    for i, b in enumerate(todo):
        row = attribute_one_building(b, models, schema,
                                     electricity, weather_all, meter_dfs)
        pd.DataFrame([row]).to_csv(out_path, mode='a', header=first, index=False)
        first = False
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(todo)}  ({b})")
    print(f"Attribution sweep complete -> {out_path}")
    return pd.read_csv(out_path)


# ── Gate-2 primary retained set ──────────────────────────────────────
retained = parity[parity['pass_primary']]['building'].tolist()
print(f"\nGate-2 primary retained: {len(retained)} buildings")

# ── RUN IT (resumes if interrupted) ───────────
attr_df = run_attribution_sweep(attr_models, retained, ANCHOR_FEATURE_SCHEMA,
                                electricity, weather_all, meter_dfs)
print(f"\nattr_df ready: {len(attr_df)} rows, "
      f"{(attr_df.status=='ok').sum()} ok")


# In[32]:


# =====================================================================
# ANALYSIS 1.2: COLLAPSE HELPER (schema-derived, not hardcoded)
# =====================================================================
# Phase 1 hardcoded single_feats = [...'gas'] with only ONE meter. Your
# Phase 2 anchor schema has FOUR meters (chilledwater, gas, hotwater,
# solar). Copying Phase 1's list verbatim would silently DROP three
# meters from the collapsed vector. Derive singles from the schema
# instead so it is correct for whatever the anchor schema actually is.
#
# Collapse rule (matches Phase 1 intent): SUM each sin/cos pair into one
# cyclical feature; keep every non-paired feature as-is. Applied BEFORE
# any ranking, top-1, top-3 or Spearman — sin/cos are two encodings of
# one variable and must be combined before attribution is compared.
# ---------------------------------------------------------------------

PAIR_MAP = {
    'Hour of Day' : ('hour_sin',  'hour_cos'),
    'Day of Week' : ('dow_sin',   'dow_cos'),
    'Month'       : ('month_sin', 'month_cos'),
}

def build_collapse_index(schema):
    """Return (collapsed_labels, collapse_fn) for a given schema.
    collapse_fn maps a length-len(schema) vector to a length-len(labels)
    collapsed vector, summing sin/cos pairs and keeping singles."""
    paired_feats = set()
    labels = []
    ops    = []   # each op: ('sum', i, j) or ('keep', i)

    for label, (s, c) in PAIR_MAP.items():
        if s in schema and c in schema:
            labels.append(label)
            ops.append(('sum', schema.index(s), schema.index(c)))
            paired_feats.update([s, c])

    # every feature NOT part of a collapsed pair is kept as a single,
    # in schema order — this automatically includes ALL meters present.
    for i, f in enumerate(schema):
        if f not in paired_feats:
            labels.append(f)
            ops.append(('keep', i))

    def collapse_fn(arr):
        out = np.empty(len(ops))
        for k, op in enumerate(ops):
            out[k] = arr[op[1]] + arr[op[2]] if op[0] == 'sum' else arr[op[1]]
        return out

    return labels, collapse_fn


COLLAPSED_LABELS, collapse_vec = build_collapse_index(ANCHOR_FEATURE_SCHEMA)
print(f"Collapsed feature space ({len(COLLAPSED_LABELS)}): {COLLAPSED_LABELS}")
# Expect 12 labels for the 15-feature anchor:
#   Hour of Day, Day of Week, Month  (3 collapsed pairs)
#   + airTemperature, dewTemperature, windDirection, windSpeed, is_weekend
#   + chilledwater, gas, hotwater, solar   (9 singles)

HOD_LABEL = 'Hour of Day'   # exact label, no more startswith() guessing


# ── STALE-9 replication collapse (reproduces the INTERIM REPORT) ──────
# Phase 1's collapse_pairs iterated a HARDCODED single_feats list that
# named only ONE meter ('gas'), silently dropping chilledwater, hotwater
# and solar from the ranked feature set. The interim report's per-type
# Hour-of-Day rates (Healthcare 76.9%, Entertainment 22.7%, ...) were
# therefore computed over 9 collapsed features, not 12.
#
# We replicate that EXACTLY here, for one purpose only: to reconcile the
# corrected-12 numbers against the interim report and quantify the change
# the omission caused. The corrected-12 collapse above is the analysis of
# record; stale-9 is diagnostic only.
PHASE1_STALE_SINGLE_FEATS = [
    'airTemperature', 'dewTemperature', 'windDirection',
    'windSpeed', 'is_weekend', 'gas',            # NOTE: no chilledwater/hotwater/solar
]

def build_stale9_collapse(schema):
    labels, ops, paired = [], [], set()
    for label, (s, c) in PAIR_MAP.items():
        if s in schema and c in schema:
            labels.append(label); ops.append(('sum', schema.index(s), schema.index(c)))
            paired.update([s, c])
    for f in PHASE1_STALE_SINGLE_FEATS:          # hardcoded list, NOT schema-derived
        if f in schema:
            labels.append(f); ops.append(('keep', schema.index(f)))
    def fn(arr):
        out = np.empty(len(ops))
        for k, op in enumerate(ops):
            out[k] = arr[op[1]] + arr[op[2]] if op[0] == 'sum' else arr[op[1]]
        return out
    return labels, fn

STALE9_LABELS, collapse_vec_stale9 = build_stale9_collapse(ANCHOR_FEATURE_SCHEMA)
print(f"Stale-9 replication space ({len(STALE9_LABELS)}): {STALE9_LABELS}")
assert HOD_LABEL in STALE9_LABELS


def hod_top1_rate_by_type(attr_df, model, collapse_fn, labels):
    """Per-type Hour-of-Day top-1 rate under a given collapse. Returns a
    dict type -> pct. Used to compare corrected-12 vs stale-9 vs interim."""
    cols = [f'perm_{model}_{f}' for f in ANCHOR_FEATURE_SCHEMA]
    hod_i = labels.index(HOD_LABEL)
    a = attr_df[attr_df.status == 'ok'].merge(
        building_summary[['building','type']], on='building', how='left').dropna(subset=['type'])
    out = {}
    for typ, g in a.groupby('type'):
        t1 = [int(np.argmax(collapse_fn(row[cols].to_numpy(dtype=float))))
              for _, row in g.iterrows()]
        out[typ] = (100 * np.mean([t == hod_i for t in t1]), len(g))
    return out


def reconcile_collapse(attr_df, model='lstm'):
    """Side-by-side: corrected-12 vs stale-9 HoD top-1 rate per type.
    Run this on the LSTM FIRST. The stale-9 column should reproduce the
    interim report; the corrected-12 column is the figure of record."""
    c12 = hod_top1_rate_by_type(attr_df, model, collapse_vec,        COLLAPSED_LABELS)
    s9  = hod_top1_rate_by_type(attr_df, model, collapse_vec_stale9, STALE9_LABELS)
    rows = []
    for typ in sorted(c12, key=lambda t: -c12[t][1]):
        pct12, n = c12[typ]
        pct9, _  = s9[typ]
        rows.append({'Type': typ, 'n': n,
                     'HoD_top1_corrected12': round(pct12, 1),
                     'HoD_top1_stale9':      round(pct9, 1),
                     'delta_pp':             round(pct12 - pct9, 1)})
    df = pd.DataFrame(rows)
    print(f"\n── Collapse reconciliation ({model.upper()}) ──")
    print("stale9 should reproduce the interim report; corrected12 is the figure of record")
    print(df.to_string(index=False))
    return df

# AFTER attr_df exists, BEFORE trusting anything:
recon = reconcile_collapse(attr_df, model='lstm')
recon.to_csv(ATTR_DIR / 'analysis1_collapse_reconciliation.csv', index=False)
# Expect stale9 Healthcare ~76.9%, Entertainment ~22.7% (interim match).
# The delta_pp column IS the documented improvement.


# In[28]:


# =====================================================================
# ANALYSIS 1.3: rank correlation on COLLAPSED vectors
# =====================================================================

def rank_agreement(df, schema):
    lstm_cols = [f'perm_lstm_{f}' for f in schema]
    tft_cols  = [f'perm_tft_{f}'  for f in schema]
    out = []
    for _, r in df[df.status == 'ok'].iterrows():
        v_lstm = collapse_vec(r[lstm_cols].to_numpy(dtype=float))
        v_tft  = collapse_vec(r[tft_cols ].to_numpy(dtype=float))
        rho, _ = spearmanr(v_lstm, v_tft)           # over 12 collapsed feats
        t1_lstm = COLLAPSED_LABELS[int(np.argmax(v_lstm))]
        t1_tft  = COLLAPSED_LABELS[int(np.argmax(v_tft))]
        s3_lstm = {COLLAPSED_LABELS[i] for i in np.argsort(v_lstm)[-3:]}
        s3_tft  = {COLLAPSED_LABELS[i] for i in np.argsort(v_tft)[-3:]}
        out.append({
            'building': r['building'],
            'spearman': rho,
            'top1_match': t1_lstm == t1_tft,
            'top1_lstm': t1_lstm, 'top1_tft': t1_tft,
            'top3_overlap': len(s3_lstm & s3_tft),
        })
    return pd.DataFrame(out)


# In[29]:


# =====================================================================
# ANALYSIS 1.4: stratification by building type on COLLAPSED vectors
# =====================================================================
def stratified_summary(agreement, attr_df, schema, min_n=30):
    lstm_cols = [f'perm_lstm_{f}' for f in schema]
    tft_cols  = [f'perm_tft_{f}'  for f in schema]
    a = attr_df[attr_df.status == 'ok'].merge(
        building_summary[['building','type']], on='building', how='left')
    hod_idx = COLLAPSED_LABELS.index(HOD_LABEL)

    rows = []
    for typ, g in a.groupby('type'):
        gi = agreement[agreement.Type == typ]
        lstm_t1 = [int(np.argmax(collapse_vec(row[lstm_cols].to_numpy(dtype=float))))
                   for _, row in g.iterrows()]
        tft_t1  = [int(np.argmax(collapse_vec(row[tft_cols ].to_numpy(dtype=float))))
                   for _, row in g.iterrows()]
        rows.append({
            'Type': typ, 'n': len(g),
            'median_spearman':   gi['spearman'].median() if len(gi) else np.nan,
            'lstm_hod_top1_pct': 100*np.mean([t == hod_idx for t in lstm_t1]),
            'tft_hod_top1_pct':  100*np.mean([t == hod_idx for t in tft_t1]),
            'top1_arch_agree_pct': 100*gi['top1_match'].mean() if len(gi) else np.nan,
        })
    df = pd.DataFrame(rows).sort_values('n', ascending=False)
    df['flag_small_n'] = df['n'] < min_n
    return df


# In[30]:


# =====================================================================
# ANALYSIS 1.5: Label-permutation null, collapsed
# =====================================================================
def stratification_pvalue(attr_df, schema, model='lstm',
                          n_permutations=10000, min_n=30, seed=SEED):
    cols = [f'perm_{model}_{f}' for f in schema]
    a = attr_df[attr_df.status == 'ok'].merge(
        building_summary[['building','type']], on='building', how='left').dropna(subset=['type'])
    hod_idx = COLLAPSED_LABELS.index(HOD_LABEL)

    top1 = np.array([int(np.argmax(collapse_vec(row[cols].to_numpy(dtype=float))))
                     for _, row in a.iterrows()])
    hod_hit = (top1 == hod_idx).astype(float)
    types   = a['type'].to_numpy()

    vc = pd.Series(types).value_counts()
    keep = vc[vc >= min_n].index
    mask = np.isin(types, keep)
    hod_hit, types = hod_hit[mask], types[mask]

    def between_type_var(labels):
        return np.var([hod_hit[labels == t].mean() for t in np.unique(labels)])

    observed = between_type_var(types)
    rng  = np.random.default_rng(seed)
    null = np.array([between_type_var(rng.permutation(types))
                     for _ in range(n_permutations)])
    p = (np.sum(null >= observed) + 1) / (n_permutations + 1)
    return {'model': model, 'observed_var': observed, 'null_mean': null.mean(),
            'null_95pct': np.percentile(null, 95), 'p_value': p,
            'n_types_tested': len(keep), 'types_tested': list(keep)}

# ── Sanity check the collapse reproduces Phase 1 before trusting it ──
# After attr_df exists, verify the LSTM HoD-top1 rates by type match
# the interim report (Healthcare 76.9%, Entertainment 22.7%, etc).
# If they DON'T match, the collapse or the permutation diverged from
# Phase 1 and must be reconciled BEFORE reporting anything.



# In[35]:


# =====================================================================
# ANALYSIS 1b — WEIGHT-DRAW VARIANCE DECOMPOSITION
# =====================================================================
# Measures within-arch (noise floor) vs between-arch attribution
# agreement, to test whether cross-architecture invariance exceeds
# draw-to-draw noise. STAGED: subsample first (headline direction),
# full sweep later (bulletproof).
# ---------------------------------------------------------------------

# ── CELL 1: Train + save 3 draws per architecture ────────────────────
# Draw 1 of each already exists on disk from the main run:
#   lstm_refit_bobcat.keras  (14.56%)  -> treat as LSTM draw 1
#   tft_best_bobcat.keras     (18.58%) -> treat as TFT  draw 1
# We train 2 MORE of each. Different seeds so the draws are genuinely
# independent (NOT relying on GPU non-determinism alone — we WANT
# initialisation variance here, that's the thing being measured).

DRAWS_DIR = os.path.join(PHASE2_DIR, 'draws')
os.makedirs(DRAWS_DIR, exist_ok=True)

def train_lstm_draw(seed):
    tf.random.set_seed(seed); np.random.seed(seed)
    inp = Input(shape=(WINDOW_SIZE, N_ANCHOR_FEATURES))
    x = LSTM(64, return_sequences=True,  dropout=0.2)(inp)
    x = LSTM(32, return_sequences=False, dropout=0.2)(x)
    out = Dense(1, activation='linear')(x)
    m = Model(inp, out)
    m.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    es = EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss')
    m.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=300, batch_size=32, callbacks=[es], verbose=0)
    return m

def train_tft_draw(seed):
    tf.random.set_seed(seed); np.random.seed(seed)
    m, _ = build_tft(seq_len=WINDOW_SIZE, n_features=N_ANCHOR_FEATURES,
                     d_model=64, n_heads=4, lstm_units=(64,32), dropout=0.2)
    m.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    es = EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss')
    m.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=300, batch_size=64, callbacks=[es], verbose=0)
    return m

def anchor_test_mae_pct(model):
    pred = scaler_y.inverse_transform(model.predict(X_test, verbose=0)).flatten()
    true = scaler_y.inverse_transform(y_test.reshape(-1,1)).flatten()
    return 100*mean_absolute_error(true, pred)/true.mean()

# Draw 1 = existing saved models (symlink/copy into the draws dir for uniformity)
shutil.copy(os.path.join(PHASE2_DIR,'lstm_refit_bobcat.keras'), os.path.join(DRAWS_DIR,'lstm_draw1.keras'))
shutil.copy(os.path.join(PHASE2_DIR,'tft_best_bobcat.keras'),   os.path.join(DRAWS_DIR,'tft_draw1.keras'))

# Draws 2 and 3 — distinct seeds
for d, seed in [(2, SEED+1), (3, SEED+2)]:
    lm = train_lstm_draw(seed)
    lm.save(os.path.join(DRAWS_DIR, f'lstm_draw{d}.keras'))
    print(f"LSTM draw {d} (seed {seed}): anchor test MAE% = {anchor_test_mae_pct(lm):.2f}%")
    del lm; gc.collect()

    tm = train_tft_draw(seed)
    tm.save(os.path.join(DRAWS_DIR, f'tft_draw{d}.keras'))
    print(f"TFT  draw {d} (seed {seed}): anchor test MAE% = {anchor_test_mae_pct(tm):.2f}%")
    del tm; gc.collect()

print("\n✓ 3 draws per architecture saved to draws/")
# Record the 6 anchor MAE%s in the notes — the SPREAD across draws is
# itself the predictive-instability baseline that motivates this analysis.


# In[36]:


# ── CELL 2: Stratified subsample for the within-arch floors ──────────
# The noise floor is a DISTRIBUTIONAL quantity — it does not need all
# 1,276 buildings. Take ~40 per major type (>= n_min), all of small types.
SUBSAMPLE_PER_TYPE = 40
TYPE_MIN_FOR_CAP   = 40   # types with >= this many get subsampled to 40

sub_rng = np.random.default_rng(SEED)
retained_types = building_summary[building_summary.building.isin(retained)]
subsample = []
for typ, g in retained_types.groupby('type'):
    ids = g['building'].tolist()
    if len(ids) > TYPE_MIN_FOR_CAP:
        subsample += list(sub_rng.choice(ids, SUBSAMPLE_PER_TYPE, replace=False))
    else:
        subsample += ids
print(f"Subsample for within-arch floors: {len(subsample)} buildings "
      f"across {retained_types['type'].nunique()} types")


# In[37]:


# ── CELL 3: Attribution across ALL 6 draws on the subsample ──────────
# Reuses the leak-free / capped permutation_importance from the main
# Analysis 1 cell. Loads each draw, sweeps the subsample, appends.
VAR_CSV = os.path.join(PHASE2_DIR, 'results', 'analysis1b_variance_subsample.csv')

DRAW_MODELS = {}
custom = {'GRN': GRN, 'VariableSelectionNetwork': VariableSelectionNetwork,
          'InterpretableMultiHeadAttention': InterpretableMultiHeadAttention}
for arch in ['lstm', 'tft']:
    for d in [1, 2, 3]:
        key = f'{arch}_draw{d}'
        path = os.path.join(DRAWS_DIR, f'{key}.keras')
        DRAW_MODELS[key] = load_model(path, compile=False,
                                      custom_objects=(custom if arch=='tft' else None))
print(f"Loaded {len(DRAW_MODELS)} draw-models")

# _infer_fns is keyed by name -> clear it so each draw gets its own fn
_infer_fns.clear()

def attribute_subsample(draw_models, buildings, schema,
                        electricity, weather_all, meter_dfs, out_path):
    first = not os.path.exists(out_path)
    done = set(pd.read_csv(out_path)['building']) if os.path.exists(out_path) else set()
    for i, b in enumerate([x for x in buildings if x not in done]):
        row = {'building': b}
        try:
            X_seq_b, y_seq_b, scaler_y_b, feat_cols_b, _ = prepare_building(
                b, electricity, weather_all, meter_dfs)
            X_aligned = align_to_schema(X_seq_b, feat_cols_b, schema)
            y_true = scaler_y_b.inverse_transform(y_seq_b.reshape(-1,1))
            n_full = len(X_aligned)
            if n_full > MAX_SEQ:
                cr = np.random.default_rng(stable_seed(b))
                idx = np.sort(cr.choice(n_full, MAX_SEQ, replace=False))
                X_aligned, y_true = X_aligned[idx], y_true[idx]
            row['status'] = 'ok'
            for key, model in draw_models.items():
                fn = _get_infer_fn(key, model)
                rng = np.random.default_rng(stable_seed(b))
                deltas = permutation_importance(fn, X_aligned, y_true, scaler_y_b,
                                                len(schema), rng)
                for fi, feat in enumerate(schema):
                    row[f'perm_{key}_{feat}'] = deltas[fi]
            del X_seq_b, y_seq_b, X_aligned, y_true; gc.collect()
        except Exception as e:
            row['status'] = f'failed: {type(e).__name__}'
        pd.DataFrame([row]).to_csv(out_path, mode='a', header=first, index=False)
        first = False
        if (i+1) % 20 == 0:
            print(f"  {i+1} ({b})")
    return pd.read_csv(out_path)

var_df = attribute_subsample(DRAW_MODELS, subsample, ANCHOR_FEATURE_SCHEMA,
                             electricity, weather_all, meter_dfs, VAR_CSV)



# In[38]:


# ── CELL 4: The decomposition — within vs between ────────────────────
def collapse_cols(row, prefix, schema):
    return collapse_vec(np.array([row[f'{prefix}_{f}'] for f in schema], dtype=float))

def decomposition(var_df, schema):
    ok = var_df[var_df.status == 'ok']
    lstm_keys = ['lstm_draw1','lstm_draw2','lstm_draw3']
    tft_keys  = ['tft_draw1','tft_draw2','tft_draw3']

    def pair_rhos(keys_a, keys_b, same):
        rhos = []
        pairs = combinations(keys_a, 2) if same else [(a,b) for a in keys_a for b in keys_b]
        for ka, kb in pairs:
            for _, r in ok.iterrows():
                va = collapse_cols(r, f'perm_{ka}', schema)
                vb = collapse_cols(r, f'perm_{kb}', schema)
                rho, _ = spearmanr(va, vb)
                if not np.isnan(rho): rhos.append(rho)
        return np.array(rhos)

    within_lstm = pair_rhos(lstm_keys, lstm_keys, same=True)
    within_tft  = pair_rhos(tft_keys,  tft_keys,  same=True)
    between     = pair_rhos(lstm_keys, tft_keys,  same=False)

    print("── Attribution agreement (median Spearman, collapsed-12) ──")
    print(f"  within-LSTM  : {np.median(within_lstm):.3f}  "
          f"[{np.percentile(within_lstm,25):.3f}, {np.percentile(within_lstm,75):.3f}]")
    print(f"  within-TFT   : {np.median(within_tft):.3f}  "
          f"[{np.percentile(within_tft,25):.3f}, {np.percentile(within_tft,75):.3f}]")
    print(f"  between-arch : {np.median(between):.3f}  "
          f"[{np.percentile(between,25):.3f}, {np.percentile(between,75):.3f}]")
    print()
    wl, wt, bt = np.median(within_lstm), np.median(within_tft), np.median(between)
    floor = min(wl, wt)
    if bt >= floor - 0.05:
        print("  => between ~ within: STRONG invariance. Crossing architecture is")
        print("     no worse than re-drawing the same one.")
    elif bt < floor - 0.15:
        print("  => between << within: invariance FAILS. Architecture introduces")
        print("     disagreement beyond draw noise.")
    else:
        print("  => between modestly below within: PARTIAL invariance. Report the gap.")
    if floor < 0.5:
        print(f"  !! within-arch floor is LOW ({floor:.3f}): attribution is unstable")
        print("     even within one architecture. The instability IS the finding.")
    return within_lstm, within_tft, between

w_lstm, w_tft, betw = decomposition(var_df, ANCHOR_FEATURE_SCHEMA)


# In[39]:


# =====================================================================
# ANALYSIS 1c — CELL 5: Paired significance test on the decomposition
# =====================================================================
# Replaces the arbitrary "< floor - 0.15" verdict with an actual test.
#
# WHY PAIRED: the same buildings appear in all three comparisons, so
# per-building rho values are paired observations. A paired test is far
# more powerful than comparing two independent distributions, and it
# controls for the fact that some buildings are intrinsically harder to
# attribute consistently than others (a building with a flat importance
# profile will have low rho in EVERY comparison).
#
# Wilcoxon signed-rank rather than paired t: rho values are bounded
# [-1,1], skewed, and not normal. Non-parametric is the honest choice.
# ---------------------------------------------------------------------

def per_building_rhos(var_df, schema):
    """Return a tidy frame: one row per building, columns for the mean
    within-LSTM, within-TFT and between-arch rho for THAT building.
    Averaging the pairwise rhos within each category first is what makes
    the comparison paired at the building level."""
    ok = var_df[var_df.status == 'ok']
    lstm_keys = ['lstm_draw1','lstm_draw2','lstm_draw3']
    tft_keys  = ['tft_draw1','tft_draw2','tft_draw3']

    def vec(row, key):
        return collapse_vec(np.array(
            [row[f'perm_{key}_{f}'] for f in schema], dtype=float))

    rows = []
    for _, r in ok.iterrows():
        v = {k: vec(r, k) for k in lstm_keys + tft_keys}

        def mean_rho(pairs):
            rs = [spearmanr(v[a], v[b])[0] for a, b in pairs]
            rs = [x for x in rs if not np.isnan(x)]
            return np.mean(rs) if rs else np.nan

        rows.append({
            'building':     r['building'],
            'within_lstm':  mean_rho(list(combinations(lstm_keys, 2))),
            'within_tft':   mean_rho(list(combinations(tft_keys, 2))),
            'between':      mean_rho([(a, b) for a in lstm_keys for b in tft_keys]),
        })
    return pd.DataFrame(rows).dropna()


def test_decomposition(var_df, schema, building_summary=None):
    pb = per_building_rhos(var_df, schema)
    n = len(pb)
    print(f"Paired per-building comparison, n = {n} buildings\n")

    print("Median (IQR) per-building mean rho:")
    for c in ['within_lstm', 'within_tft', 'between']:
        q1, med, q3 = pb[c].quantile([.25, .5, .75])
        print(f"  {c:12s}: {med:.3f}  [{q1:.3f}, {q3:.3f}]")

    print("\nPaired Wilcoxon signed-rank tests:")
    tests = [
        ('between vs within_lstm', 'between', 'within_lstm'),
        ('between vs within_tft',  'between', 'within_tft'),
        ('within_tft vs within_lstm', 'within_tft', 'within_lstm'),
    ]
    results = {}
    for label, a, b in tests:
        diff = pb[a] - pb[b]
        stat, p = wilcoxon(pb[a], pb[b])
        # Effect size: matched-pairs rank-biserial correlation.
        # Interpretable as "proportion of pairs favouring a, minus
        # proportion favouring b" — a real effect size, not just a p.
        n_nz = (diff != 0).sum()
        rb = 1 - (2 * stat) / (n_nz * (n_nz + 1) / 2) if n_nz else np.nan
        pct_lower = 100 * (diff < 0).mean()
        print(f"  {label:28s} median diff {diff.median():+.3f}  "
              f"p = {p:.3g}  rank-biserial {rb:+.3f}  "
              f"({pct_lower:.0f}% of buildings lower)")
        results[label] = {'p': p, 'median_diff': diff.median(), 'rb': rb}

    print("\nINTERPRETATION GUIDE — read the EFFECT SIZE, not just p:")
    print("  With n in the hundreds, trivially small gaps reach p<0.05.")
    print("  The question is whether the GAP MATTERS, not whether it is")
    print("  non-zero. A median diff of -0.05 is significant and useless;")
    print("  a median diff of -0.20 is a real architecture effect.")

    # Practical-significance flag, stated as a judgement not a rule
    d = results['between vs within_tft']['median_diff']
    if results['between vs within_tft']['p'] < 0.05 and abs(d) >= 0.10:
        print(f"\n  => between-arch agreement is significantly AND materially")
        print(f"     lower than within-TFT (median diff {d:+.3f}).")
    elif results['between vs within_tft']['p'] < 0.05:
        print(f"\n  => statistically lower but the gap is small ({d:+.3f}).")
        print("     Report as 'detectably but modestly lower'.")
    else:
        print(f"\n  => NOT significantly different. Between-architecture")
        print("     agreement is indistinguishable from within-TFT noise.")

    # Per-type breakdown — is the instability itself type-dependent?
    if building_summary is not None:
        pb = pb.merge(building_summary[['building','type']],
                      on='building', how='left')
        print("\nPer-type median rho (is instability itself stratified?):")
        agg = pb.groupby('type')[['within_lstm','within_tft','between']].median().round(3)
        agg['n'] = pb.groupby('type').size()
        print(agg.sort_values('n', ascending=False).to_string())

    return pb, results

pb, res = test_decomposition(var_df, ANCHOR_FEATURE_SCHEMA, building_summary)
pb.to_csv(os.path.join(PHASE2_DIR,'results','analysis1c_per_building_rho.csv'), index=False)


# In[41]:


# =====================================================================
# ANALYSIS 1d — CAP VALIDATION: estimation noise vs model instability
# =====================================================================
# THE QUESTION THIS ANSWERS — and it is the pivotal one:
#
#   within-LSTM rho = 0.775 means two identically-specified LSTMs
#   disagree on feature importance. But HOW MUCH of that disagreement is
#   real model instability, and how much is just NOISE IN OUR ESTIMATE
#   of each model's attribution?
#
#   Permutation importance on 3000 subsampled sequences with 5 repeats
#   is itself a noisy estimator. If we permuted the SAME model TWICE
#   with different subsamples, we would get slightly different vectors.
#   That is estimation noise, not instability.
#
#   => The honest noise floor is NOT 1.0. It is rho_same-model, and
#      within-LSTM rho must be compared against THAT, not against 1.0.
#
# THREE MEASUREMENTS, on the same buildings:
#   A. rho_same_model_same_cap  : one model, two independent 3000-seq
#                                 draws + different shuffle seeds.
#                                 = PURE ESTIMATION NOISE CEILING.
#   B. rho_same_model_bigger_cap: one model, 3000-cap vs 8000-cap.
#                                 = does the cap bias the estimate?
#   C. within-LSTM (from Cell 4) : two models. Already have it.
#
# INTERPRETATION:
#   If A ~ 0.95 and C = 0.775  -> the gap is REAL model instability. The
#                                 headline finding stands.
#   If A ~ 0.78 and C = 0.775  -> we have measured our own estimator's
#                                 noise, NOT model instability. The
#                                 headline finding COLLAPSES and must be
#                                 rewritten. This is the outcome that
#                                 would invalidate the current story, so
#                                 it is exactly the one worth testing.
#   If B << A                  -> the 3000 cap biases attribution;
#                                 raise the cap or report the sensitivity.
# ---------------------------------------------------------------------

CAP_TEST_N   = 25      # buildings; enough for a median, cheap to run
CAP_BIG      = 8000    # the "more data" comparison
CAP_SMALL    = MAX_SEQ # 3000

def _perm_with_cap(model_key, model, building, schema, cap, seed_offset,
                   electricity, weather_all, meter_dfs):
    """Permutation vector for one building at a given cap, with an
    independent subsample AND independent shuffle stream."""
    X_seq_b, y_seq_b, scaler_y_b, feat_cols_b, _ = prepare_building(
        building, electricity, weather_all, meter_dfs)
    X_aligned = align_to_schema(X_seq_b, feat_cols_b, schema)
    y_true    = scaler_y_b.inverse_transform(y_seq_b.reshape(-1, 1))

    n_full = len(X_aligned)
    if n_full > cap:
        # DIFFERENT seed_offset -> genuinely independent subsample
        cr  = np.random.default_rng(stable_seed(building) + seed_offset)
        idx = np.sort(cr.choice(n_full, cap, replace=False))
        X_aligned, y_true = X_aligned[idx], y_true[idx]

    fn  = _get_infer_fn(f'{model_key}_cap{cap}_{seed_offset}', model)
    rng = np.random.default_rng(stable_seed(building) + 7919 * seed_offset)
    deltas = permutation_importance(fn, X_aligned, y_true, scaler_y_b,
                                    len(schema), rng)
    del X_seq_b, y_seq_b, X_aligned, y_true; gc.collect()
    return collapse_vec(deltas)


def cap_validation(model, model_key, buildings, schema,
                   electricity, weather_all, meter_dfs, n_test=CAP_TEST_N):
    """A: same model, two independent 3000-draws  -> estimation noise
       B: same model, 3000 vs 8000               -> cap bias"""
    test_set = buildings[:n_test]
    rows = []
    for i, b in enumerate(test_set):
        try:
            v_a1 = _perm_with_cap(model_key, model, b, schema, CAP_SMALL, 0,
                                  electricity, weather_all, meter_dfs)
            v_a2 = _perm_with_cap(model_key, model, b, schema, CAP_SMALL, 1,
                                  electricity, weather_all, meter_dfs)
            v_big = _perm_with_cap(model_key, model, b, schema, CAP_BIG, 0,
                                   electricity, weather_all, meter_dfs)
            rows.append({
                'building': b,
                'rho_same_model_same_cap': spearmanr(v_a1, v_a2)[0],
                'rho_3000_vs_8000':        spearmanr(v_a1, v_big)[0],
                'top1_3000': COLLAPSED_LABELS[int(np.argmax(v_a1))],
                'top1_8000': COLLAPSED_LABELS[int(np.argmax(v_big))],
            })
        except Exception as e:
            rows.append({'building': b, 'error': f'{type(e).__name__}: {e}'})
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{n_test}")

    df = pd.DataFrame(rows)
    good = df[df.get('rho_same_model_same_cap').notna()] if 'rho_same_model_same_cap' in df else df

    print(f"\n{'='*64}\nCAP VALIDATION ({model_key}, n={len(good)})\n{'='*64}")
    a   = good['rho_same_model_same_cap']
    b8  = good['rho_3000_vs_8000']
    print(f"A. ESTIMATION-NOISE CEILING (same model, 2 independent 3000-draws)")
    print(f"     median rho = {a.median():.3f}  [{a.quantile(.25):.3f}, {a.quantile(.75):.3f}]")
    print(f"B. CAP SENSITIVITY (3000 vs 8000 sequences, same model)")
    print(f"     median rho = {b8.median():.3f}  [{b8.quantile(.25):.3f}, {b8.quantile(.75):.3f}]")
    print(f"     top-1 unchanged in {100*(good.top1_3000==good.top1_8000).mean():.0f}% of buildings")

    print(f"\nCOMPARE AGAINST within-LSTM = 0.775 (from Cell 4):")
    gap = a.median() - 0.775
    if gap >= 0.15:
        print(f"  Ceiling {a.median():.3f} >> 0.775. The {gap:.3f} gap is REAL")
        print("  model instability. Headline finding STANDS.")
    elif gap >= 0.05:
        print(f"  Ceiling {a.median():.3f} > 0.775 by {gap:.3f}. Some real")
        print("  instability, but estimation noise is a material component.")
        print("  Report BOTH numbers; quantify instability as the RESIDUAL.")
    else:
        print(f"  Ceiling {a.median():.3f} ~= 0.775. We have largely measured")
        print("  OUR OWN ESTIMATOR'S NOISE, not model instability.")
        print("  ** HEADLINE FINDING MUST BE REWRITTEN. ** Raise repeats/cap")
        print("  and re-measure before making any instability claim.")
    return df

cap_lstm = cap_validation(DRAW_MODELS['lstm_draw1'], 'lstm_draw1',
                          subsample, ANCHOR_FEATURE_SCHEMA,
                          electricity, weather_all, meter_dfs)
cap_lstm.to_csv(os.path.join(PHASE2_DIR,'results','analysis1d_cap_validation.csv'), index=False)
# Optionally repeat for the TFT — its within-arch rho (0.622) is LOWER,
# so it is the more likely to be estimation-noise-dominated:
cap_tft = cap_validation(DRAW_MODELS['tft_draw1'], 'tft_draw1',
                         subsample, ANCHOR_FEATURE_SCHEMA,
                         electricity, weather_all, meter_dfs)
cap_tft.to_csv(os.path.join(PHASE2_DIR,'results','analysis1d_cap_validation_tft.csv'), index=False)


# In[42]:


# =====================================================================
# ANALYSIS 1e — SHAP ACROSS WEIGHT DRAWS
# =====================================================================
# THE GAP THIS PLUGS:
#   Phase 1's headline SHAP result — Hour of Day is rank-1 for 100% of
#   1,294 buildings — was computed with ONE model. It has never been
#   checked across weight draws. Permutation attribution has now been
#   shown to be draw-unstable (within-LSTM rho 0.696 vs an estimation
#   ceiling of 0.971). Whether SHAP shares that instability is unknown.
#
# EITHER OUTCOME IS REPORTABLE:
#   - SHAP stays 100% HoD across all 3 draws  -> gradient-based
#     attribution is draw-INVARIANT where perturbation-based is not.
#     That makes the SHAP/permutation divergence a real METHODOLOGICAL
#     finding, not noise, and directly supports the §1.1 inversion.
#   - SHAP's hierarchy ALSO shifts -> a bigger finding than anything
#     else in the project: BOTH attribution families are draw-unstable,
#     and Phase 1's 100% figure is an artefact of a single draw.
#
# METHOD NOTE — matches Phase 1 exactly (decoded .tex, section 9):
#   background = X_tr[:SHAP_SAMPLES]              (anchor training set)
#   explainer  = shap.GradientExplainer(model, background)
#   shap_raw   = explainer.shap_values(X_aligned[:n_shap])
#   if 4-dim: squeeze(-1)   # the documented (200,48,15,1) quirk
#   mean_shap  = np.mean(np.abs(shap_arr), axis=(0,1))
# Then collapse sin/cos pairs before ranking, as everywhere else.
# ---------------------------------------------------------------------

SHAP_SAMPLES  = 200    # matches Phase 1
SHAP_TEST_N   = 25     # buildings; cheap, enough for a clear signal

def shap_vector(model, background, building, schema,
                electricity, weather_all, meter_dfs):
    """Collapsed mean-|SHAP| vector for one building, Phase 1 convention."""
    X_seq_b, y_seq_b, scaler_y_b, feat_cols_b, _ = prepare_building(
        building, electricity, weather_all, meter_dfs)
    X_aligned = align_to_schema(X_seq_b, feat_cols_b, schema)

    explainer = shap.GradientExplainer(model, background)
    n_shap    = min(SHAP_SAMPLES, len(X_aligned))
    shap_raw  = explainer.shap_values(X_aligned[:n_shap])

    shap_arr = np.array(shap_raw)
    if shap_arr.ndim == 4:
        shap_arr = shap_arr.squeeze(-1)      # (200,48,15,1) -> (200,48,15)
    mean_shap = np.mean(np.abs(shap_arr), axis=(0, 1))   # -> (15,)

    del X_seq_b, y_seq_b, X_aligned; gc.collect()
    return collapse_vec(mean_shap)


def shap_across_draws(draw_models, arch, buildings, schema,
                      electricity, weather_all, meter_dfs,
                      n_test=SHAP_TEST_N):
    """SHAP for draws 1/2/3 of one architecture, same buildings."""
    keys = [f'{arch}_draw{d}' for d in (1, 2, 3)]
    background = X_train[:SHAP_SAMPLES]      # anchor training set, Phase 1 convention

    rows = []
    for i, b in enumerate(buildings[:n_test]):
        row = {'building': b}
        try:
            vecs = {k: shap_vector(draw_models[k], background, b, schema,
                                   electricity, weather_all, meter_dfs)
                    for k in keys}
            for k in keys:
                row[f'top1_{k}'] = COLLAPSED_LABELS[int(np.argmax(vecs[k]))]
            # pairwise agreement across draws
            pairs = [(keys[0],keys[1]), (keys[0],keys[2]), (keys[1],keys[2])]
            rhos = [spearmanr(vecs[a], vecs[b_])[0] for a, b_ in pairs]
            row['mean_rho']   = float(np.nanmean(rhos))
            row['top1_all_same'] = len({row[f'top1_{k}'] for k in keys}) == 1
            row['status'] = 'ok'
        except Exception as e:
            row['status'] = f'{type(e).__name__}: {e}'
        rows.append(row)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{n_test}")

    df = pd.DataFrame(rows)
    ok = df[df.status == 'ok']

    print(f"\n{'='*64}\nSHAP ACROSS DRAWS ({arch}, n={len(ok)})\n{'='*64}")
    print(f"Per-building mean rho across the 3 draws:")
    print(f"  median {ok['mean_rho'].median():.3f}  "
          f"[{ok['mean_rho'].quantile(.25):.3f}, {ok['mean_rho'].quantile(.75):.3f}]")
    print(f"Top-1 identical across all 3 draws: "
          f"{100*ok['top1_all_same'].mean():.0f}% of buildings")

    print("\nTop-1 feature distribution per draw:")
    for k in keys:
        vc = ok[f'top1_{k}'].value_counts()
        print(f"  {k}: " + ", ".join(f"{f} {100*c/len(ok):.0f}%" for f, c in vc.items()))

    hod_rates = [100*(ok[f'top1_{k}'] == 'Hour of Day').mean() for k in keys]
    print(f"\nHour-of-Day rank-1 rate by draw: "
          + ", ".join(f"{r:.0f}%" for r in hod_rates))

    print("\nCOMPARE against permutation on the same architecture:")
    print("  permutation within-LSTM rho = 0.696 (per-building, Cell 5)")
    print("  permutation estimation ceiling = 0.971 (Cell 1d)")
    m = ok['mean_rho'].median()
    if m >= 0.95 and min(hod_rates) >= 99:
        print(f"\n  => SHAP is DRAW-INVARIANT (rho {m:.3f}, HoD 100% in every draw)")
        print("     while permutation is not. The SHAP/permutation divergence")
        print("     is a genuine METHODOLOGICAL property, not noise. This")
        print("     directly supports the §1.1 inversion.")
    elif m >= 0.95:
        print(f"\n  => SHAP rankings are stable overall (rho {m:.3f}) but the")
        print("     HoD rate is NOT 100% in every draw. Phase 1's 100% figure")
        print("     is draw-specific — report the per-draw rates.")
    else:
        print(f"\n  => SHAP IS ALSO DRAW-UNSTABLE (rho {m:.3f}).")
        print("     ** This is the bigger finding. ** Both attribution")
        print("     families are draw-sensitive, and Phase 1's 100% HoD")
        print("     result is an artefact of a single weight draw.")
        print("     NOTE: SHAP has its own estimation noise (sampled")
        print("     GradientExplainer). Before claiming instability, run the")
        print("     SAME ceiling test on SHAP: two SHAP runs on ONE frozen")
        print("     model. Do not repeat the mistake of interpreting a")
        print("     difference without calibrating the instrument.")
    return df

shap_lstm = shap_across_draws(DRAW_MODELS, 'lstm', subsample,
                              ANCHOR_FEATURE_SCHEMA,
                              electricity, weather_all, meter_dfs)
shap_lstm.to_csv(os.path.join(PHASE2_DIR,'results','analysis1e_shap_draws_lstm.csv'), index=False)

# TFT NOTE: GradientExplainer on the TFT may fail or need the
# attribution_model variant (custom layers, multi-output). If it errors,
# that is itself worth recording — "SHAP could not be applied to the TFT
# under matched settings" is a legitimate methodological limitation, and
# is likely WHY Phase 1 used permutation for the cross-architecture work.


# In[43]:


# =====================================================================
# ANALYSIS 1f — SHAP SATURATION CHECK
# =====================================================================
# THE CHALLENGE THIS PRE-EMPTS:
#   SHAP is draw-invariant (rho 0.984, HoD rank-1 in 100% of buildings
#   across 3 draws). But is that ROBUSTNESS or SATURATION? If Hour of
#   Day dominates the gradient signal so overwhelmingly that nothing
#   else could ever rank first, SHAP is "stable" the way a broken
#   thermometer is stable.
#
# FOUR DIAGNOSTICS, cheap, on data already computed:
#   1. DOMINANCE RATIO  = SHAP(rank1) / SHAP(rank2), per building.
#      Huge ratio (>20x) => saturation is a live concern.
#      Modest (2-5x)     => genuinely resolving a hierarchy.
#   2. CONCENTRATION     = share of total |SHAP| held by rank-1.
#      ~1.0 => one feature absorbs everything.
#   3. SUB-HIERARCHY STABILITY = rho across draws computed with
#      Hour of Day REMOVED. This is the decisive one: if the remaining
#      11 features still agree across draws, SHAP is resolving real
#      structure below the dominant feature, and the stability is NOT
#      an artefact of one feature swamping the ranking.
#   4. Same diagnostics for PERMUTATION, as the contrast. If
#      permutation's dominance ratio is much lower, that explains WHY
#      it is draw-unstable: near-ties flip easily.
#
# INTERPRETATION: diagnostic 3 is what settles it. 1 and 2 describe the
# shape of the distribution; 3 tests whether stability survives removing
# the dominant term.
# ---------------------------------------------------------------------

def saturation_check(draw_models, arch, buildings, schema,
                     electricity, weather_all, meter_dfs, n_test=25):
    keys = [f'{arch}_draw{d}' for d in (1, 2, 3)]
    background = X_train[:SHAP_SAMPLES]
    hod_i = COLLAPSED_LABELS.index('Hour of Day')

    rows = []
    for i, b in enumerate(buildings[:n_test]):
        try:
            sv = {k: shap_vector(draw_models[k], background, b, schema,
                                 electricity, weather_all, meter_dfs)
                  for k in keys}
            v = sv[keys[0]]                      # draw 1 for the shape stats
            order = np.argsort(v)[::-1]
            r1, r2 = v[order[0]], v[order[1]]

            # sub-hierarchy: drop Hour of Day, re-correlate across draws
            mask = np.ones(len(COLLAPSED_LABELS), dtype=bool); mask[hod_i] = False
            sub_rhos = [spearmanr(sv[a][mask], sv[b_][mask])[0]
                        for a, b_ in [(keys[0],keys[1]),(keys[0],keys[2]),(keys[1],keys[2])]]

            rows.append({
                'building': b,
                'rank1': COLLAPSED_LABELS[order[0]],
                'rank2': COLLAPSED_LABELS[order[1]],
                'dominance_ratio': r1 / r2 if r2 > 0 else np.inf,
                'rank1_share':     r1 / v.sum() if v.sum() > 0 else np.nan,
                'sub_rho':         float(np.nanmean(sub_rhos)),
                'status': 'ok',
            })
        except Exception as e:
            rows.append({'building': b, 'status': f'{type(e).__name__}: {e}'})
        if (i+1) % 5 == 0: print(f"  {i+1}/{n_test}")
        gc.collect()

    df = pd.DataFrame(rows); ok = df[df.status=='ok']
    dr, sh, sr = ok['dominance_ratio'], ok['rank1_share'], ok['sub_rho']

    print(f"\n{'='*64}\nSHAP SATURATION CHECK ({arch}, n={len(ok)})\n{'='*64}")
    print(f"1. DOMINANCE RATIO  rank1/rank2")
    print(f"     median {dr.median():.2f}x   [{dr.quantile(.25):.2f}, {dr.quantile(.75):.2f}]")
    print(f"2. CONCENTRATION    rank1 share of total |SHAP|")
    print(f"     median {sh.median():.3f}   [{sh.quantile(.25):.3f}, {sh.quantile(.75):.3f}]")
    print(f"3. SUB-HIERARCHY    across-draw rho with Hour of Day REMOVED")
    print(f"     median {sr.median():.3f}   [{sr.quantile(.25):.3f}, {sr.quantile(.75):.3f}]")
    print(f"\n   rank-2 feature distribution: "
          + ", ".join(f"{f} {100*c/len(ok):.0f}%" for f,c in ok['rank2'].value_counts().items()))

    print("\nVERDICT:")
    if sr.median() >= 0.80:
        print(f"  Sub-hierarchy rho = {sr.median():.3f} WITHOUT Hour of Day.")
        print("  SHAP resolves stable structure BELOW the dominant feature.")
        print("  => NOT saturation. The stability is real. Report diagnostic 3")
        print("     as the pre-emptive answer to the saturation challenge.")
    elif sr.median() >= 0.50:
        print(f"  Sub-hierarchy rho = {sr.median():.3f}. Partial: the top rank")
        print("  is stable but the lower ranks are less so. State this honestly —")
        print("  SHAP's stability is strongest at rank-1 and weakens below it.")
    else:
        print(f"  Sub-hierarchy rho = {sr.median():.3f}. Once Hour of Day is")
        print("  removed, SHAP's remaining ranking is NOT stable across draws.")
        print("  ** The rho=0.984 headline IS largely saturation. ** SHAP's")
        print("  apparent robustness reflects one feature dominating the")
        print("  ranking, not genuine resolution of a hierarchy. This must be")
        print("  reported — it materially weakens the SHAP-is-robust claim.")

    if dr.median() > 20:
        print(f"\n  NOTE: dominance ratio {dr.median():.1f}x is extreme. Even with a")
        print("  stable sub-hierarchy, state that rank-1 is effectively")
        print("  uncontested and top-1 agreement is therefore a weak test.")
    return df

sat = saturation_check(DRAW_MODELS, 'lstm', subsample, ANCHOR_FEATURE_SCHEMA,
                       electricity, weather_all, meter_dfs)
sat.to_csv(os.path.join(PHASE2_DIR,'results','analysis1f_saturation.csv'), index=False)

# ── OPTIONAL CONTRAST: same shape stats for PERMUTATION ──────────────
# If permutation's dominance ratio is much LOWER than SHAP's, that is
# the mechanistic explanation for its draw-instability: near-ties at the
# top flip under re-initialisation. Uses var_df, already computed —
# no new model calls needed.
def permutation_dominance(var_df, schema, key='lstm_draw1'):
    ok = var_df[var_df.status=='ok']
    cols = [f'perm_{key}_{f}' for f in schema]
    out = []
    for _, r in ok.iterrows():
        v = collapse_vec(r[cols].to_numpy(dtype=float))
        o = np.argsort(v)[::-1]
        if v[o[1]] > 0:
            out.append({'dominance_ratio': v[o[0]]/v[o[1]],
                        'rank1_share': v[o[0]]/v[v>0].sum() if (v>0).any() else np.nan})
    d = pd.DataFrame(out)
    print(f"\nPERMUTATION dominance ({key}, n={len(d)}):")
    print(f"  median ratio {d['dominance_ratio'].median():.2f}x  "
          f"share {d['rank1_share'].median():.3f}")
    print("  Compare against SHAP above. A much lower ratio here is the")
    print("  MECHANISM for permutation's instability: the top ranks are")
    print("  close together, so small weight changes reorder them.")
    return d

perm_dom = permutation_dominance(var_df, ANCHOR_FEATURE_SCHEMA)


# ## Integrated Gradients — zero-baseline configuration (superseded)
# 
# Retained as a record of the initial IG configuration (zero baseline, 50 integration steps). This produced a SHAP–IG cross-method agreement of rho = 0.16. The completeness-axiom error at 50 steps was 0.096. Superseded by the canonical configuration in the next cell (Analysis 3a: training-mean baseline, 200 steps), which reduces the completeness error to 0.019 and raises cross-method agreement to rho = 0.54. The comparison between the two baselines is itself a reported result on the baseline-sensitivity of gradient attribution.

# In[63]:


# =====================================================================
# ANALYSIS 3 — INTEGRATED GRADIENTS: method-family test
# =====================================================================
# THE QUESTION: is the SHAP-vs-permutation split a GRADIENT-vs-
# PERTURBATION distinction, or something specific to SHAP?
#
#   IG stable (~0.98, like SHAP)  -> the split is METHOD FAMILY.
#     Two independent gradient methods agree; the perturbation method
#     does not. Mechanistic claim: gradient methods read the learned
#     input-output sensitivity (constrained by the data); perturbation
#     measures downstream error dependence (a property of the specific
#     function that draw converged to).
#   IG unstable (~0.70, like perm) -> NOT a family split. SHAP's
#     stability becomes the anomaly and needs its own explanation.
#
# Flagged as future work in the interim report — delivering it as a
# method-FAMILY test rather than "one more attribution method" is the
# stronger contribution.
#
# ORDER OF OPERATIONS (learned the hard way with permutation):
#   1. CEILING FIRST. Two IG runs on ONE frozen model. An attribution
#      number is meaningless without the instrument's precision.
#   2. Only then interpret across-draw agreement.
#
# METHOD: Integrated Gradients (Sundararajan et al. 2017).
#   IG_i = (x_i - x'_i) * mean_k[ dF/dx_i at x' + (k/m)(x - x') ]
#   Baseline x' = ZEROS in scaled space. NOTE: after MinMaxScaler,
#   zero is the training-partition MINIMUM, not the mean. This is the
#   conventional choice and matches "absence of signal" in a [0,1]
#   scaled space, but it IS a choice — state it in the Methodology.
#   Aggregation matches the SHAP convention exactly:
#   mean(|attribution|) over (samples, timesteps) -> per-feature vector,
#   then collapse sin/cos pairs. Same aggregation = fair comparison.
# ---------------------------------------------------------------------

IG_STEPS   = 50     # Riemann steps along the path. 50 is standard.
IG_SAMPLES = 200    # matches SHAP_SAMPLES for a like-for-like comparison
IG_BATCH   = 25     # samples per gradient batch; keeps GPU memory bounded


def integrated_gradients(model, X, baseline=None,
                         steps=IG_STEPS, batch=IG_BATCH):
    """Per-feature mean(|IG|) over (samples, timesteps). Returns (n_features,).

    Memory-bounded: processes `batch` samples at a time, and within each
    sample-batch loops the interpolation steps rather than materialising
    (batch * steps) sequences at once. Learned from the permutation OOMs
    — the TFT's VSN expands intermediates per feature.
    """
    X = np.asarray(X, dtype=np.float32)
    if baseline is None:
        baseline = np.zeros_like(X[:1]).astype(np.float32)   # zero baseline
    total = np.zeros(X.shape[1:], dtype=np.float64)   # (window, n_features)
    n = 0

    for s in range(0, len(X), batch):
        xb = X[s:s+batch]
        bl = np.repeat(baseline, len(xb), axis=0)
        diff = xb - bl
        acc = np.zeros_like(xb, dtype=np.float64)

        for k in range(1, steps + 1):
            alpha = k / steps
            pt = tf.convert_to_tensor(bl + alpha * diff)
            with tf.GradientTape() as tape:
                tape.watch(pt)
                out = model(pt, training=False)
            g = tape.gradient(out, pt)
            acc += g.numpy()

        # average gradient along path, times (x - baseline)
        ig = (acc / steps) * diff
        total += np.abs(ig).sum(axis=0)
        n += len(xb)
        del acc, ig; gc.collect()

    per_feature = (total / n).mean(axis=0)        # mean over timesteps
    return per_feature


def ig_vector(model, building, schema, electricity, weather_all, meter_dfs,
              subsample_seed=0, n_samples=IG_SAMPLES):
    """Collapsed IG vector for one building."""
    X_seq_b, y_seq_b, scaler_y_b, feat_cols_b, _ = prepare_building(
        building, electricity, weather_all, meter_dfs)
    X_aligned = align_to_schema(X_seq_b, feat_cols_b, schema)

    # independent subsample -> lets us measure the estimation ceiling
    if len(X_aligned) > n_samples:
        rng = np.random.default_rng(stable_seed(building) + subsample_seed)
        idx = np.sort(rng.choice(len(X_aligned), n_samples, replace=False))
        X_aligned = X_aligned[idx]

    raw = integrated_gradients(model, X_aligned)
    del X_seq_b, y_seq_b, X_aligned; gc.collect()
    return collapse_vec(raw)


# =====================================================================
# STEP 1 — CEILING. Two IG runs, ONE frozen model, independent subsamples
# =====================================================================
def ig_ceiling(model, model_key, buildings, schema,
               electricity, weather_all, meter_dfs, n_test=25):
    rows = []
    for i, b in enumerate(buildings[:n_test]):
        try:
            v1 = ig_vector(model, b, schema, electricity, weather_all,
                           meter_dfs, subsample_seed=0)
            v2 = ig_vector(model, b, schema, electricity, weather_all,
                           meter_dfs, subsample_seed=1)
            rows.append({'building': b,
                         'rho': spearmanr(v1, v2)[0],
                         'top1_a': COLLAPSED_LABELS[int(np.argmax(v1))],
                         'top1_b': COLLAPSED_LABELS[int(np.argmax(v2))],
                         'status': 'ok'})
        except Exception as e:
            rows.append({'building': b, 'status': f'{type(e).__name__}: {e}'})
        if (i+1) % 5 == 0: print(f"  {i+1}/{n_test}")

    df = pd.DataFrame(rows); ok = df[df.status=='ok']
    print(f"\n{'='*64}\nIG ESTIMATION CEILING ({model_key}, n={len(ok)})\n{'='*64}")
    print(f"  median rho = {ok['rho'].median():.3f}  "
          f"[{ok['rho'].quantile(.25):.3f}, {ok['rho'].quantile(.75):.3f}]")
    print(f"  top-1 identical: {100*(ok.top1_a==ok.top1_b).mean():.0f}%")
    print("\n  Reference ceilings: permutation LSTM 0.971, TFT 0.986")
    if ok['rho'].median() < 0.90:
        print("  !! IG's own estimate is noisy. Raise IG_SAMPLES or IG_STEPS")
        print("     BEFORE interpreting any across-draw number.")
    return df


# =====================================================================
# STEP 2 — ACROSS DRAWS (only meaningful once the ceiling is known)
# =====================================================================
def ig_across_draws(draw_models, arch, buildings, schema,
                    electricity, weather_all, meter_dfs, n_test=25):
    keys = [f'{arch}_draw{d}' for d in (1, 2, 3)]
    hod_i = COLLAPSED_LABELS.index('Hour of Day')
    rows = []
    for i, b in enumerate(buildings[:n_test]):
        row = {'building': b}
        try:
            v = {k: ig_vector(draw_models[k], b, schema,
                              electricity, weather_all, meter_dfs)
                 for k in keys}
            pairs = [(keys[0],keys[1]),(keys[0],keys[2]),(keys[1],keys[2])]
            row['mean_rho'] = float(np.nanmean(
                [spearmanr(v[a], v[c])[0] for a, c in pairs]))
            # sub-hierarchy with Hour of Day removed — same saturation
            # guard we applied to SHAP
            m = np.ones(len(COLLAPSED_LABELS), bool); m[hod_i] = False
            row['sub_rho'] = float(np.nanmean(
                [spearmanr(v[a][m], v[c][m])[0] for a, c in pairs]))
            for k in keys:
                row[f'top1_{k}'] = COLLAPSED_LABELS[int(np.argmax(v[k]))]
            vv = v[keys[0]]; o = np.argsort(vv)[::-1]
            row['dominance_ratio'] = vv[o[0]]/vv[o[1]] if vv[o[1]] > 0 else np.inf
            row['rank1_share']     = vv[o[0]]/vv.sum() if vv.sum() > 0 else np.nan
            row['top1_all_same']   = len({row[f'top1_{k}'] for k in keys}) == 1
            row['status'] = 'ok'
        except Exception as e:
            row['status'] = f'{type(e).__name__}: {e}'
        rows.append(row)
        if (i+1) % 5 == 0: print(f"  {i+1}/{n_test}")

    df = pd.DataFrame(rows); ok = df[df.status=='ok']
    print(f"\n{'='*64}\nIG ACROSS DRAWS ({arch}, n={len(ok)})\n{'='*64}")
    print(f"  across-draw rho     : {ok['mean_rho'].median():.3f}  "
          f"[{ok['mean_rho'].quantile(.25):.3f}, {ok['mean_rho'].quantile(.75):.3f}]")
    print(f"  sub-hierarchy rho   : {ok['sub_rho'].median():.3f}   (Hour of Day removed)")
    print(f"  top-1 same all draws: {100*ok['top1_all_same'].mean():.0f}%")
    print(f"  dominance ratio     : {ok['dominance_ratio'].median():.2f}x")
    print(f"  rank-1 share        : {ok['rank1_share'].median():.3f}")
    for k in keys:
        vc = ok[f'top1_{k}'].value_counts()
        print(f"  {k}: " + ", ".join(f"{f} {100*c/len(ok):.0f}%" for f,c in vc.items()))

    print("\n  COMPARISON TABLE (LSTM, across draws):")
    print("    method        rho     ceiling   dom.ratio  rank1 share")
    print(f"    SHAP          0.984   ~1.0      1.35x      0.262")
    print(f"    permutation   0.696   0.971     2.52x      0.607")
    print(f"    IG            {ok['mean_rho'].median():.3f}   [see step 1] "
          f"{ok['dominance_ratio'].median():.2f}x      {ok['rank1_share'].median():.3f}")

    m = ok['mean_rho'].median()
    print("\nVERDICT:")
    if m >= 0.95:
        print("  IG is draw-STABLE, like SHAP. => The split is GRADIENT vs")
        print("  PERTURBATION, not SHAP-specific. Two independent gradient")
        print("  methods agree across initialisations; the perturbation")
        print("  method does not. This is the method-FAMILY claim.")
    elif m <= 0.80:
        print("  IG is draw-UNSTABLE, like permutation. => NOT a family split.")
        print("  SHAP's stability is the anomaly and needs its own explanation.")
        print("  Check whether SHAP's GradientExplainer background (fixed")
        print("  anchor X_train[:200]) is imposing the stability.")
    else:
        print(f"  IG is INTERMEDIATE (rho {m:.3f}). Report the ordering")
        print("  SHAP > IG > permutation and resist a clean binary story.")
    return df


# ── RUN ORDER — ceiling first, always ────────────────────────────────
ig_cal = ig_ceiling(DRAW_MODELS['lstm_draw1'], 'lstm_draw1', subsample,
                    ANCHOR_FEATURE_SCHEMA, electricity, weather_all, meter_dfs)
ig_cal.to_csv(os.path.join(PHASE2_DIR,'results','analysis3_ig_ceiling_ZEROBASELINE.csv'), index=False)

ig_draws = ig_across_draws(DRAW_MODELS, 'lstm', subsample,
                           ANCHOR_FEATURE_SCHEMA, electricity, weather_all, meter_dfs)
ig_draws.to_csv(os.path.join(PHASE2_DIR,'results','analysis3_ig_draws_lstm_ZEROBASELINE.csv'), index=False)


# In[45]:


# =====================================================================
# ANALYSIS 4 — DOES GRADIENT ATTRIBUTION STRATIFY BY BUILDING TYPE?
# =====================================================================
# THE OPEN QUESTION AFTER ANALYSES 1 AND 3:
#   Gradient methods (SHAP, IG) are draw-STABLE (instability ~0.02-0.03)
#   but rank-1 is Hour of Day in 100% of buildings — zero stratification
#   at the top. Perturbation STRATIFIES by type but is draw-UNSTABLE
#   (0.275). So no method currently supports a type-stratification claim.
#
#   BUT: rank-1 is not the whole ranking. SHAP's dominance ratio is only
#   1.35x and rank-1 holds just 26% of total attribution — the hierarchy
#   is DISTRIBUTED. If ranks 2/3/4 vary systematically by building type,
#   we have stratification FROM A STABLE METHOD, which rescues the claim.
#   If they do not, the tension in §000 is confirmed at full population
#   and becomes the reported finding.
#
# WHY THIS BEATS THE PERMUTATION FULL SWEEP:
#   Sweeping permutation at n=1,276 gives precise estimates of an
#   IRREPRODUCIBLE quantity. Sweeping gradient methods gives estimates
#   of a REPRODUCIBLE one. Same compute, one of them is worth having.
#
# DESIGN:
#   - ONE model per architecture (draw 1). Justified: gradient
#     attribution is draw-invariant (rho 0.984 / 0.967), so extra draws
#     add ~nothing. This is what makes full population affordable.
#   - Both SHAP and IG, so any stratification is confirmed by two
#     independent gradient methods, not one.
#   - Full ranking recorded per building, not just top-1.
# ---------------------------------------------------------------------

STRAT_CSV = Path(PHASE2_DIR) / 'results' / 'analysis4_gradient_stratification.csv'


def gradient_vectors_one_building(model, background, building, schema,
                                  electricity, weather_all, meter_dfs):
    """Both gradient methods on one building. Returns (shap_vec, ig_vec),
    each collapsed to the 12-feature space."""
    v_shap = shap_vector(model, background, building, schema,
                         electricity, weather_all, meter_dfs)
    v_ig   = ig_vector(model, building, schema,
                       electricity, weather_all, meter_dfs, subsample_seed=0)
    return v_shap, v_ig


def stratification_sweep(model, model_key, buildings, schema,
                         electricity, weather_all, meter_dfs,
                         out_path=STRAT_CSV, resume=True):
    """Full-population SHAP + IG. Records the COMPLETE collapsed vector
    per building per method, so any downstream question (rank-2, rank-3,
    share-of-total, type contrasts) can be answered without re-running."""
    background = X_train[:SHAP_SAMPLES]

    done = set()
    if resume and out_path.exists():
        done = set(pd.read_csv(out_path)['building'])
        print(f"Resuming: {len(done)} already done.")
    todo  = [b for b in buildings if b not in done]
    first = not out_path.exists()
    print(f"To do: {len(todo)}")

    for i, b in enumerate(todo):
        row = {'building': b, 'model': model_key}
        try:
            v_shap, v_ig = gradient_vectors_one_building(
                model, background, b, schema,
                electricity, weather_all, meter_dfs)
            for j, lab in enumerate(COLLAPSED_LABELS):
                row[f'shap_{lab}'] = v_shap[j]
                row[f'ig_{lab}']   = v_ig[j]
            # convenience columns
            for tag, v in (('shap', v_shap), ('ig', v_ig)):
                o = np.argsort(v)[::-1]
                row[f'{tag}_rank1'] = COLLAPSED_LABELS[o[0]]
                row[f'{tag}_rank2'] = COLLAPSED_LABELS[o[1]]
                row[f'{tag}_rank3'] = COLLAPSED_LABELS[o[2]]
                row[f'{tag}_dom_ratio'] = v[o[0]]/v[o[1]] if v[o[1]] > 0 else np.inf
            row['shap_ig_rho'] = spearmanr(v_shap, v_ig)[0]
            row['status'] = 'ok'
        except Exception as e:
            row['status'] = f'{type(e).__name__}: {e}'
        pd.DataFrame([row]).to_csv(out_path, mode='a', header=first, index=False)
        first = False
        if (i+1) % 25 == 0:
            print(f"  {i+1}/{len(todo)}  ({b})")
        gc.collect()

    print(f"Done -> {out_path}")
    return pd.read_csv(out_path)


# =====================================================================
# ANALYSIS — does the gradient hierarchy vary by type?
# =====================================================================
def gradient_stratification_analysis(df, building_summary, min_n=30):
    ok = df[df.status == 'ok'].merge(
        building_summary[['building','type']], on='building', how='left').dropna(subset=['type'])
    big = ok[ok.groupby('type')['type'].transform('size') >= min_n]

    print(f"{'='*70}\nGRADIENT STRATIFICATION (n={len(ok)}, types with n>={min_n}: "
          f"{big['type'].nunique()})\n{'='*70}")

    for tag, name in (('shap','SHAP'), ('ig','IG')):
        print(f"\n── {name} ──")
        print(f"rank-1 distribution overall: "
              + ", ".join(f"{f} {100*c/len(ok):.1f}%"
                          for f,c in ok[f'{tag}_rank1'].value_counts().head(3).items()))
        print(f"rank-2 distribution overall: "
              + ", ".join(f"{f} {100*c/len(ok):.1f}%"
                          for f,c in ok[f'{tag}_rank2'].value_counts().head(3).items()))
        print(f"rank-3 distribution overall: "
              + ", ".join(f"{f} {100*c/len(ok):.1f}%"
                          for f,c in ok[f'{tag}_rank3'].value_counts().head(3).items()))

        # Per-type rank-2 and rank-3 composition — where stratification
        # would show up if the top slot is saturated.
        for rk in ('rank2','rank3'):
            ct = pd.crosstab(big['type'], big[f'{tag}_{rk}'], normalize='index')*100
            print(f"\n  per-type {rk} composition (%):")
            print(ct.round(1).to_string())

        # KRUSKAL-WALLIS on the CONTINUOUS attribution share, per feature.
        # Non-parametric, no distributional assumption. Tests whether the
        # attribution given to feature f differs across building types.
        # This is the RIGHT test: it uses the full continuous signal
        # rather than throwing it away by taking an argmax.
        print(f"\n  Kruskal-Wallis: does {name} attribution differ by type?")
        print(f"  {'feature':22s} {'H':>10s} {'p':>12s}")
        results = []
        for lab in COLLAPSED_LABELS:
            groups = [g[f'{tag}_{lab}'].to_numpy(dtype=float)
                      for _, g in big.groupby('type') if len(g) >= min_n]
            if len(groups) < 2: continue
            try:
                H, p = kruskal(*groups)
                results.append((lab, H, p))
            except Exception:
                continue
        # Holm-Bonferroni: we are testing 12 features simultaneously.
        # Without correction, ~1 in 20 will look significant by chance.
        results.sort(key=lambda r: r[2])
        m = len(results)
        for i, (lab, H, p) in enumerate(results):
            adj = min(1.0, p * (m - i))
            flag = '***' if adj < 0.001 else '**' if adj < 0.01 else '*' if adj < 0.05 else ''
            print(f"  {lab:22s} {H:10.2f} {p:12.3g}  Holm-adj {adj:8.3g} {flag}")

    print(f"\n── Cross-method agreement (SHAP vs IG, same model) ──")
    print(f"  median rho = {ok['shap_ig_rho'].median():.3f}  "
          f"[{ok['shap_ig_rho'].quantile(.25):.3f}, {ok['shap_ig_rho'].quantile(.75):.3f}]")
    print("  (If this is HIGH, the two gradient methods agree with each other")
    print("   as well as across draws — the family claim is doubly supported.)")

    print("\nHOW TO READ THIS:")
    print("  Significant Kruskal-Wallis on several features => gradient")
    print("  attribution DOES vary by building type in MAGNITUDE, even")
    print("  though rank-1 does not vary. That rescues the stratification")
    print("  claim with a STABLE method — a much stronger result than the")
    print("  permutation stratification, which is not reproducible.")
    print("  NOT significant => the tension in §000 is confirmed at full")
    print("  population: the stable method carries no type signal. Report")
    print("  the tension as the finding; do not hunt for a method that")
    print("  gives the desired answer.")
    return ok




# In[47]:


# IG completeness axiom: sum(IG) ≈ F(x) - F(baseline)
b = retained[0]
X_seq_b, y_seq_b, sc, fc, _ = prepare_building(b, electricity, weather_all, meter_dfs)
Xa = align_to_schema(X_seq_b, fc, ANCHOR_FEATURE_SCHEMA)[:50]
m = DRAW_MODELS['lstm_draw1']
baseline = np.zeros_like(Xa[:1])

# per-sample IG WITHOUT the abs() and without averaging
diff = Xa - np.repeat(baseline, len(Xa), axis=0)
acc = np.zeros_like(Xa, dtype=np.float64)
for k in range(1, 201):
    pt = tf.convert_to_tensor((np.repeat(baseline, len(Xa), axis=0) + (k/200)*diff).astype(np.float32))
    with tf.GradientTape() as tape:
        tape.watch(pt); out = m(pt, training=False)
    acc += tape.gradient(out, pt).numpy()
ig_signed = (acc/200) * diff

lhs = ig_signed.sum(axis=(1,2))                    # summed attribution per sample
rhs = (m(Xa, training=False).numpy().ravel()
       - m(np.repeat(baseline, len(Xa), axis=0).astype(np.float32), training=False).numpy().ravel())
print("IG completeness — should be ~equal:")
print("  mean |lhs-rhs| :", np.abs(lhs-rhs).mean())
print("  mean |rhs|     :", np.abs(rhs).mean())
print("  relative error :", np.abs(lhs-rhs).mean()/np.abs(rhs).mean())


# ## Integrated Gradients — canonical configuration (Analysis 3a)
# 
# The reported IG results use this configuration: training-mean baseline, 200 integration steps. The mean baseline is preferred over the zero/all-minimum baseline because zero in scaled space is a physically implausible all-minimum state, and 200 steps reduces the completeness-axiom error to 0.019 (versus 0.096 at 50 steps). This configuration gives an across-draw stability of rho = 0.94 (estimation ceiling 0.99) and a SHAP-IG cross-method agreement of rho = 0.54. All Analysis 3/4 figures in the report derive from this cell, not the superseded zero-baseline cell above.

# In[50]:


# =====================================================================
# ANALYSIS 3a — INTEGRATED GRADIENTS: method-family test
# =====================================================================
# THE QUESTION: is the SHAP-vs-permutation split a GRADIENT-vs-
# PERTURBATION distinction, or something specific to SHAP?
#
#   IG stable (~0.98, like SHAP)  -> the split is METHOD FAMILY.
#     Two independent gradient methods agree; the perturbation method
#     does not. Mechanistic claim: gradient methods read the learned
#     input-output sensitivity (constrained by the data); perturbation
#     measures downstream error dependence (a property of the specific
#     function that draw converged to).
#   IG unstable (~0.70, like perm) -> NOT a family split. SHAP's
#     stability becomes the anomaly and needs its own explanation.
#
# Flagged as future work in the interim report — delivering it as a
# method-FAMILY test rather than "one more attribution method" is the
# stronger contribution.
#
# ORDER OF OPERATIONS (learned the hard way with permutation):
#   1. CEILING FIRST. Two IG runs on ONE frozen model. An attribution
#      number is meaningless without the instrument's precision.
#   2. Only then interpret across-draw agreement.
#
# METHOD: Integrated Gradients (Sundararajan et al. 2017).
#   IG_i = (x_i - x'_i) * mean_k[ dF/dx_i at x' + (k/m)(x - x') ]
#   Baseline x' = TRAINING MEAN (X_train.mean over samples, per timestep
#   and feature). Chosen over the zero/all-minimum baseline because in
#   MinMaxScaler space zero is the training-partition MINIMUM — a
#   physically implausible all-minimum state — whereas the training mean
#   is a real, interpretable reference. The baseline is a reported choice:
#   it shifts attribution and roughly triples SHAP-IG agreement relative
#   to the zero baseline (rho 0.16 -> 0.54). State this in the Methodology.
#   Aggregation matches the SHAP convention exactly:
#   mean(|attribution|) over (samples, timesteps) -> per-feature vector,
#   then collapse sin/cos pairs. Same aggregation = fair comparison.
# ---------------------------------------------------------------------

IG_STEPS   = 200     # Riemann steps along the path. 50 is standard.
IG_SAMPLES = 200    # matches SHAP_SAMPLES for a like-for-like comparison
IG_BATCH   = 25     # samples per gradient batch; keeps GPU memory bounded


def integrated_gradients(model, X, baseline=None,
                         steps=IG_STEPS, batch=IG_BATCH):
    """Per-feature mean(|IG|) over (samples, timesteps). Returns (n_features,).

    Memory-bounded: processes `batch` samples at a time, and within each
    sample-batch loops the interpolation steps rather than materialising
    (batch * steps) sequences at once. Learned from the permutation OOMs
    — the TFT's VSN expands intermediates per feature.
    """
    X = np.asarray(X, dtype=np.float32)
    if baseline is None:
        baseline = X_train.mean(axis=0, keepdims=True).astype(np.float32)
    total = np.zeros(X.shape[1:], dtype=np.float64)   # (window, n_features)
    n = 0

    for s in range(0, len(X), batch):
        xb = X[s:s+batch]
        bl = np.repeat(baseline, len(xb), axis=0)
        diff = xb - bl
        acc = np.zeros_like(xb, dtype=np.float64)

        for k in range(1, steps + 1):
            alpha = k / steps
            pt = tf.convert_to_tensor(bl + alpha * diff)
            with tf.GradientTape() as tape:
                tape.watch(pt)
                out = model(pt, training=False)
            g = tape.gradient(out, pt)
            acc += g.numpy()

        # average gradient along path, times (x - baseline)
        ig = (acc / steps) * diff
        total += np.abs(ig).sum(axis=0)
        n += len(xb)
        del acc, ig; gc.collect()

    per_feature = (total / n).mean(axis=0)        # mean over timesteps
    return per_feature


def ig_vector(model, building, schema, electricity, weather_all, meter_dfs,
              subsample_seed=0, n_samples=IG_SAMPLES):
    """Collapsed IG vector for one building."""
    X_seq_b, y_seq_b, scaler_y_b, feat_cols_b, _ = prepare_building(
        building, electricity, weather_all, meter_dfs)
    X_aligned = align_to_schema(X_seq_b, feat_cols_b, schema)

    # independent subsample -> lets us measure the estimation ceiling
    if len(X_aligned) > n_samples:
        rng = np.random.default_rng(stable_seed(building) + subsample_seed)
        idx = np.sort(rng.choice(len(X_aligned), n_samples, replace=False))
        X_aligned = X_aligned[idx]

    raw = integrated_gradients(model, X_aligned)
    del X_seq_b, y_seq_b, X_aligned; gc.collect()
    return collapse_vec(raw)


# =====================================================================
# STEP 1 — CEILING. Two IG runs, ONE frozen model, independent subsamples
# =====================================================================
def ig_ceiling(model, model_key, buildings, schema,
               electricity, weather_all, meter_dfs, n_test=25):
    rows = []
    for i, b in enumerate(buildings[:n_test]):
        try:
            v1 = ig_vector(model, b, schema, electricity, weather_all,
                           meter_dfs, subsample_seed=0)
            v2 = ig_vector(model, b, schema, electricity, weather_all,
                           meter_dfs, subsample_seed=1)
            rows.append({'building': b,
                         'rho': spearmanr(v1, v2)[0],
                         'top1_a': COLLAPSED_LABELS[int(np.argmax(v1))],
                         'top1_b': COLLAPSED_LABELS[int(np.argmax(v2))],
                         'status': 'ok'})
        except Exception as e:
            rows.append({'building': b, 'status': f'{type(e).__name__}: {e}'})
        if (i+1) % 5 == 0: print(f"  {i+1}/{n_test}")

    df = pd.DataFrame(rows); ok = df[df.status=='ok']
    print(f"\n{'='*64}\nIG ESTIMATION CEILING ({model_key}, n={len(ok)})\n{'='*64}")
    print(f"  median rho = {ok['rho'].median():.3f}  "
          f"[{ok['rho'].quantile(.25):.3f}, {ok['rho'].quantile(.75):.3f}]")
    print(f"  top-1 identical: {100*(ok.top1_a==ok.top1_b).mean():.0f}%")
    print("\n  Reference ceilings: permutation LSTM 0.971, TFT 0.986")
    if ok['rho'].median() < 0.90:
        print("  !! IG's own estimate is noisy. Raise IG_SAMPLES or IG_STEPS")
        print("     BEFORE interpreting any across-draw number.")
    return df


# =====================================================================
# STEP 2 — ACROSS DRAWS (only meaningful once the ceiling is known)
# =====================================================================
def ig_across_draws(draw_models, arch, buildings, schema,
                    electricity, weather_all, meter_dfs, n_test=25):
    keys = [f'{arch}_draw{d}' for d in (1, 2, 3)]
    hod_i = COLLAPSED_LABELS.index('Hour of Day')
    rows = []
    for i, b in enumerate(buildings[:n_test]):
        row = {'building': b}
        try:
            v = {k: ig_vector(draw_models[k], b, schema,
                              electricity, weather_all, meter_dfs)
                 for k in keys}
            pairs = [(keys[0],keys[1]),(keys[0],keys[2]),(keys[1],keys[2])]
            row['mean_rho'] = float(np.nanmean(
                [spearmanr(v[a], v[c])[0] for a, c in pairs]))
            # sub-hierarchy with Hour of Day removed — same saturation
            # guard we applied to SHAP
            m = np.ones(len(COLLAPSED_LABELS), bool); m[hod_i] = False
            row['sub_rho'] = float(np.nanmean(
                [spearmanr(v[a][m], v[c][m])[0] for a, c in pairs]))
            for k in keys:
                row[f'top1_{k}'] = COLLAPSED_LABELS[int(np.argmax(v[k]))]
            vv = v[keys[0]]; o = np.argsort(vv)[::-1]
            row['dominance_ratio'] = vv[o[0]]/vv[o[1]] if vv[o[1]] > 0 else np.inf
            row['rank1_share']     = vv[o[0]]/vv.sum() if vv.sum() > 0 else np.nan
            row['top1_all_same']   = len({row[f'top1_{k}'] for k in keys}) == 1
            row['status'] = 'ok'
        except Exception as e:
            row['status'] = f'{type(e).__name__}: {e}'
        rows.append(row)
        if (i+1) % 5 == 0: print(f"  {i+1}/{n_test}")

    df = pd.DataFrame(rows); ok = df[df.status=='ok']
    print(f"\n{'='*64}\nIG ACROSS DRAWS ({arch}, n={len(ok)})\n{'='*64}")
    print(f"  across-draw rho     : {ok['mean_rho'].median():.3f}  "
          f"[{ok['mean_rho'].quantile(.25):.3f}, {ok['mean_rho'].quantile(.75):.3f}]")
    print(f"  sub-hierarchy rho   : {ok['sub_rho'].median():.3f}   (Hour of Day removed)")
    print(f"  top-1 same all draws: {100*ok['top1_all_same'].mean():.0f}%")
    print(f"  dominance ratio     : {ok['dominance_ratio'].median():.2f}x")
    print(f"  rank-1 share        : {ok['rank1_share'].median():.3f}")
    for k in keys:
        vc = ok[f'top1_{k}'].value_counts()
        print(f"  {k}: " + ", ".join(f"{f} {100*c/len(ok):.0f}%" for f,c in vc.items()))

    print("\n  COMPARISON TABLE (LSTM, across draws):")
    print("    method        rho     ceiling   dom.ratio  rank1 share")
    print(f"    SHAP          0.984   ~1.0      1.35x      0.262")
    print(f"    permutation   0.696   0.971     2.52x      0.607")
    print(f"    IG            {ok['mean_rho'].median():.3f}   [see step 1] "
          f"{ok['dominance_ratio'].median():.2f}x      {ok['rank1_share'].median():.3f}")

    m = ok['mean_rho'].median()
    print("\nVERDICT:")
    if m >= 0.95:
        print("  IG is draw-STABLE, like SHAP. => The split is GRADIENT vs")
        print("  PERTURBATION, not SHAP-specific. Two independent gradient")
        print("  methods agree across initialisations; the perturbation")
        print("  method does not. This is the method-FAMILY claim.")
    elif m <= 0.80:
        print("  IG is draw-UNSTABLE, like permutation. => NOT a family split.")
        print("  SHAP's stability is the anomaly and needs its own explanation.")
        print("  Check whether SHAP's GradientExplainer background (fixed")
        print("  anchor X_train[:200]) is imposing the stability.")
    else:
        print(f"  IG is INTERMEDIATE (rho {m:.3f}). Report the ordering")
        print("  SHAP > IG > permutation and resist a clean binary story.")
    return df


# ── RUN ORDER — ceiling first, always ────────────────────────────────
ig_cal = ig_ceiling(DRAW_MODELS['lstm_draw1'], 'lstm_draw1', subsample,
                    ANCHOR_FEATURE_SCHEMA, electricity, weather_all, meter_dfs)
ig_cal.to_csv(os.path.join(PHASE2_DIR,'results','analysis3_ig_ceiling.csv'), index=False)

ig_draws = ig_across_draws(DRAW_MODELS, 'lstm', subsample,
                           ANCHOR_FEATURE_SCHEMA, electricity, weather_all, meter_dfs)
ig_draws.to_csv(os.path.join(PHASE2_DIR,'results','analysis3_ig_draws_lstm.csv'), index=False)


# In[52]:


rng = np.random.default_rng(2016)
sample25 = list(rng.choice(retained, 25, replace=False))

strat_test = stratification_sweep(
    DRAW_MODELS['lstm_draw1'], 'lstm_draw1',
    sample25, ANCHOR_FEATURE_SCHEMA,
    electricity, weather_all, meter_dfs,
    out_path=Path(PHASE2_DIR)/'results'/'a4_shapig_meanbaseline_200steps_n25.csv')

print("\nSHAP vs IG cross-method agreement (mean baseline, 200 steps, n=25 random):")
print(strat_test['shap_ig_rho'].describe())
print("\nrank-3 by method (where they diverged before):")
print("  SHAP:", strat_test['shap_rank3'].value_counts().head(3).to_dict())
print("  IG:  ", strat_test['ig_rank3'].value_counts().head(3).to_dict())


# In[54]:


# Sanity-check the config is what we think before timing anything
print(f"IG_STEPS = {IG_STEPS}  (expect 200)")
print(f"IG_SAMPLES = {IG_SAMPLES}  (expect 200)")

# 5 RANDOM buildings (not retained[:5], which is the anchor + neighbours)
rng = np.random.default_rng(7)          # different seed from the n=25 run
timer5 = list(rng.choice(retained, 5, replace=False))
print(f"Timing on: {timer5}\n")

t0 = time.time()
timer_df = stratification_sweep(
    DRAW_MODELS['lstm_draw1'], 'lstm_draw1',
    timer5, ANCHOR_FEATURE_SCHEMA,
    electricity, weather_all, meter_dfs,
    out_path=Path(PHASE2_DIR)/'results'/'a4_TIMER_DELETE_ME.csv')
elapsed = time.time() - t0

per_building = elapsed / 5
full_n = len(retained)
print(f"\n{'='*60}")
print(f"5 buildings in {elapsed:.0f}s  =>  {per_building:.1f}s/building")
print(f"Projected full sweep (n={full_n}): "
      f"{per_building*full_n/3600:.1f} hours")
print(f"{'='*60}")
print("  < 20h  -> one overnight, launch it")
print("  20-30h -> drop IG_STEPS to 100 (completeness still ~0.04) or split")
print("  > 30h  -> SHAP full-pop, IG on the 319 subsample")

# confirm the rows are actually valid, not silently erroring
print(f"\nstatus: {timer_df['status'].value_counts().to_dict()}")
print(f"shap_ig_rho present: {timer_df['shap_ig_rho'].notna().sum()}/5")


# In[59]:


# ── Config guard — confirm before committing the night ───────────────
print(f"IG_STEPS   = {IG_STEPS}    (expect 200)")
print(f"IG_SAMPLES = {IG_SAMPLES}    (expect 200)")
print(f"SHAP_SAMPLES = {SHAP_SAMPLES}  (expect 200)")
assert IG_STEPS == 200, "IG_STEPS is not 200 — re-run the config cell first"

# baseline check: integrated_gradients should use the TRAINING MEAN.
# (This just prints the default the function will apply, so you can eyeball it.)
_src = inspect.getsource(integrated_gradients)
print("baseline line:", [l.strip() for l in _src.split(chr(10)) if 'baseline' in l and '=' in l][:2])

FULL_CSV = Path(PHASE2_DIR) / 'results' / 'analysis4_shapig_meanbaseline_200steps_FULL.csv'
print(f"\nOutput -> {FULL_CSV}")
print(f"Population: {len(retained)} Gate-2-retained buildings")
print(f"Already done (resume): "
      f"{len(set(__import__('pandas').read_csv(FULL_CSV)['building'])) if FULL_CSV.exists() else 0}")

# ── FULL SWEEP — SHAP + IG, LSTM draw 1, full population ──────────────
# One draw is sufficient: gradient attribution is draw-invariant
# (SHAP 0.984, IG 0.944). Resumes automatically if interrupted.
strat_df = stratification_sweep(
    DRAW_MODELS['lstm_draw1'], 'lstm_draw1',
    retained, ANCHOR_FEATURE_SCHEMA,
    electricity, weather_all, meter_dfs,
    out_path=FULL_CSV)

print(f"\n{'='*60}\nSWEEP COMPLETE\n{'='*60}")
print(f"rows: {len(strat_df)}  ok: {(strat_df.status=='ok').sum()}  "
      f"failed: {(strat_df.status!='ok').sum()}")
print(f"SHAP-vs-IG rho (full pop): "
      f"median {strat_df[strat_df.status=='ok']['shap_ig_rho'].median():.3f}")


# In[60]:


res = gradient_stratification_analysis(strat_df, building_summary)
res.to_csv(Path(PHASE2_DIR)/'results'/'analysis4_stratification_analysed.csv', index=False)


# ## Diagnostic: distinguishing zero-fill artefacts from genuine zero attribution
# 
# GradientExplainer returns exactly zero SHAP for some features. The two checks below establish that this arises from two distinct causes: chilled water is zero because the building lacks that meter (schema alignment zero-fills it, a working control), whereas Month is zero because the trained model places no weight on it (the feature is present and varying). Only the second is a genuine model property; it is the clearest single instance of the SHAP–IG baseline divergence, since IG assigns Month a non-zero contribution where SHAP assigns exactly zero.

# In[56]:


b = 'Panther_parking_Lorriane'
X_seq_b, y_seq_b, sc, fc, _ = prepare_building(b, electricity, weather_all, meter_dfs)
Xa = align_to_schema(X_seq_b, fc, ANCHOR_FEATURE_SCHEMA)
expl = shap.GradientExplainer(DRAW_MODELS['lstm_draw1'], X_train[:200])
sv = np.array(expl.shap_values(Xa[:200]))
if sv.ndim == 4: sv = sv.squeeze(-1)
raw = np.mean(np.abs(sv), axis=(0,1))
for f, v in zip(ANCHOR_FEATURE_SCHEMA, raw):
    print(f"{f:18s} {v:.10f}")


# In[57]:


b = 'Panther_parking_Lorriane'
_, _, _, fc, _ = prepare_building(b, electricity, weather_all, meter_dfs)
print("chilledwater present:", 'chilledwater' in fc)   # expect False
print("month_sin present:", 'month_sin' in fc)          # expect True


# In[62]:


d = pd.read_csv(os.path.join(PHASE2_DIR, 'results',
                'analysis4_shapig_meanbaseline_200steps_FULL.csv'))
ok = d[d.status == 'ok']

LAB = ['Hour of Day','Day of Week','Month','airTemperature','dewTemperature',
       'windDirection','windSpeed','is_weekend','chilledwater','gas','hotwater','solar']
sub = [l for l in LAB if l != 'Hour of Day']

rho_sub = [spearmanr([r[f'shap_{l}'] for l in sub],
                     [r[f'ig_{l}']   for l in sub])[0] for _, r in ok.iterrows()]

print(f"n = {len(ok)}")
print(f"full-12 rho : {ok['shap_ig_rho'].median():.3f}")
print(f"sans-HoD    : {np.nanmedian(rho_sub):.3f}")
print(f"change      : {np.nanmedian(rho_sub) - ok['shap_ig_rho'].median():+.3f}")

