#!/usr/bin/env python
r"""
rerun_tft_confirmation.py
==========================
Faithful re-run of the "Top-5 confirmation: 3 runs each" step from
FYP_Phase2_TFT_FINAL.py (cell "In[16]" in the notebook), for the audit
question of where main.tex's 19.06%/17.79% TFT confirmation figures came
from. Neither number is reproducible from anything saved in C:\\Phase2 —
see audit_ledger.md §0.3. This script produces a genuine, dated, fully
reproducible replacement.

WHAT THIS DOES
--------------
Rebuilds the anchor building's (Bobcat_education_Alissa) training data
exactly as the original script did, then re-runs the 3-seeded-run
confirmation for whichever of the original top-5 grid configs you select
via RANKS_TO_RUN below (configs pulled from tft_extended_grid_log.csv,
same order as the original, so there's no risk of drift). Defaults to
RANKS_TO_RUN = [2] — just the config main.tex cites (P9) — because this
machine has no GPU (see check_versions.py output) and the full 5-config
sweep is a multi-day job on CPU alone. Widen RANKS_TO_RUN to [1,2,3,4] or
[1,2,3,4,5] if you have the time, which also refreshes P8's top-4 spread.

WHAT IT DOES NOT DO
--------------------
It does NOT overwrite tft_top5_confirmation_log.csv or any other existing
file — the 2026-07-21 result is evidence and should stay untouched. Output
goes to a new, timestamped file: tft_top5_confirmation_RERUN_<UTC date+time>.csv

HOW TO RUN (this machine: Windows + conda, not the original WSL2 box)
----------------------------------------------------------------------
1. Run check_versions.py first and compare against the printed reference
   (Python 3.11.15 / TF 2.20.0 / RTX 4070 SUPER) — this is a DIFFERENT
   computer, so expect some drift (GPU model, driver, maybe TF patch
   version); note what you see rather than assuming a match.
2. Paths are set below for this machine: PHASE2_DIR = C:\Phase2 (this
   file's own folder, which already has fyp_pipeline.py in it),
   PHASE1_DIR = C:\Phase2\Phase1DataKeep, and DATA_DIR pointed directly
   at C:\Users\pauli\OneDrive\Desktop\Dump\Uni\Y4\S1\ZEIT4500 - FYP\Data
   (the raw electricity/weather/meter CSVs don't live under Phase1DataKeep
   on this machine, so DATA_DIR is set independently rather than derived
   from PHASE1_DIR). Not verified from this end — outside the connected
   C:\Phase2 folder, so the preflight check below is what actually
   confirms it on your machine, with a clear message if anything's off.
3. Run in your conda environment:
       python rerun_tft_confirmation.py

   ** IMPORTANT — CHECK YOUR check_versions.py OUTPUT FIRST **
   If it printed "GPUs visible : []" (no GPU), the time estimate below is
   wrong by a large factor. The original 5-config x 3-run confirmation
   took roughly 4 hours end-to-end on an RTX 4070 SUPER (from
   tft_extended_grid_log.csv's own screening epoch times: ~4-15s/epoch,
   70-150 epochs to convergence per run, x15 runs total). CPU-only
   training of an attention+LSTM model like this is commonly 10-40x
   slower per epoch, which puts a CPU-only full run at very roughly
   1.5-6 DAYS — likely impractical to just leave running.
   By default (see CONFIGS_TO_RUN below) this script only re-runs RANK 2
   — the one config main.tex actually cites — cutting that to roughly
   1/5th: still very roughly 3-12 hours on CPU alone for just that one
   config's 3 runs. Treat this as an overnight-or-longer background job
   on CPU, not something to sit and watch. If even that's too slow,
   either reduce MAX_EPOCHS/PATIENCE below (documented in the output as
   a deviation from the original protocol, still a valid fresh number,
   just not a bit-for-bit reproduction of the original run length), or
   find GPU access (a free Colab GPU runtime, a university lab machine,
   a rented cloud GPU) and run there instead.
4. When it finishes, the new CSV plus the printed summary is what to send
   back for the ledger update.

Everything below the imports is copied as closely as possible from
FYP_Phase2_TFT_FINAL.py so this is a genuine re-run of the same
procedure, not a reimplementation.
"""
import sys, os, random, joblib, datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED']       = '2016'

import warnings
warnings.filterwarnings('ignore', message=".*does not support masking.*")
warnings.filterwarnings('ignore', message='The structure of `inputs` doesn')

# ── Reproducibility ────────────────────────────────────────────────────
import numpy as np
SEED = 2016
random.seed(SEED)
np.random.seed(SEED)

# ── Core imports ───────────────────────────────────────────────────────
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except RuntimeError as e:
        print("memory_growth must be set before GPU init:", e)
print("GPU memory growth enabled:", [g.name for g in gpus])

import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error

tf.random.set_seed(SEED)

# ── fyp_pipeline (unchanged — do not edit this file, per its own header) ──
# Windows/conda paths for this machine (adjust if anything differs):
PHASE2_DIR = r'C:\Phase2'
PHASE1_DIR = r'C:\Phase2\Phase1DataKeep'
# Raw BDGP2 data — supplied directly, since it doesn't live under
# PHASE1_DIR on this machine the way it did in the original layout:
DATA_DIR   = r'C:\Users\pauli\OneDrive\Desktop\Dump\Uni\Y4\S1\ZEIT4500 - FYP\Data'

# ── Preflight check: fail fast with a clear message rather than a deep,
#    confusing pandas traceback if the raw BDGP2 data isn't where expected.
#    As of this script being written, C:\Phase2\Phase1DataKeep contains
#    only the Phase 1 provenance artefacts (models, summary CSVs,
#    requirements_frozen.txt) — NOT the raw electricity/weather/meter
#    CSVs this script needs. Point DATA_DIR at wherever those actually
#    live on this machine before running.
_REQUIRED_DATA_FILES = ['electricity.csv', 'weather.csv']
_missing = [f for f in _REQUIRED_DATA_FILES if not os.path.exists(os.path.join(DATA_DIR, f))]
if not os.path.isdir(PHASE2_DIR):
    sys.exit(f"PHASE2_DIR not found: {PHASE2_DIR}\nFix the PHASE2_DIR constant above.")
if not os.path.isfile(os.path.join(PHASE2_DIR, 'fyp_pipeline.py')):
    sys.exit(f"fyp_pipeline.py not found in PHASE2_DIR ({PHASE2_DIR}).\n"
             f"This script must sit alongside fyp_pipeline.py, or PHASE2_DIR must "
             f"point at the folder that contains it.")
if not os.path.isdir(DATA_DIR) or _missing:
    sys.exit(
        f"Raw BDGP2 data not found at DATA_DIR = {DATA_DIR}\n"
        f"Missing: {_missing if os.path.isdir(DATA_DIR) else _REQUIRED_DATA_FILES}\n\n"
        f"This script needs electricity.csv, weather.csv, metadata.csv, and the meter "
        f"CSVs (gas/hotwater/chilledwater/steam/solar/water/irrigation) — the same raw "
        f"BDGP2 files fyp_pipeline.py's prepare_building() consumes. C:\\Phase2\\Phase1DataKeep "
        f"currently holds only the Phase 1 provenance artefacts, not this raw data.\n"
        f"Point DATA_DIR (near the top of this script) at wherever that data actually "
        f"lives on this machine, then re-run."
    )

sys.path.insert(0, PHASE2_DIR)
from fyp_pipeline import prepare_building, split_sequences  # noqa: E402

print(sys.executable)
print(f"TF: {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")
print(f"SEED: {SEED}")
RUN_START = datetime.datetime.now(datetime.timezone.utc)
print(f"RUN START (UTC): {RUN_START.isoformat()}")


# =====================================================================
# Model definition — copied verbatim from FYP_Phase2_TFT_FINAL.py
# (Gated Residual Network / Variable Selection Network / Interpretable
# Multi-Head Attention / causal mask / build_tft). Do not modify — the
# point of this rerun is to use the SAME model, not a reimplementation.
# =====================================================================

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


class VariableSelectionNetwork(layers.Layer):
    """Per-timestep feature weighting."""
    def __init__(self, n_features, units, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.units = units
        self.feature_grns = [GRN(units, dropout) for _ in range(n_features)]
        self.weight_grn = GRN(n_features, dropout)
        self.softmax = layers.Softmax(axis=-1)

    def call(self, x, training=None):
        flat = x
        weights = self.weight_grn(flat, training=training)
        weights = self.softmax(weights)
        transformed = []
        for j in range(self.n_features):
            feat_j = x[..., j:j+1]
            transformed.append(self.feature_grns[j](feat_j, training=training))
        stacked = tf.stack(transformed, axis=-1)
        weights_exp = tf.expand_dims(weights, axis=-2)
        selected = tf.reduce_sum(stacked * weights_exp, axis=-1)
        return selected, weights


class InterpretableMultiHeadAttention(layers.Layer):
    """TFT attention: separate Q/K per head, but V is SHARED across heads."""
    def __init__(self, d_model, n_heads, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_dense = layers.Dense(d_model)
        self.k_dense = layers.Dense(d_model)
        self.v_dense = layers.Dense(self.d_head)
        self.out_dense = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout)

    def call(self, x, mask=None, training=None):
        B, T = tf.shape(x)[0], tf.shape(x)[1]
        q = tf.reshape(self.q_dense(x), (B, T, self.n_heads, self.d_head))
        k = tf.reshape(self.k_dense(x), (B, T, self.n_heads, self.d_head))
        v = self.v_dense(x)

        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(self.d_head, tf.float32))
        if mask is not None:
            scores += (1.0 - mask) * -1e9
        attn = tf.nn.softmax(scores, axis=-1)
        attn = self.dropout(attn, training=training)
        attn_mean = tf.reduce_mean(attn, axis=1)
        out = tf.matmul(attn_mean, v)
        return self.out_dense(out), attn_mean


def causal_mask(T):
    m = tf.linalg.band_part(tf.ones((T, T)), -1, 0)
    return m[tf.newaxis, tf.newaxis, :, :]


def build_tft(seq_len=48, n_features=15, d_model=32, n_heads=4, lstm_units=(64, 32), dropout=0.2):
    inputs = layers.Input(shape=(seq_len, n_features))

    vsn = VariableSelectionNetwork(n_features, d_model, dropout)
    selected, vsn_weights = vsn(inputs)

    x = layers.LSTM(lstm_units[0], return_sequences=True, dropout=dropout)(selected)
    x = layers.LSTM(lstm_units[1], return_sequences=True, dropout=dropout)(x)

    attn_layer = InterpretableMultiHeadAttention(d_model=lstm_units[1], n_heads=n_heads, dropout=dropout)
    attn_out, attn_weights = attn_layer(x, mask=causal_mask(seq_len))

    grn_out = GRN(lstm_units[1], dropout)(attn_out)
    final_step = grn_out[:, -1, :]
    output = layers.Dense(1, activation='linear')(final_step)

    model = tf.keras.Model(inputs, output)
    attribution_model = tf.keras.Model(inputs, [output, vsn_weights, attn_weights])
    return model, attribution_model


# =====================================================================
# Data — anchor building, same as every other Phase 2 TFT run
# =====================================================================
elec_path   = os.path.join(DATA_DIR, 'electricity.csv')
electricity = pd.read_csv(elec_path, parse_dates=['timestamp'])
print(f"Electricity   : {electricity.shape}")

wx_path     = os.path.join(DATA_DIR, 'weather.csv')
weather_all = pd.read_csv(wx_path, parse_dates=['timestamp'])
print(f"Weather       : {weather_all.shape}")

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

BUILDING_ID = 'Bobcat_education_Alissa'   # ANCHOR_BUILDING, unchanged

X_seq, y_seq, scaler_y, feature_cols, raw_y_mean = prepare_building(
    BUILDING_ID, electricity, weather_all, meter_dfs
)
print(f"Feature schema ({len(feature_cols)}): {feature_cols}")
print(f"Raw mean consumption: {raw_y_mean:.2f} kWh")

X_train, y_train, X_val, y_val, X_test, y_test = split_sequences(
    X_seq, y_seq, train_frac=0.80, val_frac=0.10
)
print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

N_ANCHOR_FEATURES = len(feature_cols)
assert N_ANCHOR_FEATURES == 15, f"Expected 15 features, got {N_ANCHOR_FEATURES}"
WINDOW_SIZE = X_train.shape[1]
print(f"WINDOW_SIZE (from data): {WINDOW_SIZE}")


# =====================================================================
# The original top-5 configs, pulled from tft_extended_grid_log.csv
# (sorted by val_rmse) so this is the exact same 5 configs, same order,
# as the 2026-07-21 run recorded in tft_top5_confirmation_log.csv —
# no risk of re-deriving a different top-5 by accident.
# =====================================================================
TOP5_CONFIGS = [
    dict(d_model=16, n_heads=4, dropout=0.3, lstm_units=(32, 16), batch_size=64,  screening_val_rmse=11.860699399890825),  # rank 1
    dict(d_model=64, n_heads=4, dropout=0.2, lstm_units=(64, 32), batch_size=64,  screening_val_rmse=11.883155451762402),  # rank 2 <- the one main.tex cites
    dict(d_model=16, n_heads=4, dropout=0.1, lstm_units=(32, 16), batch_size=64,  screening_val_rmse=11.890040593516623),  # rank 3
    dict(d_model=64, n_heads=4, dropout=0.1, lstm_units=(32, 16), batch_size=256, screening_val_rmse=11.99344517558721),   # rank 4
    dict(d_model=64, n_heads=4, dropout=0.2, lstm_units=(64, 32), batch_size=256, screening_val_rmse=12.090910947023144),  # rank 5
]

# =====================================================================
# Scope + protocol knobs
# =====================================================================
# Which ranks to (re)confirm, by 1-indexed rank (1-5). Defaults to JUST
# rank 2 — the config main.tex actually cites (P9) — since a CPU-only
# machine makes the full 5-config sweep (needed for P8's top-4 spread)
# a multi-day job. Change to e.g. [1, 2, 3, 4] or [1, 2, 3, 4, 5] if you
# have the time/hardware to also refresh P8.
RANKS_TO_RUN = [1, 2, 3, 4]

# Faithful to the original protocol by default. Lower these ONLY if
# CPU wall-clock time makes the full protocol impractical — doing so
# is still a valid, honestly-reportable fresh number, just say so
# alongside it (the script records these values in the output CSV
# either way, so the deviation is self-documenting).
MAX_EPOCHS = 300
PATIENCE   = 15

N_RUNS = 3
configs_to_run = [(r, TOP5_CONFIGS[r - 1]) for r in RANKS_TO_RUN]
print(f"\nRe-confirming rank(s) {RANKS_TO_RUN} of the original top-5 "
      f"(MAX_EPOCHS={MAX_EPOCHS}, PATIENCE={PATIENCE}, N_RUNS={N_RUNS})")

confirmation_results = []

for rank_1indexed, cfg in configs_to_run:
    rank = rank_1indexed - 1
    arch_kwargs = dict(d_model=cfg['d_model'], n_heads=cfg['n_heads'],
                        dropout=cfg['dropout'], lstm_units=cfg['lstm_units'])
    run_maes = []
    for run in range(N_RUNS):
        tf.random.set_seed(SEED)
        model, _ = build_tft(seq_len=WINDOW_SIZE, n_features=N_ANCHOR_FEATURES, **arch_kwargs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        es = EarlyStopping(patience=PATIENCE, restore_best_weights=True, monitor='val_loss')
        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=MAX_EPOCHS, batch_size=cfg['batch_size'], callbacks=[es], verbose=0)

        pred_kwh = scaler_y.inverse_transform(model.predict(X_test, verbose=0)).flatten()
        true_kwh = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        mae_pct = 100 * mean_absolute_error(true_kwh, pred_kwh) / raw_y_mean
        run_maes.append(mae_pct)
        print(f"  rank {rank+1} run {run+1}/{N_RUNS}: {mae_pct:.4f}%")

    mean_mae, std_mae = float(np.mean(run_maes)), float(np.std(run_maes))
    confirmation_results.append(dict(
        rank=rank + 1, **arch_kwargs, batch_size=cfg['batch_size'],
        screening_val_rmse=cfg['screening_val_rmse'],
        confirmed_mean=mean_mae, confirmed_std=std_mae, runs=run_maes,
        max_epochs=MAX_EPOCHS, patience=PATIENCE,
        rerun_utc_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    ))
    print(f"Rank {rank+1} (screening val_RMSE={cfg['screening_val_rmse']:.3f}): "
          f"confirmed test MAE% = {mean_mae:.2f}% ± {std_mae:.2f}%  (runs: {run_maes})")

# ── Save to a NEW, timestamped file — never overwrite the original ────
stamp = RUN_START.strftime('%Y-%m-%dT%H%M%SZ')
out_path = os.path.join(PHASE2_DIR, f'tft_top5_confirmation_RERUN_{stamp}.csv')
pd.DataFrame(confirmation_results).to_csv(out_path, index=False)

RUN_END = datetime.datetime.now(datetime.timezone.utc)
print("\n" + "=" * 70)
print(f"RUN START (UTC): {RUN_START.isoformat()}")
print(f"RUN END   (UTC): {RUN_END.isoformat()}")
print(f"Elapsed         : {RUN_END - RUN_START}")
print(f"Saved -> {out_path}")
print("=" * 70)

by_rank = {r['rank']: r for r in confirmation_results}

if 2 in by_rank:
    r2 = by_rank[2]
    print("\nRank-2 result (the one main.tex's P9 cites, to replace 19.06%/17.79%):")
    print(f"  {r2['confirmed_mean']:.2f}% ± {r2['confirmed_std']:.2f}%  (runs: {r2['runs']})")
    print(f"  On record for the same config (2026-07-21, tft_top5_confirmation_log.csv):")
    print(f"    18.82% ± 0.11%")
else:
    print("\n(Rank 2 was not in RANKS_TO_RUN this time, so no fresh P9 number to report.)")

if all(r in by_rank for r in (1, 2, 3, 4)):
    print("\nTop-4 spread (the one main.tex's P8 cites — 0.47pp / 0.9-1.5pp claimed):")
    top4 = [by_rank[r] for r in (1, 2, 3, 4)]
    means = [r['confirmed_mean'] for r in top4]
    stds  = [r['confirmed_std'] for r in top4]
    print(f"  means: {[f'{m:.2f}' for m in means]}  -> span {max(means)-min(means):.2f}pp")
    print(f"  stds : {[f'{s:.2f}' for s in stds]}  -> range {min(stds):.2f}-{max(stds):.2f}pp")
else:
    print("\n(Ranks 1-4 weren't all run this time, so no fresh top-4 spread for P8. "
          "Set RANKS_TO_RUN = [1, 2, 3, 4] and re-run if you want that too.)")
