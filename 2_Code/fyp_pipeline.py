"""
fyp_pipeline.py
===============
Shared preprocessing pipeline for LSTM-VCA Phase 1 and Phase 2 analyses.

Extracted verbatim from FYP_MultiSite_BDGP2_SHAP_CPU.ipynb (Sections 5 and 8).
Do NOT modify this file without updating both Phase 1 and Phase 2 notebooks
to reflect the change — any divergence between this module and the Phase 1
notebook invalidates the "identical preprocessing" control claim in Analysis 1.

Last verified identical to Phase 1 notebook: 2026-07-09
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ── Constants (must match Phase 1 notebook globals) ───────────────────────────
YEAR              = 2016
WINDOW_SIZE       = 48
METER_MISS_THRESH = 0.05   # max missing fraction for a meter covariate

WEATHER_COLS_ALL = [
    'airTemperature', 'dewTemperature',
    'windDirection', 'windSpeed'
]

TIME_COLS = [
    'hour_sin', 'hour_cos',
    'dow_sin',  'dow_cos',
    'month_sin','month_cos',
    'is_weekend'
]

# ── Utility ───────────────────────────────────────────────────────────────────

def infer_site(building_name: str) -> str:
    """Extract site prefix from BDGP2 building name (e.g. 'Bobcat' from 'Bobcat_education_Alissa')."""
    return building_name.split('_')[0]


# ── Core preprocessing function ───────────────────────────────────────────────

def prepare_building(building_name, electricity_df, weather_df,
                     meter_dfs_dict=None, year=YEAR, window=WINDOW_SIZE,
                     meter_miss_thresh=METER_MISS_THRESH):
    """
    Prepare scaled sequences for any BDGP2 building.

    Loads electricity as target, merges site weather, then adds every
    available meter from meter_dfs_dict as a covariate (gas, hotwater,
    chilledwater, steam, solar, water, irrigation).

    Feature order: WEATHER_COLS_ALL + TIME_COLS + [available meters sorted]
    This order is locked by the anchor building and all other buildings
    align to it via align_to_schema().

    DATA LEAKAGE FIX: scalers are fitted on training data only (first 80%),
    then used to transform the full sequence array. This prevents test set
    statistics from influencing the scaling applied during training.

    Parameters
    ----------
    building_name     : str   — BDGP2 building identifier
    electricity_df    : pd.DataFrame — full electricity table (all buildings)
    weather_df        : pd.DataFrame — full weather table (all sites)
    meter_dfs_dict    : dict[str, pd.DataFrame] — meter name → DataFrame
    year              : int   — calendar year to filter (default 2016)
    window            : int   — lookback window in hours (default 48)
    meter_miss_thresh : float — max missing fraction for a meter covariate

    Returns
    -------
    X_seq        : np.ndarray, shape (N, window, n_features)
    y_seq        : np.ndarray, shape (N,)
    scaler_y     : fitted MinMaxScaler for target inverse-transform
    feature_cols : list[str]  — ordered feature names (matches X_seq axis-2)
    raw_y_mean   : float      — mean raw electricity consumption (kWh)
    """
    if meter_dfs_dict is None:
        meter_dfs_dict = {}

    site = infer_site(building_name)

    # ── Electricity ───────────────────────────────────────────────────────────
    target = electricity_df[['timestamp', building_name]].copy()
    target = target[target['timestamp'].dt.year == year].reset_index(drop=True)
    target = target.rename(columns={building_name: 'electricity'})

    # ── Site weather ──────────────────────────────────────────────────────────
    wx = weather_df[weather_df['site_id'] == site].drop(columns='site_id')
    wx = wx[wx['timestamp'].dt.year == year].reset_index(drop=True)
    merged = target.merge(wx, on='timestamp', how='left')

    # ── Meter covariates (all available) ──────────────────────────────────────
    meters_included = []
    for meter_name in sorted(meter_dfs_dict.keys()):   # sorted = consistent order
        df_m = meter_dfs_dict[meter_name]
        if building_name not in df_m.columns:
            continue
        g = df_m[['timestamp', building_name]].copy()
        g = g[g['timestamp'].dt.year == year].reset_index(drop=True)
        g = g.rename(columns={building_name: meter_name})
        merged = merged.merge(g, on='timestamp', how='left')
        miss = merged[meter_name].isnull().mean()
        if miss <= meter_miss_thresh:
            meters_included.append(meter_name)
        else:
            merged = merged.drop(columns=meter_name)

    # ── Drop high-miss weather columns ────────────────────────────────────────
    drop_wx = ['precipDepth6HR', 'seaLvlPressure', 'precipDepth1HR', 'cloudCoverage']
    merged  = merged.drop(columns=[c for c in drop_wx if c in merged.columns])
    merged  = merged.ffill().bfill()

    # ── Cyclical time features ────────────────────────────────────────────────
    ts = merged['timestamp']
    merged['hour_sin']   = np.sin(2 * np.pi * ts.dt.hour      / 24)
    merged['hour_cos']   = np.cos(2 * np.pi * ts.dt.hour      / 24)
    merged['dow_sin']    = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
    merged['dow_cos']    = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
    merged['month_sin']  = np.sin(2 * np.pi * ts.dt.month     / 12)
    merged['month_cos']  = np.cos(2 * np.pi * ts.dt.month     / 12)
    merged['is_weekend'] = (ts.dt.dayofweek >= 5).astype(float)

    # ── Assemble feature list (fixed order: weather → time → meters) ──────────
    weather_present = [c for c in WEATHER_COLS_ALL if c in merged.columns]
    feature_cols    = weather_present + TIME_COLS + meters_included

    # ── Raw arrays ────────────────────────────────────────────────────────────
    X_data = merged[feature_cols].values
    y_data = merged['electricity'].values.reshape(-1, 1)

    # ── Chronological train/val/test split BEFORE scaling ─────────────────────
    n_rows    = len(X_data)
    train_end = int(n_rows * 0.80)

    # ── Fit scalers on TRAINING DATA ONLY, transform all ─────────────────────
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    scaler_X.fit(X_data[:train_end])
    scaler_y.fit(y_data[:train_end])

    X_scaled = scaler_X.transform(X_data)
    y_scaled = scaler_y.transform(y_data).flatten()

    # ── Sequence windows ──────────────────────────────────────────────────────
    Xs, ys = [], []
    for i in range(len(X_scaled) - window):
        Xs.append(X_scaled[i : i + window])
        ys.append(y_scaled[i + window])

    return (np.array(Xs), np.array(ys),
            scaler_y, feature_cols,
            merged['electricity'].mean())


# ── Schema alignment ──────────────────────────────────────────────────────────

def align_to_schema(X_seq, feature_cols_used, schema):
    """
    Reorder / zero-fill a building's X_seq to match the anchor schema.

    Features absent from the target building receive a column of zeros
    (neutral / mean-scaled value). Features present are placed at their
    schema-defined index position.

    Parameters
    ----------
    X_seq            : np.ndarray, shape (N, window, n_features_used)
    feature_cols_used: list[str] — feature names in X_seq axis-2 order
    schema           : list[str] — anchor feature schema (defines output order)

    Returns
    -------
    X_aligned : np.ndarray, shape (N, window, len(schema)), dtype float32
    """
    n_samples, window, _ = X_seq.shape
    X_aligned = np.zeros((n_samples, window, len(schema)), dtype=np.float32)
    for i, feat in enumerate(schema):
        if feat in feature_cols_used:
            j = feature_cols_used.index(feat)
            X_aligned[:, :, i] = X_seq[:, :, j]
    return X_aligned


# ── Train/val/test split helper ───────────────────────────────────────────────

def split_sequences(X_seq, y_seq, train_frac=0.80, val_frac=0.10):
    """
    Chronological 80/10/10 split of windowed sequences.

    Parameters
    ----------
    X_seq      : np.ndarray, shape (N, window, features)
    y_seq      : np.ndarray, shape (N,)
    train_frac : float — fraction for training (default 0.80)
    val_frac   : float — fraction for validation (default 0.10)

    Returns
    -------
    X_tr, y_tr, X_vl, y_vl, X_te, y_te : np.ndarray pairs
    """
    n  = len(X_seq)
    te = int(n * train_frac)
    ve = int(n * (train_frac + val_frac))
    return (X_seq[:te],   y_seq[:te],
            X_seq[te:ve], y_seq[te:ve],
            X_seq[ve:],   y_seq[ve:])