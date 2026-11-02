"""
A7 CONFIRMATION RUN — is the SHAP rank-2 identity draw-stable?
==============================================================

QUESTION
--------
Phase 1 weights on the anchor test partition put Weekend at SHAP rank 2
(0.00205) ahead of Day of Week (0.00133). The Analysis 4 sweep puts Day of
Week at rank 2 for 1,252/1,276 buildings and Weekend at rank 2 for ZERO of
them. Three things differ: weight draw, evaluation partition, and which
buildings were measured. This script removes two of the three so that only
the weight draw varies.

DESIGN
------
Held fixed: the anchor building, the 874-sequence test partition, the SHAP
background protocol (anchor X_train[:200]), the 12-group sin/cos collapse.
Varied: the model weights only.

Gates, in order of cost:
  GATE 0   fitted scaler_y must match archived scaler_y_bobcat.pkl (3.0/160.0)
  GATE 0b  feature order must match archived feature_cols_bobcat.pkl
  GATE 1   multisite_anchor_best.keras must give test MAE 10.659 kWh
  GATE 2   the same model must put Weekend above Day of Week
0 and 0b validate preprocessing without touching a model. 1 and 2 validate the
model and the SHAP protocol. If the Phase 1 model is not present on this
machine the run continues in DRAWS-ONLY mode, which can still answer the
stronger question of whether the draws disagree with each other.

SAFETY
------
Read-only with respect to every archived artefact. No .fit(), no .save(). One
new CSV is written. Do NOT Run All the Phase 2 notebook: its training cell
would overwrite lstm_refit_bobcat.keras (byte-identical to lstm_draw1.keras,
sha256 39a66b21...5860) and break the tie to every Analysis 1/3/4 number.

USAGE
-----
    conda activate fyp_env
    python a7_rank2_draw_check.py

Paths are discovered by search, so this runs on either machine unmodified.
"""

# ========================================================== SECTION 1: setup
import os

os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
# CPU only. A6 records a 5-point MAE% swing from GPU non-determinism and the
# effect under test is a rank swap between two mid-magnitude features.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
import zipfile
from pathlib import Path

import h5py
import io
import joblib
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from sklearn.metrics import mean_absolute_error

SEED = 2016
np.random.seed(SEED)
tf.random.set_seed(SEED)

ANCHOR = "Bobcat_education_Alissa"
N_SHAP = 200
EXPECT_SEQ = 8736   # 2016 is a LEAP year: 366*24 - 48. The paper's "8,712"
                    # assumed 8,760 hours and is an error; the notebook prints
                    # 6988/874/874, which only follows from 8,736.
EXPECT_TEST = 874
EXPECT_MAE = 10.659
ARCHIVE_NAME = "2026-07-21_analysis1"

print(f"python {sys.version.split()[0]} | TF {tf.__version__} | SHAP {shap.__version__}")
print(f"GPUs visible: {tf.config.list_physical_devices('GPU')}  (expect [])")

# ==================================================== SECTION 2: locate files
# Search upward from the script location so the same file works on the Windows
# OneDrive tree and the WSL2 box without editing paths.
HERE = Path(__file__).resolve().parent
SEARCH_ROOTS = [HERE, HERE.parent, HERE.parent.parent]


def find_one(pattern, what, required=True):
    """Return the first match for `pattern` under any search root."""
    hits = []
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        hits.extend(sorted(root.rglob(pattern)))
        if hits:
            break
    hits = list(dict.fromkeys(hits))
    if not hits:
        if required:
            sys.exit(
                f"STOP: could not find {what} (pattern '{pattern}') under any of:\n"
                + "\n".join(f"  {r}" for r in SEARCH_ROOTS)
            )
        print(f"  [absent] {what}")
        return None
    if len(hits) > 1:
        print(f"  [{len(hits)} copies] {what} -> using {hits[0]}")
        for h in hits[1:]:
            print(f"      also: {h}")
    else:
        print(f"  [found] {what} -> {hits[0]}")
    return hits[0]


print("\nLocating artefacts...")
PIPELINE_PY = find_one("fyp_pipeline.py", "fyp_pipeline.py")
ELEC_CSV = find_one("electricity.csv", "electricity.csv")
DATA_DIR = ELEC_CSV.parent
FEATCOLS_PKL = find_one("feature_cols_bobcat.pkl", "archived feature_cols")
SCALER_PKL = find_one("scaler_y_bobcat.pkl", "archived scaler_y")
P1_MODEL = find_one("multisite_anchor_best.keras", "Phase 1 anchor model",
                    required=False)

draw_hits = []
for root in SEARCH_ROOTS:
    if root.exists():
        draw_hits.extend(sorted(root.rglob("lstm_draw*.keras")))
    if draw_hits:
        break
# Deduplicate by content hash. The same draw exists in more than one folder
# (archive copy plus working copy), and counting a duplicate twice would
# inflate n and bias the unanimity test toward agreement.
import hashlib


def _sha(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(buf):
            h.update(chunk)
    return h.hexdigest()


seen, draw_paths = {}, []
for p_ in dict.fromkeys(draw_hits):
    dg = _sha(p_)
    if dg in seen:
        print(f"      [dup of {seen[dg].name}] {p_}")
        continue
    seen[dg] = p_
    draw_paths.append(p_)
if not draw_paths:
    sys.exit(f"STOP: no lstm_draw*.keras found. Expected them in {ARCHIVE_NAME}/.")
print(f"  [found] {len(draw_paths)} distinct LSTM draws "
      f"(from {len(draw_hits)} files)")
for p_ in draw_paths:
    print(f"      {p_}")

OUT_CSV = HERE / "a7_rank2_draw_check.csv"

sys.path.insert(0, str(PIPELINE_PY.parent))
from fyp_pipeline import prepare_building, split_sequences  # noqa: E402

# ============================================== SECTION 3: schema + groups
# Schema comes from the checksum-verified archive pickle, not retyped, so the
# positional indexing below cannot drift from the trained input space.
SCHEMA = list(joblib.load(FEATCOLS_PKL))
print(f"\nAnchor schema ({len(SCHEMA)}): {SCHEMA}")
assert len(SCHEMA) == 15, f"expected 15 features, got {len(SCHEMA)}"

GROUPS = {
    "Hour of Day": ["hour_sin", "hour_cos"],
    "Day of Week": ["dow_sin", "dow_cos"],
    "Month": ["month_sin", "month_cos"],
    "Weekend": ["is_weekend"],
    "Air Temp": ["airTemperature"],
    "Dew Temp": ["dewTemperature"],
    "Wind Direction": ["windDirection"],
    "Wind Speed": ["windSpeed"],
    "Gas": ["gas"],
    "Hot Water": ["hotwater"],
    "Chilled Water": ["chilledwater"],
    "Solar": ["solar"],
}
IDX = {f: i for i, f in enumerate(SCHEMA)}
missing = [f for g in GROUPS.values() for f in g if f not in IDX]
assert not missing, f"schema is missing {missing}"

# ================================== SECTION 4: rebuild the anchor partition
print(f"\nLoading BDGP2 tables from {DATA_DIR} ...")
electricity = pd.read_csv(DATA_DIR / "electricity.csv", parse_dates=["timestamp"])
weather_all = pd.read_csv(DATA_DIR / "weather.csv", parse_dates=["timestamp"])

METER_FILES = {
    "gas": "gas.csv", "hotwater": "hotwater.csv",
    "chilledwater": "chilledwater.csv", "steam": "steam.csv",
    "solar": "solar.csv", "water": "water.csv", "irrigation": "irrigation.csv",
}
meter_dfs = {}
for name, fname in METER_FILES.items():
    p = DATA_DIR / fname
    if p.exists():
        meter_dfs[name] = pd.read_csv(p, parse_dates=["timestamp"])
print(f"  electricity {electricity.shape} | weather {weather_all.shape}")
print(f"  meters: {sorted(meter_dfs.keys())}")

X_seq, y_seq, scaler_y, feature_cols, raw_y_mean = prepare_building(
    ANCHOR, electricity, weather_all, meter_dfs
)

X_tr, y_tr, X_vl, y_vl, X_te, y_te = split_sequences(X_seq, y_seq)
y_te_kwh = scaler_y.inverse_transform(y_te.reshape(-1, 1))

print(f"\nsequences {X_seq.shape} | train {len(X_tr)} | val {len(X_vl)} | test {len(X_te)}")
print(f"raw_y_mean (full year) {raw_y_mean:.2f} kWh")
print(f"test-partition mean    {y_te_kwh.mean():.2f} kWh   (paper Table III: 61.65)")
assert X_seq.shape[0] == EXPECT_SEQ, f"got {X_seq.shape[0]} sequences, expected {EXPECT_SEQ}"
assert len(X_te) == EXPECT_TEST, f"test partition {len(X_te)}, expected {EXPECT_TEST}"

# ------------------------------- GATE 0 / 0b: preprocessing, no model needed
print("\n" + "=" * 70)
print("GATES 0 / 0b — preprocessing reproduces Analysis 1")
print("=" * 70)

same_schema = list(feature_cols) == SCHEMA
print(f"GATE 0b feature order matches archive : {'PASS' if same_schema else 'FAIL'}")
if not same_schema:
    sys.exit(f"STOP: schema mismatch.\n  archive: {SCHEMA}\n  pipeline: {list(feature_cols)}")

sy_arch = joblib.load(SCALER_PKL)
ok = (np.isclose(scaler_y.data_min_[0], sy_arch.data_min_[0])
      and np.isclose(scaler_y.data_max_[0], sy_arch.data_max_[0]))
print(f"GATE 0  scaler_y {scaler_y.data_min_[0]:.1f}/{scaler_y.data_max_[0]:.1f} "
      f"vs archived {sy_arch.data_min_[0]:.1f}/{sy_arch.data_max_[0]:.1f} "
      f": {'PASS' if ok else 'FAIL'}")
if not ok:
    sys.exit("STOP: target scaler differs from the archived one. Preprocessing "
             "is not reproducing Analysis 1.")

# ====================================== SECTION 4b: version-proof weight load
# The Phase 2 draws were saved with Keras 3.15.0 (this env has 3.14-era Keras
# via TF 2.20.0). Keras 3.15 writes 'input_axes'/'output_axes' into the
# GlorotUniform initialiser config, which the older deserialiser rejects, so
# load_model() fails on a purely cosmetic config field. The WEIGHTS themselves
# are plain arrays and are unaffected.
#
# Fix: rebuild the architecture from the archived config and inject the weight
# arrays read straight out of model.weights.h5. Read-only, nothing re-saved,
# so archive integrity against CHECKSUMS.txt is preserved. Same reasoning as
# the E4 load_weights workaround agreed for the TFT.
#
# Draw architecture (from lstm_draw1.keras config.json):
#   Input(48,15) -> LSTM(64, dropout=0.2, return_sequences=True)
#                -> LSTM(32, dropout=0.2) -> Dense(1)
# Note this differs cosmetically from Phase 1, which used separate Dropout
# layers rather than the LSTM's internal dropout argument. Dropout is inactive
# at inference either way, so predictions and SHAP are unaffected.

WEIGHT_ORDER = [
    "layers/lstm/cell/vars/0", "layers/lstm/cell/vars/1", "layers/lstm/cell/vars/2",
    "layers/lstm_1/cell/vars/0", "layers/lstm_1/cell/vars/1", "layers/lstm_1/cell/vars/2",
    "layers/dense/vars/0", "layers/dense/vars/1",
]


def build_draw_arch():
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import LSTM, Dense
    inp = Input(shape=(48, len(SCHEMA)))
    x = LSTM(64, return_sequences=True, dropout=0.2)(inp)
    x = LSTM(32, return_sequences=False, dropout=0.2)(x)
    out = Dense(1)(x)
    return Model(inp, out)


def load_any(path):
    """load_model if possible, else rebuild + inject archived weights."""
    try:
        return tf.keras.models.load_model(path, compile=False), "load_model"
    except Exception as e:
        print(f"  load_model failed ({type(e).__name__}), falling back to "
              f"weight injection")
        with zipfile.ZipFile(path) as z:
            with h5py.File(io.BytesIO(z.read("model.weights.h5")), "r") as f:
                # dense layer name varies across draws (dense, dense_2, ...)
                dense_key = next(k for k in f["layers"] if k.startswith("dense"))
                keys = [k.replace("layers/dense/", f"layers/{dense_key}/")
                        for k in WEIGHT_ORDER]
                arrays = [np.array(f[k]) for k in keys]
        m = build_draw_arch()
        expected = [tuple(w.shape) for w in m.get_weights()]
        got = [a.shape for a in arrays]
        if expected != got:
            raise RuntimeError(
                f"weight shape mismatch.\n  model expects {expected}\n"
                f"  archive has  {got}"
            )
        m.set_weights(arrays)
        return m, "weight injection"


# =========================================== SECTION 5: shared measurement
def test_mae(model):
    pred = scaler_y.inverse_transform(model.predict(X_te, verbose=0))
    return mean_absolute_error(y_te_kwh, pred)


def shap_groups(model, seed=SEED):
    """Mean |phi| per 12-group feature over the anchor test partition."""
    np.random.seed(seed)
    expl = shap.GradientExplainer(model, X_tr[:N_SHAP])
    sv = expl.shap_values(X_te[:N_SHAP])
    sv = sv[0] if isinstance(sv, list) else sv
    sv = np.squeeze(sv)                       # -> (200, 48, 15)
    per_feat = np.abs(sv).mean(axis=(0, 1))
    return {g: float(sum(per_feat[IDX[f]] for f in fs)) for g, fs in GROUPS.items()}


def report(tag, model):
    mae = test_mae(model)
    g = shap_groups(model)
    order = sorted(g, key=g.get, reverse=True)
    print(f"\n--- {tag} ---")
    print(f"  test MAE {mae:.3f} kWh  ({100 * mae / y_te_kwh.mean():.2f}% of test mean)")
    for i, k in enumerate(order[:5], 1):
        print(f"  rank {i}: {k:<15} {g[k]:.6f}")
    print(f"  Weekend {g['Weekend']:.6f}  vs  Day of Week {g['Day of Week']:.6f}"
          f"   ->  rank2 = {order[1]}")
    return {
        "model": tag,
        "test_mae_kwh": round(mae, 4),
        "rank1": order[0], "rank2": order[1], "rank3": order[2],
        "shap_hour_of_day": g["Hour of Day"],
        "shap_weekend": g["Weekend"],
        "shap_day_of_week": g["Day of Week"],
        "weekend_over_dow": g["Weekend"] > g["Day of Week"],
    }

# ============================================= SECTION 6: gates 1/2 (Phase 1)
rows = []
row_p1 = None

if P1_MODEL is None:
    print("\n" + "=" * 70)
    print("DRAWS-ONLY MODE — Phase 1 anchor model not on this machine")
    print("=" * 70)
    print("Gates 1 and 2 cannot run, so no Phase 1 comparison is possible.")
    print("The run can still show whether the three draws AGREE WITH EACH")
    print("OTHER, which is the stronger form of A7 and needs no Phase 1 row.")
else:
    print("\n" + "=" * 70)
    print("GATES 1 / 2 — the Phase 1 model reproduces its archived result")
    print("=" * 70)
    m1 = tf.keras.models.load_model(P1_MODEL, compile=False)
    row_p1 = report("PHASE1 multisite_anchor_best", m1)
    g1 = abs(row_p1["test_mae_kwh"] - EXPECT_MAE) < 0.01
    g2 = bool(row_p1["weekend_over_dow"])
    print(f"\nGATE 1  test MAE == {EXPECT_MAE} : {'PASS' if g1 else 'FAIL'}")
    print(f"GATE 2  Weekend > Day of Week : {'PASS' if g2 else 'FAIL'}")
    if not (g1 and g2):
        sys.exit(
            "\nSTOP. A gate failed. The reconstruction does not match the "
            "archived Phase 1 result, so any comparison would measure "
            "reconstruction error rather than the weight draw.\n"
            "If GATE 1 failed only narrowly, retry with "
            "TF_ENABLE_ONEDNN_OPTS=0 before concluding anything."
        )
    print("Gates passed. Preprocessing, model and SHAP protocol validated.")
    rows.append(row_p1)

# ============================================ SECTION 7: Phase 2 LSTM draws
print("\n" + "=" * 70)
print("PHASE 2 DRAWS — identical protocol, only the weights differ")
print("=" * 70)
# TFT draws are excluded on purpose: custom layers lack get_config, so
# load_model fails in a fresh session (E4). Architecture invariance is a
# separate question from draw stability.
# lstm_refit_bobcat.keras is byte-identical to lstm_draw1.keras and is
# therefore not globbed, to avoid double-counting draw 1.
print(f"Using: {[p.name for p in draw_paths]}")

for path in draw_paths:
    print(f"\nloading {path.name} ...")
    try:
        m, how = load_any(path)
    except Exception as e:
        print(f"[SKIP] {path.name}: {type(e).__name__}: {e}")
        continue
    print(f"  loaded via {how}")
    if tuple(m.input_shape[1:]) != (48, len(SCHEMA)):
        print(f"[SKIP] {path.name}: input shape {m.input_shape[1:]} is not "
              f"(48, {len(SCHEMA)}), not comparable.")
        continue
    r = report(f"PHASE2 {path.stem}", m)
    # GATE 3 — draw 1 is lstm_refit_bobcat, recorded at 14.56% anchor test MAE%.
    # If injection had mis-ordered the weights this would be far off.
    if path.stem == "lstm_draw1":
        pct = 100 * r["test_mae_kwh"] / y_te_kwh.mean()
        ok3 = abs(pct - 14.56) < 0.5
        print(f"  GATE 3  draw1 MAE% {pct:.2f} vs recorded 14.56 : "
              f"{'PASS' if ok3 else 'FAIL — weights may be mis-ordered'}")
    rows.append(r)

# ==================================================== SECTION 8: verdict
df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(df[["model", "test_mae_kwh", "rank1", "rank2", "rank3"]].to_string(index=False))

p2 = df[df.model.str.startswith("PHASE2")]
print("\nINTERPRETATION")
if len(p2) == 0:
    print("  No Phase 2 draw ran. A7 stays UNRESOLVED.")
elif p2["rank2"].nunique() > 1:
    print(f"  Phase 2 rank2 across {len(p2)} draws: {p2['rank2'].value_counts().to_dict()}")
    print("  -> Draws DISAGREE among themselves. Rank-2 identity is not "
          "draw-stable. This is the stronger form of A7 and stands on its own, "
          "with or without the Phase 1 row.")
elif row_p1 is None:
    print(f"  Phase 2 rank2 across {len(p2)} draws: {p2['rank2'].iloc[0]} (unanimous)")
    print("  -> Draws agree with each other. Whether this differs from Phase 1 "
          "is UNANSWERED without multisite_anchor_best.keras. Rerun on the "
          "machine that has it before writing anything into A7.")
elif p2["rank2"].iloc[0] != row_p1["rank2"]:
    print(f"  Phase 1 rank2: {row_p1['rank2']} | Phase 2 rank2: {p2['rank2'].iloc[0]} (unanimous)")
    print("  -> Draws agree with each other but DIFFER from Phase 1. Partition "
          "and building are held fixed, so the weight draw is isolated as the "
          "cause. A7 CONFIRMED.")
else:
    print(f"  Phase 1 and all {len(p2)} draws agree: rank2 = {row_p1['rank2']}")
    print("  -> The 98.1% Day-of-Week result then comes from the partition or "
          "the building population, NOT the draw. A7 is disconfirmed as "
          "written. Rewrite the note, do not delete it: a disconfirmed "
          "pre-registered hypothesis is still a result.")

print(f"\nWrote {OUT_CSV}")
print("No archived artefact was modified.")
