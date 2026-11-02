#!/usr/bin/env python3
"""
a9_rank2_vs_rank5_tenseed.py
============================
ANALYSIS 9 -- ARE RANK 2 AND RANK 5 DISTINGUISHABLE?

WHY THIS EXISTS
---------------
The Temporal Fusion Transformer selection rule in FYP_Phase2_TFT_FINAL.py
is explicit:

    winner = min(confirmation_results, key=lambda r: r['confirmed_mean'])

It was executed. Its recorded output, preserved in
archive/2026-07-21_analysis1/FYP_Phase2_TFT_analysis1.ipynb, names
screening rank 5 -- batch size 256, confirmed mean 18.11 per cent -- as
the winner. The next cell hardcodes screening rank 2 -- batch size 64,
confirmed mean 18.82 per cent -- and every Phase 2 Temporal Fusion
Transformer result is computed on that model. The two configurations are
identical in every respect except batch size.

This script does NOT re-select the model. Re-selecting after inspecting
the outcomes would convert a disclosed clerical divergence into a
post-hoc choice, and the 2026-09-02 rerun already showed the ordering of
these configurations does not survive repetition: no configuration held
its position, and the rank correlation with the July ordering was -0.6.
There is no "correct" configuration for a fresh run to find.

What this script does is answer the question the divergence actually
raises: ARE THE TWO DISTINGUISHABLE AT ALL, given enough seeds? If they
are not, the divergence is a defect in the record with no consequence
for any result, and can be reported as such with a measurement behind
it. If they are, the finding is that the n=3 selection protocol was
underpowered -- which is the contribution already claimed in the
Introduction -- and the model still does not change, because the
Temporal Fusion Transformer is a comparator in this work rather than its
object of study. Section sec:optparity already states that the
comparison is between each architecture as configured.

Either outcome is reportable. Do not tune anything to obtain one.

WHAT THE ORIGINAL PROTOCOL ACTUALLY VARIED -- READ THIS
-------------------------------------------------------
The 2026-07-21 confirmation and the 2026-09-02 reruns both call
tf.random.set_seed(SEED) with SEED = 2016 before EVERY run. The three
runs per configuration were therefore SAME-SEED retrainings, and their
reported standard deviation is retraining variance in the sense of
Analysis 6, not initialisation variance. This is not what "n=3 seeds" is
usually taken to mean, and it is worth stating plainly wherever those
figures are used.

This script defaults to TEN DISTINCT SEEDS (2016-2025), matching
Analysis 5, because the practitioner-relevant question is whether the
two configurations separate under the full variance a retrained model
faces. Pass --same-seed to reproduce the original protocol instead. The
seed is recorded per run either way, so both readings are available
from the output.

PROTOCOL
--------
Copied from rerun_tft_confirmation.py, which is itself copied from
FYP_Phase2_TFT_FINAL.py. Do not change these.

    anchor        Bobcat_education_Alissa, 2016, 48-hour window
    split         6,988 / 874 / 874 chronological
    optimiser     Adam lr=0.001, mse loss
    epochs        300, EarlyStopping(patience=15, restore_best_weights)
    metric        100 * MAE(kWh) / raw_y_mean -- the anchor's FULL-YEAR
                  mean, matching the grid and the confirmation log. The
                  test-partition denominator is recorded alongside it as
                  a second column, since it is free and main.tex quotes
                  both conventions in different places.
    rank 2        d_model 64, heads 4, dropout 0.2, units (64,32), batch 64
    rank 5        d_model 64, heads 4, dropout 0.2, units (64,32), batch 256

The model definition below is spliced verbatim out of
rerun_tft_confirmation.py and its SHA-256 is checked against that file at
startup, so it cannot drift from the model of record.

WHY ANALYSIS 5'S TFT SEEDS CANNOT BE REUSED HERE
------------------------------------------------
Analysis5_TenSeed_Confirmation.ipynb already ran ten seeds on exactly
the rank-2 configuration, so it looks like half this job is done. It is
not. Analysis 5 computes mae_pct as 100 * MAE / y_test_mean -- the
TEST-PARTITION mean -- while the grid, the confirmation log and this
script use raw_y_mean, the anchor's FULL-YEAR mean. The two are
different quantities and must not be pooled. That is why Analysis 5's
TFT figure is 18.42 and the confirmation log's rank-2 figure is 18.82
for what is otherwise the same model and protocol; the gap is the
denominator, not the model. Both configurations are therefore run here
from scratch, and every row records both denominators so the numbers can
be read either way without a second run.

DATA FILE CONVENTION
--------------------
Reads electricity.csv, weather.csv and the plain meter CSVs, matching
FYP_Phase2_TFT_FINAL.py and rerun_tft_confirmation.py. Not the
*_cleaned.csv variants, which a6_sameseed_rung.py uses.

STATISTICS
----------
Reported on the ten per-seed MAE values for each configuration:

    Welch's t          difference in mean accuracy
    Mann-Whitney U     the same without a normality assumption
    Brown-Forsythe     difference in variance (Levene, centre=median),
                       the same test Analysis 5 uses
    Cohen's d          effect size, so "no difference" is bounded rather
                       than merely unrejected
    paired by seed     both configurations see seeds 2016-2025, so the
                       per-seed differences are also reported; a paired
                       reading removes seed-to-seed common variance

A non-significant result at n=10 is not proof of equivalence. The script
therefore also reports the smallest difference in means the design could
have detected at 80 per cent power, so the negative can be stated with a
bound rather than as an absence.

OUTPUTS
-------
    results/analysis9_rank2_vs_rank5.csv          one row per run
    results/analysis9_rank2_vs_rank5_summary.csv  the statistics
    analysis9_models/                             20 .keras files
    stdout                                        the same, human readable

Resumable: re-running skips (config, seed) pairs already in the CSV.

Archive integrity: reads BDGP2 source data only. Trains new models into
its own output directory. No existing archive, result file or model is
read for training, modified, or overwritten.

USAGE
-----
    python a9_rank2_vs_rank5_tenseed.py --pipeline-dir . \
        --data-dir "/mnt/e/Claude/FYP/ZEIT4500 - FYP/Data"

    --seeds 10        seeds per configuration (default 10)
    --same-seed       reproduce the original protocol: every run at 2016
    --dry-run         one run per configuration, to check it all loads

Roughly 9-10 minutes per run on an RTX 4070 Super, from the Analysis 5
timings, so about 3 hours for the default 20 runs.
"""

import argparse
import datetime
import gc
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error

SEED = 2016
os.environ.setdefault("PYTHONHASHSEED", str(SEED))

import tensorflow as tf                                     # noqa: E402
for _g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_g, True)

from tensorflow.keras import layers                         # noqa: E402
from tensorflow.keras.callbacks import EarlyStopping        # noqa: E402
from tensorflow.keras.optimizers import Adam                # noqa: E402

ANCHOR      = "Bobcat_education_Alissa"
MAX_EPOCHS  = 300
PATIENCE    = 15
ARCH_SHA256 = "c6c6055d8ee4bd6e7bcedb28bb6110832fcb95c78364f3e834f5d48d43753472"

METER_NAMES = ["chilledwater", "gas", "hotwater", "solar",
               "steam", "water", "irrigation"]

DATA_DIR_CANDIDATES = [
    os.environ.get("FYP_DATA_DIR", ""),
    "../Data", "./Data", "../../Data",
    "/mnt/e/Claude/FYP/ZEIT4500 - FYP/Data",
    "/mnt/c/Users/Paul/OneDrive/Desktop/Dump/Uni/Y4/S1/ZEIT4500 - FYP/Data",
    "/mnt/c/Users/pauli/OneDrive/Desktop/Dump/Uni/Y4/S1/ZEIT4500 - FYP/Data",
]

# The two configurations, verbatim from TOP5_CONFIGS in
# rerun_tft_confirmation.py. They differ in batch_size and nothing else.
CONFIGS = {
    2: dict(d_model=64, n_heads=4, dropout=0.2, lstm_units=(64, 32),
            batch_size=64,  screening_val_rmse=11.883155451762402,
            july_confirmed_mean=18.821535117854896,
            july_confirmed_std=0.10840370765801832,
            note="used throughout Phase 2"),
    5: dict(d_model=64, n_heads=4, dropout=0.2, lstm_units=(64, 32),
            batch_size=256, screening_val_rmse=12.090910947023144,
            july_confirmed_mean=18.11222288917676,
            july_confirmed_std=0.924300994658641,
            note="returned by the selection rule"),
}


def resolve_data_dir(given):
    tried = []
    for cand in ([given] if given else DATA_DIR_CANDIDATES):
        if not cand:
            continue
        tried.append(cand)
        d = Path(cand)
        if (d / "electricity.csv").is_file() and (d / "weather.csv").is_file():
            return d
    raise SystemExit(
        "Could not locate the BDGP2 data directory.\n"
        "Needed: electricity.csv and weather.csv in one directory.\n"
        "Tried:\n  " + "\n  ".join(tried) + "\n\n"
        'Pass it explicitly, e.g. --data-dir "/mnt/e/Claude/FYP/'
        'ZEIT4500 - FYP/Data", or set FYP_DATA_DIR.')


def check_architecture_matches_source(path):
    """The model below is a verbatim splice. Refuse to run if the source
    it was spliced from has changed, rather than silently comparing a
    model that is no longer the model of record."""
    p = Path(path)
    if not p.is_file():
        print(f"  !! {path} not found -- cannot verify the architecture "
              f"splice. Continuing, but the model of record is unconfirmed.")
        return
    lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
    block = "".join(lines[160:268])
    got = hashlib.sha256(block.encode()).hexdigest()
    if got == ARCH_SHA256:
        print(f"  architecture splice verified against {path}")
    else:
        raise SystemExit(
            f"Architecture splice does NOT match {path}.\n"
            f"  expected {ARCH_SHA256}\n  got      {got}\n"
            "The model definition in this script was copied from that file. "
            "If that file has legitimately changed, re-splice; do not run a "
            "comparison against a model that is not the model of record.")


# =====================================================================
# Model definition -- spliced verbatim from rerun_tft_confirmation.py,
# which is itself verbatim from FYP_Phase2_TFT_FINAL.py. Do not modify.
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
# Seeding call. Analysis5_TenSeed_Confirmation.ipynb uses
# tf.keras.utils.set_random_seed(), which additionally seeds Python's own
# generator; this script sets the TensorFlow and NumPy generators directly.
# A seed value therefore names a DIFFERENT model under the two scripts, and
# per-seed results are not comparable across them (distributions are).
# Recorded per row as `seed_api` so any future comparison can check first.
# Do not change the default: the 2026-09-03 run was made under "tf+np" and
# changing it would break reproduction of that run from this file.
SEED_API = "tf+np"


def apply_seed(seed):
    if SEED_API == "keras":
        tf.keras.utils.set_random_seed(seed)
    else:
        tf.random.set_seed(seed)
        np.random.seed(seed)


def train_one(cfg, seed, X_tr, y_tr, X_vl, y_vl, X_te, y_te,
              scaler_y, raw_y_mean, window, n_features, model_dir, rank):
    apply_seed(seed)
    model, _ = build_tft(seq_len=window, n_features=n_features,
                         d_model=cfg["d_model"], n_heads=cfg["n_heads"],
                         dropout=cfg["dropout"], lstm_units=cfg["lstm_units"])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    es = EarlyStopping(patience=PATIENCE, restore_best_weights=True,
                       monitor="val_loss")
    t0 = datetime.datetime.now(datetime.timezone.utc)
    hist = model.fit(X_tr, y_tr, validation_data=(X_vl, y_vl),
                     epochs=MAX_EPOCHS, batch_size=cfg["batch_size"],
                     callbacks=[es], verbose=0)
    elapsed = (datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds()

    pred = scaler_y.inverse_transform(model.predict(X_te, verbose=0)).flatten()
    true = scaler_y.inverse_transform(y_te.reshape(-1, 1)).flatten()
    mae = mean_absolute_error(true, pred)

    wpath = model_dir / f"tft_rank{rank}_seed{seed}.keras"
    model.save(wpath)
    h = hashlib.sha256()
    with open(wpath, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)

    row = dict(
        rank=rank, seed=seed, batch_size=cfg["batch_size"], seed_api=SEED_API,
        mae_pct=100 * mae / raw_y_mean,            # full-year denominator
        mae_pct_testpartition=100 * mae / true.mean(),
        mae_kwh=mae,
        n_epochs_trained=len(hist.history["loss"]),
        elapsed_s=round(elapsed, 1),
        weights_path=str(wpath), weights_sha256=h.hexdigest(),
        max_epochs=MAX_EPOCHS, patience=PATIENCE,
        run_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status="ok",
    )
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    return row


def summarise(df, out_path):
    a = df[df["rank"] == 2].sort_values("seed")
    b = df[df["rank"] == 5].sort_values("seed")
    A, B = a.mae_pct.values, b.mae_pct.values
    n = min(len(A), len(B))

    print("\n" + "=" * 68)
    print("RANK 2 (batch 64, used) vs RANK 5 (batch 256, rule's winner)")
    print("=" * 68)
    for lbl, v in (("rank 2", A), ("rank 5", B)):
        if len(v):
            print(f"  {lbl}: n={len(v)}  mean {v.mean():.3f}  "
                  f"sd {v.std(ddof=1):.3f}pp  "
                  f"min {v.min():.3f}  max {v.max():.3f}")

    if n < 3:
        print("  fewer than 3 runs per configuration -- no statistics")
        return

    t, pt = stats.ttest_ind(A, B, equal_var=False)
    u, pu = stats.mannwhitneyu(A, B, alternative="two-sided")
    lev, pl = stats.levene(A, B, center="median")
    sp = np.sqrt(((len(A) - 1) * A.var(ddof=1) + (len(B) - 1) * B.var(ddof=1))
                 / (len(A) + len(B) - 2))
    d = (A.mean() - B.mean()) / sp if sp > 0 else np.nan

    m = a.merge(b, on="seed", suffixes=("_r2", "_r5"))
    diff = (m.mae_pct_r2 - m.mae_pct_r5).values
    if len(m) >= 3:
        tp, ppp = stats.ttest_rel(m.mae_pct_r2, m.mae_pct_r5)
    else:
        tp, ppp = np.nan, np.nan

    # smallest detectable difference, two-sided alpha .05, power .80
    mdd = 2.8 * sp * np.sqrt(2.0 / n)

    print(f"\n  difference in means : {A.mean() - B.mean():+.3f} pp "
          f"(rank 2 minus rank 5)")
    print(f"  Welch's t           : t={t:.3f}  p={pt:.4f}")
    print(f"  Mann-Whitney U      : U={u:.1f}  p={pu:.4f}")
    print(f"  Brown-Forsythe      : W={lev:.3f}  p={pl:.4f}  (variance)")
    print(f"  Cohen's d           : {d:+.3f}")
    if len(m) >= 3:
        print(f"  paired by seed (n={len(m)}) : mean diff {diff.mean():+.3f}pp "
              f"sd {diff.std(ddof=1):.3f}  t={tp:.3f}  p={ppp:.4f}")
    print(f"  smallest difference detectable at 80% power: {mdd:.3f} pp")

    print("\n  On record, same two configs, 2026-07-21, n=3 at one fixed seed:")
    print("    rank 2  18.822 +/- 0.108      rank 5  18.112 +/- 0.924")

    print(f"\n  VERDICT at n={n}: the two configurations are "
          f"{'NOT distinguishable' if pt >= 0.05 else 'distinguishable'} "
          f"on mean MAE (Welch p={pt:.4f}).")
    if pt >= 0.05:
        print(f"  State the negative with its bound: a true difference larger "
              f"than {mdd:.2f}pp would probably have been detected.")
        print("  The selection divergence recorded in sec:corrections "
              "therefore has no measurable consequence for any result.")
    else:
        print("  The n=3 selection protocol was underpowered. This does NOT "
              "license swapping the model now -- that would be selection on "
              "an inspected result. Report it as a limitation.")

    rows = [dict(statistic=k, value=v) for k, v in [
        ("n_per_config", n), ("mean_rank2", A.mean()), ("mean_rank5", B.mean()),
        ("sd_rank2", A.std(ddof=1)), ("sd_rank5", B.std(ddof=1)),
        ("diff_means_pp", A.mean() - B.mean()),
        ("welch_t", t), ("welch_p", pt),
        ("mannwhitney_u", u), ("mannwhitney_p", pu),
        ("brownforsythe_w", lev), ("brownforsythe_p", pl),
        ("cohens_d", d),
        ("paired_mean_diff_pp", diff.mean() if len(m) else np.nan),
        ("paired_t", tp), ("paired_p", ppp),
        ("min_detectable_diff_pp_80pct", mdd),
    ]]
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


def main(args):
    sys.path.insert(0, args.pipeline_dir)
    from fyp_pipeline import prepare_building, split_sequences

    check_architecture_matches_source(
        Path(args.pipeline_dir) / "rerun_tft_confirmation.py")

    data_dir = resolve_data_dir(args.data_dir)
    print(f"  data dir: {data_dir}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("\nPreparing anchor ...")
    electricity = pd.read_csv(data_dir / "electricity.csv",
                              parse_dates=["timestamp"])
    weather = pd.read_csv(data_dir / "weather.csv", parse_dates=["timestamp"])
    meter_dfs = {}
    for name in METER_NAMES:
        p = data_dir / f"{name}.csv"
        if p.is_file():
            meter_dfs[name] = pd.read_csv(p, parse_dates=["timestamp"])
    print(f"  meters loaded: {sorted(meter_dfs)}")

    X_seq, y_seq, scaler_y, schema, raw_y_mean = prepare_building(
        ANCHOR, electricity, weather, meter_dfs)
    X_tr, y_tr, X_vl, y_vl, X_te, y_te = split_sequences(X_seq, y_seq)
    assert len(schema) == 15, f"expected 15 features, got {len(schema)}"
    window, n_features = X_tr.shape[1], len(schema)
    print(f"  schema ({n_features}): {schema}")
    print(f"  sequences {len(X_seq)}  train {len(X_tr)}  val {len(X_vl)}  "
          f"test {len(X_te)}")
    print(f"  raw_y_mean (full-year): {raw_y_mean:.4f} kWh")

    seeds = ([SEED] * args.seeds if args.same_seed
             else list(range(SEED, SEED + args.seeds)))
    if args.dry_run:
        seeds = seeds[:1]
    print(f"\nseeds      : {seeds}")
    print(f"protocol   : "
          f"{'SAME seed every run (original)' if args.same_seed else 'distinct seeds (Analysis 5 convention)'}")
    print(f"seed call  : {SEED_API}")
    print(f"runs       : {len(seeds)} per configuration, {2 * len(seeds)} total\n")

    prov = out.parent / (out.stem + "_PROVENANCE.txt")
    with open(prov, "w", encoding="utf-8") as fh:
        fh.write("ANALYSIS 9 -- rank 2 vs rank 5, ten seeds each\n")
        fh.write(f"run (UTC)   : {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n")
        fh.write(f"script      : {Path(__file__).name}\n")
        fh.write(f"arch splice : sha256 {ARCH_SHA256}\n")
        fh.write(f"data dir    : {data_dir}   (raw CSVs)\n")
        fh.write(f"seed call   : {SEED_API}   "
                 f"(Analysis 5 uses keras; per-seed results are not "
                 f"comparable across the two)\n")
        fh.write(f"seeds       : {seeds}\n")
        fh.write(f"protocol    : epochs {MAX_EPOCHS}, patience {PATIENCE}, "
                 f"Adam lr=0.001, mse\n")
        fh.write(f"metric      : mae_pct = 100*MAE/raw_y_mean (full-year); "
                 f"mae_pct_testpartition recorded alongside\n")
    print(f"provenance -> {prov}\n")

    done = set()
    if out.exists():
        prev = pd.read_csv(out)
        done = set(zip(prev["rank"], prev["seed"]))
        print(f"resuming, {len(done)} (config, seed) pairs already done")

    for rank, cfg in CONFIGS.items():
        for i, seed in enumerate(seeds):
            if (rank, seed) in done and not args.same_seed:
                continue
            print(f"-- rank {rank} (batch {cfg['batch_size']}, {cfg['note']}) "
                  f"seed {seed}  [{i + 1}/{len(seeds)}]", flush=True)
            try:
                row = train_one(cfg, seed, X_tr, y_tr, X_vl, y_vl, X_te, y_te,
                                scaler_y, raw_y_mean, window, n_features,
                                model_dir, rank)
                print(f"   MAE% {row['mae_pct']:.3f}  "
                      f"({row['n_epochs_trained']} epochs, "
                      f"{row['elapsed_s']:.0f}s)")
            except Exception as e:
                row = dict(rank=rank, seed=seed,
                           status=f"{type(e).__name__}: {e}")
                print(f"   FAILED: {row['status']}")
            df = pd.DataFrame([row])
            if out.exists():
                df = pd.concat([pd.read_csv(out), df], ignore_index=True)
            df.to_csv(out, index=False)

    full = pd.read_csv(out)
    ok = full[full.status == "ok"]
    print(f"\nwrote {out}  ({len(ok)}/{len(full)} runs ok, "
          f"{ok.weights_sha256.nunique()} unique model checksums)")
    summarise(ok, Path(str(out).replace(".csv", "_summary.csv")))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline-dir", default=".",
                    help="directory containing fyp_pipeline.py and "
                         "rerun_tft_confirmation.py")
    ap.add_argument("--data-dir", default=None,
                    help="BDGP2 CSV directory; auto-discovered if omitted")
    ap.add_argument("--seeds", type=int, default=10,
                    help="seeds per configuration (default 10)")
    ap.add_argument("--same-seed", action="store_true",
                    help="reproduce the original protocol: every run at "
                         "seed 2016, so only retraining varies")
    ap.add_argument("--model-dir", default="analysis9_models")
    ap.add_argument("--out", default="results/analysis9_rank2_vs_rank5.csv")
    ap.add_argument("--dry-run", action="store_true")
    sys.exit(main(ap.parse_args()))
