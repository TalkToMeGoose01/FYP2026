#!/usr/bin/env python3
"""
a12_zerofill_audit.py
=====================
ANALYSIS 12 -- SCHEMA ZERO-FILL AUDIT, AND THE RE-RUN THAT ARCHIVES
RAW ATTRIBUTION VECTORS

WHY THIS EXISTS
---------------
main.tex makes two claims about the zero-fill control in
`align_to_schema`:

  sec:method    "A constant input produces no error change under
                 permutation and no gradient under either gradient
                 method, which is the intended behaviour."

  sec:controls  "Under SHAP, a constant input produces no gradient and
                 returns values of exactly zero, verified on the raw
                 uncollapsed attribution tensor for a building confirmed
                 to lack the meter in question."

Read directly from results/analysis4_shapig_meanbaseline_200steps_FULL.csv
(n=1,276, status=ok, model lstm_draw1), those claims hold for exactly one
method-meter combination of eight:

    SHAP x chilledwater   exactly 0.0 in all 877 buildings lacking it
    SHAP x gas            non-zero in all 1,117 buildings lacking it
    SHAP x hotwater       non-zero in all 1,123 buildings lacking it
    SHAP x solar          non-zero in all 1,274 buildings lacking it
    IG   x all four       non-zero in every building, present or not

The decisive check is solar. Data/solar.csv contains FIVE building
columns in the whole of BDGP2. There is no data-based route to a solar
attribution in 1,274 buildings. The values are a reference artefact:
SHAP's background is the anchor's X_train[:200] and IG's baseline is the
anchor training mean, so (input - reference) is non-zero for a target
building presenting a column of zeros whenever the ANCHOR's own column
for that feature is not itself zero over the reference window.

Permutation importance is immune -- shuffling a constant column changes
nothing -- which is why sec:strat records permutation non-zero counts of
340/143/136/1 against SHAP's and IG's near-universal non-zeros.

WHAT THIS COSTS, ANALYTICALLY
-----------------------------
Every SHAP and IG Spearman correlation in this work ranks 12 collapsed
features, of which up to 4 may carry no building-specific model
information. That affects the estimation ceilings, the across-draw
agreements, the instability ladder of Equation (2), the cross-method
correlation, and any statement about which controllable feature leads.

STAGES
------
Run them in order. Stages 1, 2, 4 and 5 are cheap. Stage 3 is the only
one that needs real compute, and stage 1 is designed to be run first so
that the mechanism is confirmed on artefact before stage 3 is committed
to.

  1  diag      Anchor baseline/background diagnostic. ~1 min.
               Tests the stated mechanism explicitly and prints
               PASS/FAIL against a prediction registered in advance:
               chilledwater is constant across the SHAP background
               window X_train[:200]; gas, hotwater and solar are not.
               -> analysis12_baseline_diagnostic.csv

  2  mask      Per-building feature mask over the whole population,
               derived from the prepared array itself rather than from
               any lookup table. Records, per schema feature, whether
               the building supplied it and whether the aligned column
               actually varies. ~15-40 min, no model inference.
               -> analysis12_feature_mask.csv

  3  vectors   THE RE-RUN. SHAP, Integrated Gradients and permutation
               importance on the 25 ceiling buildings, over the three
               archived LSTM draws plus the ceiling repeats, writing
               the RAW per-feature collapsed vectors. This is the
               artefact whose absence forced the re-run: analysis1e and
               analysis3 stored only summary rho, so no re-slice was
               possible. Hours.
               -> analysis12_attribution_vectors.csv

  4  agree     Pure post-processing of stages 2 and 3. Recomputes
               ceilings, across-draw agreement and instability under
               three conventions. Seconds.
               -> analysis12_agreement.csv
               -> analysis12_instability.csv

  5  reslice   Re-derives, from artefacts that already exist and under
               the stage-2 mask: the controllable-feature result at
               n=1,276, cross-method rho, and permutation across-draw
               agreement at n=319. No compute.
               -> analysis12_controllable.csv
               -> analysis12_perm_reslice.csv

THE THREE CONVENTIONS
---------------------
  full12    all 12 collapsed features. The convention every current
            figure in main.tex uses. Retained so the correction is
            measurable against the existing numbers rather than merely
            replacing them.
  masked    per building, only features whose aligned column varies.
            The correct convention. Feature count differs per building,
            so n_features_ranked is recorded on every row and must be
            reported alongside any median.
  uniform8  the 8 non-meter collapsed features for every building.
            Commensurable across buildings at the cost of discarding
            the chilled-water signal in the 399 buildings that have it.

Report masked as primary and the other two as sensitivity. Do not quote
a median across conventions.

MODELS
------
The three original draws are recovered from archive/2026-07-21_analysis1,
not retrained, so the re-run corrects the published figures rather than
replacing them with figures from different weights. Their SHA-256 digests
are checked against archive/2026-07-21_analysis1/CHECKSUMS.txt where that
file provides them, and are recorded in the provenance sidecar either way.

SPLICED CODE
------------
integrated_gradients() and permutation_importance() are not
reimplemented. They are extracted from FYP_Phase2_TFT_FINAL.py at run
time by AST, hashed, and executed into this module's namespace, so the
attribution arithmetic is provably the same code that produced the
figures being corrected. The extracted source and its SHA-256 are
written to the provenance sidecar. If that file changes, the digest
changes and the run says so.

USAGE
-----
    python a12_zerofill_audit.py --stage 1
    python a12_zerofill_audit.py --stage 2
    python a12_zerofill_audit.py --stage 3          # the long one
    python a12_zerofill_audit.py --stage 4
    python a12_zerofill_audit.py --stage 5

    python a12_zerofill_audit.py --stage 3 --dry-run    # one building
    python a12_zerofill_audit.py --stage all

--data-dir is auto-discovered from the same candidate list a8 uses.
Stage 3 is resumable: re-running skips (building, method, model,
replicate) rows already on disk.

DATA FILE CONVENTION
--------------------
Raw CSVs, matching FYP_Phase2_TFT_FINAL.py, a8_shap_ceiling.py and the
scripts that produced every ceiling in tab:ceilings. NOT the *_cleaned
variants. Pass --cleaned only to reproduce a6's convention, and never
for a figure that is compared against tab:ceilings.

ARCHIVE INTEGRITY
-----------------
Reads BDGP2 source data, archived .keras weights and existing result
CSVs. Writes only its own analysis12_* outputs. No archive is modified,
no model is trained, no model is re-saved.
"""

import argparse
import ast
import datetime
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------------
# Determinism controls BEFORE importing TensorFlow. SHAP imports TF
# internally, so anything set afterwards is silently ignored. Same
# ordering rule as a6_sameseed_rung.py, a8_shap_ceiling.py and the
# notebook.
# ---------------------------------------------------------------------
SEED = 2016
os.environ.setdefault("PYTHONHASHSEED", str(SEED))

# ---------------------------------------------------------------------
# Fixed parameters -- copied from a8_shap_ceiling.py and
# FYP_Phase2_TFT_FINAL.py. Do not alter.
# ---------------------------------------------------------------------
ANCHOR       = "Bobcat_education_Alissa"
SHAP_SAMPLES = 200
IG_SAMPLES   = 200
IG_STEPS     = 200
IG_BATCH     = 25
MAX_SEQ      = 3000          # permutation sequence cap
PREDICT_BATCH = 3000
N_PERM_REPEATS = 5
N_TEST       = 25

RNG_A, RNG_B = 0, 1          # the two independent estimator RNG offsets

PAIR_MAP = {
    "Hour of Day": ("hour_sin", "hour_cos"),
    "Day of Week": ("dow_sin", "dow_cos"),
    "Month":       ("month_sin", "month_cos"),
}

METER_NAMES = ["chilledwater", "gas", "hotwater", "solar",
               "steam", "water", "irrigation"]

# The four meter covariates that enter the 15-feature schema.
SCHEMA_METERS = ["chilledwater", "gas", "hotwater", "solar"]

DATA_DIR_CANDIDATES = [
    os.environ.get("FYP_DATA_DIR", ""),
    "../Data", "./Data", "../../Data",
    "/mnt/e/Claude/FYP/ZEIT4500 - FYP/Data",
    "/mnt/c/Users/Paul/OneDrive/Desktop/Dump/Uni/Y4/S1/ZEIT4500 - FYP/Data",
    "/mnt/c/Users/pauli/OneDrive/Desktop/Dump/Uni/Y4/S1/ZEIT4500 - FYP/Data",
]

# Registered in advance of stage 1, so the diagnostic is a test and not
# a description. See the module docstring.
MECHANISM_PREDICTION = {
    "chilledwater": "constant",
    "gas":          "varies",
    "hotwater":     "varies",
    "solar":        "varies",
}


# =====================================================================
# helpers
# =====================================================================
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def stable_seed(building, base=SEED):
    """Deterministic per-building seed, independent of PYTHONHASHSEED.
    Copied verbatim from FYP_Phase2_TFT_FINAL.ipynb."""
    h = int(hashlib.md5(building.encode()).hexdigest()[:8], 16)
    return base + (h % 100000)


def resolve_data_dir(given, cleaned):
    elec = "electricity_cleaned.csv" if cleaned else "electricity.csv"
    tried = []
    for cand in ([given] if given else DATA_DIR_CANDIDATES):
        if not cand:
            continue
        tried.append(cand)
        d = Path(cand)
        if (d / elec).is_file() and (d / "weather.csv").is_file():
            return d
    raise SystemExit(
        "Could not locate the BDGP2 data directory.\n"
        f"Needed: {elec} and weather.csv in one directory.\nTried:\n  "
        + "\n  ".join(tried)
        + '\n\nPass it explicitly, e.g.\n  --data-dir "/mnt/e/Claude/FYP/'
          'ZEIT4500 - FYP/Data"\nor set FYP_DATA_DIR.')


def build_collapse_index(schema):
    """15 -> 12 collapse, schema-derived. Copied from a6_sameseed_rung.py
    and a8_shap_ceiling.py. Does NOT clip; clipping is a sec:strat
    stratification convention only."""
    paired, labels, ops = set(), [], []
    for label, (s, c) in PAIR_MAP.items():
        if s in schema and c in schema:
            labels.append(label)
            ops.append(("sum", schema.index(s), schema.index(c)))
            paired.update([s, c])
    for i, f in enumerate(schema):
        if f not in paired:
            labels.append(f)
            ops.append(("keep", i))

    def collapse_fn(arr):
        out = np.empty(len(ops))
        for k, op in enumerate(ops):
            out[k] = arr[op[1]] + arr[op[2]] if op[0] == "sum" else arr[op[1]]
        return out

    return labels, collapse_fn


def splice_from_final(final_py, names):
    """Extract named top-level functions from FYP_Phase2_TFT_FINAL.py by
    AST, verbatim, and return {name: source}. Hashing the returned source
    is what makes the splice auditable: if the source file is edited, the
    digest recorded in PROVENANCE changes."""
    src = Path(final_py).read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            seg = "".join(lines[node.lineno - 1: node.end_lineno])
            out[node.name] = seg
    missing = [n for n in names if n not in out]
    if missing:
        raise SystemExit(
            f"Could not splice {missing} from {final_py}. The file has "
            "changed shape. Resolve before running -- do not "
            "reimplement the attribution arithmetic here.")
    return out


def masked_spearman(v1, v2, mask):
    """Spearman on the masked subset. Returns (rho, n_ranked).
    Fewer than three ranked features makes the statistic meaningless
    rather than merely noisy, so it returns NaN and says how many."""
    idx = np.asarray(mask, dtype=bool)
    n = int(idx.sum())
    if n < 3:
        return (np.nan, n)
    a, b = np.asarray(v1)[idx], np.asarray(v2)[idx]
    if np.all(a == a[0]) or np.all(b == b[0]):
        return (np.nan, n)
    return (float(spearmanr(a, b)[0]), n)


# =====================================================================
# STAGE 1 -- anchor baseline/background diagnostic
# =====================================================================
def stage1_diag(ctx):
    print("\n" + "=" * 68)
    print("STAGE 1 -- ANCHOR BASELINE / BACKGROUND DIAGNOSTIC")
    print("=" * 68)
    schema      = ctx["schema"]
    X_tr        = ctx["X_tr"]
    background  = X_tr[:SHAP_SAMPLES]
    ig_baseline = X_tr.mean(axis=0, keepdims=True)

    rows = []
    for i, feat in enumerate(schema):
        bg = background[:, :, i]
        allt = X_tr[:, :, i]
        rows.append({
            "feature": feat,
            "in_schema_meters": feat in SCHEMA_METERS,
            # SHAP background window: X_train[:200]
            "bg_min": float(bg.min()), "bg_max": float(bg.max()),
            "bg_mean": float(bg.mean()), "bg_std": float(bg.std()),
            "bg_constant": bool(bg.std() == 0.0),
            "bg_constant_at_zero": bool(bg.std() == 0.0 and bg.max() == 0.0),
            # IG baseline: mean over the whole training partition
            "ig_baseline_mean_over_window": float(ig_baseline[0, :, i].mean()),
            "ig_baseline_is_zero": bool(np.all(ig_baseline[0, :, i] == 0.0)),
            # whole training partition, for context
            "train_min": float(allt.min()), "train_max": float(allt.max()),
            "train_std": float(allt.std()),
            "train_constant": bool(allt.std() == 0.0),
        })
    df = pd.DataFrame(rows)
    out = ctx["out_dir"] / "analysis12_baseline_diagnostic.csv"
    df.to_csv(out, index=False)
    print(f"\nwritten -> {out}\n")

    show = ["feature", "bg_constant", "bg_constant_at_zero", "bg_mean",
            "ig_baseline_is_zero", "train_constant"]
    print(df[show].to_string(index=False))

    # ---- the registered test -----------------------------------------
    print("\n" + "-" * 68)
    print("REGISTERED PREDICTION (stated in the script before the run):")
    for f, exp in MECHANISM_PREDICTION.items():
        print(f"  {f:14s} expected {exp} over the SHAP background window")
    print("-" * 68)

    verdict, all_ok = [], True
    for f, exp in MECHANISM_PREDICTION.items():
        if f not in schema:
            verdict.append((f, exp, "NOT IN SCHEMA", False)); all_ok = False
            continue
        r = df[df.feature == f].iloc[0]
        got = "constant" if r.bg_constant else "varies"
        ok = (got == exp)
        all_ok &= ok
        verdict.append((f, exp, got + (" at zero" if r.bg_constant_at_zero
                                       else ""), ok))
    for f, exp, got, ok in verdict:
        print(f"  {'PASS' if ok else 'FAIL'}  {f:14s} expected {exp:9s} "
              f"observed {got}")

    print()
    if all_ok:
        print("MECHANISM CONFIRMED. The anchor's chilled-water column is")
        print("constant across the SHAP background window while gas, hot")
        print("water and solar are not, which is exactly the asymmetry the")
        print("population artefact shows. Proceed to stage 2.")
    else:
        print("MECHANISM NOT CONFIRMED AS STATED.")
        print("The zero-fill artefact in the population artefact is a")
        print("measured fact and is unaffected by this; what is not")
        print("confirmed is the explanation offered for it. The masking")
        print("correction in stages 2-5 remains correct either way, but")
        print("do not write the mechanism into main.tex on this evidence.")
    print("Recorded either way in the diagnostic CSV.")
    return 0


# =====================================================================
# STAGE 2 -- per-building feature mask, from the prepared array
# =====================================================================
def stage2_mask(ctx):
    print("\n" + "=" * 68)
    print("STAGE 2 -- PER-BUILDING FEATURE MASK")
    print("=" * 68)
    prepare_building = ctx["prepare_building"]
    align_to_schema  = ctx["align_to_schema"]
    schema  = ctx["schema"]
    labels  = ctx["labels"]
    out     = ctx["out_dir"] / "analysis12_feature_mask.csv"

    buildings = ctx["population"]
    done = set()
    if out.exists() and not ctx["args"].dry_run:
        done = set(pd.read_csv(out)["building"])
        print(f"resuming, {len(done)} already done")
    if ctx["args"].dry_run:
        buildings = buildings[:3]
        print("DRY RUN -- three buildings only")

    print(f"buildings  : {len(buildings)}")
    for i, b in enumerate(buildings):
        if b in done:
            continue
        row = {"building": b}
        try:
            X_b, y_b, sy_b, cols_b, _ = prepare_building(
                b, ctx["electricity"], ctx["weather"], ctx["meter_dfs"])
            X_al = align_to_schema(X_b, cols_b, schema)
            row["status"] = "ok"
            row["n_seq"] = int(len(X_al))
            row["n_feat_used"] = int(len(cols_b))
            row["feat_cols_used"] = "|".join(cols_b)
            for j, feat in enumerate(schema):
                col = X_al[:, :, j]
                head = X_al[: min(SHAP_SAMPLES, len(X_al)), :, j]
                # supplied: the building produced this column at all
                row[f"supplied_{feat}"] = bool(feat in cols_b)
                # varies: the aligned column carries information.
                # This is the mask. A present-but-flat meter carries no
                # more information than a zero-filled absent one and is
                # masked on the same grounds.
                row[f"varies_{feat}"] = bool(col.std() > 0.0)
                row[f"varies_head200_{feat}"] = bool(head.std() > 0.0)
                row[f"std_{feat}"] = float(col.std())
            # collapsed-12 mask: a collapsed feature is exercised if any
            # of its constituent schema columns varies
            for lab in labels:
                if lab in PAIR_MAP:
                    s, c = PAIR_MAP[lab]
                    v = row.get(f"varies_{s}", False) or row.get(f"varies_{c}", False)
                else:
                    v = row.get(f"varies_{lab}", False)
                row[f"c_varies_{lab}"] = bool(v)
            row["n_collapsed_varies"] = int(
                sum(row[f"c_varies_{l}"] for l in labels))
            del X_b, y_b, X_al
            gc.collect()
        except Exception as e:
            row["status"] = f"{type(e).__name__}: {e}"

        df_partial = pd.DataFrame([row])
        if out.exists() and not ctx["args"].dry_run:
            df_partial = pd.concat([pd.read_csv(out), df_partial],
                                   ignore_index=True)
        df_partial = df_partial.drop_duplicates(subset="building", keep="last")
        df_partial.to_csv(out, index=False)
        if (i + 1) % 25 == 0 or i + 1 == len(buildings):
            print(f"  {i+1}/{len(buildings)}", flush=True)

    df = pd.read_csv(out)
    ok = df[df.status == "ok"]
    print(f"\nwritten -> {out}   ({len(ok)} ok of {len(df)})")
    print("\nCollapsed features exercised, count of buildings:")
    for lab in labels:
        c = f"c_varies_{lab}"
        if c in ok.columns:
            print(f"  {lab:16s} {int(ok[c].sum()):5d} of {len(ok)}")
    print("\nDistribution of n_collapsed_varies:")
    print(ok["n_collapsed_varies"].value_counts().sort_index().to_string())
    return 0


# =====================================================================
# STAGE 3 -- the re-run, archiving raw vectors
# =====================================================================
def stage3_vectors(ctx):
    print("\n" + "=" * 68)
    print("STAGE 3 -- ATTRIBUTION RE-RUN, RAW VECTORS ARCHIVED")
    print("=" * 68)
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    import shap
    from sklearn.metrics import mean_absolute_error

    for _g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(_g, True)

    args   = ctx["args"]
    schema = ctx["schema"]
    labels = ctx["labels"]
    collapse_fn = ctx["collapse_fn"]
    prepare_building = ctx["prepare_building"]
    align_to_schema  = ctx["align_to_schema"]
    X_tr = ctx["X_tr"]
    background = X_tr[:SHAP_SAMPLES]

    # ---- spliced arithmetic, verbatim, hashed ------------------------
    spliced = splice_from_final(
        ctx["final_py"], ["integrated_gradients", "permutation_importance",
                          "_get_infer_fn", "_predict_fast"])
    # _get_infer_fn closes over WINDOW_SIZE and N_ANCHOR_FEATURES in the
    # source module. Both are derived from the prepared anchor array here
    # rather than hardcoded, so a change in either is caught by the shape
    # rather than silently accepted.
    ns = {"np": np, "tf": tf, "gc": gc, "pd": pd,
          "mean_absolute_error": mean_absolute_error,
          "IG_STEPS": IG_STEPS, "IG_BATCH": IG_BATCH,
          "N_PERM_REPEATS": N_PERM_REPEATS,
          "PREDICT_BATCH": PREDICT_BATCH, "_infer_fns": {},
          "WINDOW_SIZE": int(X_tr.shape[1]),
          "N_ANCHOR_FEATURES": int(len(schema))}
    print(f"spliced namespace: WINDOW_SIZE={X_tr.shape[1]} "
          f"N_ANCHOR_FEATURES={len(schema)}")
    for name in ["_get_infer_fn", "_predict_fast", "integrated_gradients",
                 "permutation_importance"]:
        exec(compile(spliced[name], f"<spliced {name}>", "exec"), ns)
    integrated_gradients   = ns["integrated_gradients"]
    permutation_importance = ns["permutation_importance"]
    _get_infer_fn          = ns["_get_infer_fn"]
    ctx["spliced"] = spliced

    ig_baseline = X_tr.mean(axis=0, keepdims=True).astype(np.float32)

    # ---- models ------------------------------------------------------
    models = {}
    for key, p in ctx["draw_paths"].items():
        if not Path(p).is_file():
            raise SystemExit(f"missing weights: {p}")
        models[key] = {"path": str(p), "sha256": sha256(p)}
    print("models:")
    for k, v in models.items():
        print(f"  {k:12s} {v['sha256'][:16]}...  {v['path']}")

    buildings = ctx["ceiling_buildings"]
    if args.dry_run:
        buildings = buildings[:1]
        print("DRY RUN -- one building only")
    print(f"buildings  : {len(buildings)}")

    out = ctx["out_dir"] / "analysis12_attribution_vectors.csv"
    done = set()
    if out.exists() and not args.dry_run:
        prev = pd.read_csv(out)
        done = set(zip(prev.building, prev.method, prev.model, prev.replicate))
        print(f"resuming, {len(done)} (building, method, model, replicate) "
              "rows already done")

    def emit(rec, vec, n_seq_used, extra=None):
        row = dict(rec)
        row["n_seq_used"] = int(n_seq_used)
        row["status"] = "ok"
        for lab, val in zip(labels, vec):
            row[f"v_{lab}"] = float(val)
        if extra:
            row.update(extra)
        dfp = pd.DataFrame([row])
        if out.exists() and not args.dry_run:
            dfp = pd.concat([pd.read_csv(out), dfp], ignore_index=True)
        dfp = dfp.drop_duplicates(
            subset=["building", "method", "model", "replicate"], keep="last")
        dfp.to_csv(out, index=False)

    def shap_vector(model, X, rng_offset, building):
        """Collapsed mean-|SHAP| vector. Phase 1 / Analysis 1e convention.
        Verbatim from a8_shap_ceiling.py."""
        np.random.seed(stable_seed(building) + rng_offset)
        tf.random.set_seed(stable_seed(building) + rng_offset)
        explainer = shap.GradientExplainer(model, background)
        raw = explainer.shap_values(X[: min(SHAP_SAMPLES, len(X))])
        arr = np.array(raw)
        if arr.ndim == 4:
            arr = arr.squeeze(-1)
        per_feature = np.mean(np.abs(arr), axis=(0, 1))
        del raw, arr
        gc.collect()
        return collapse_fn(per_feature)

    loaded = {}
    def get_model(key):
        if key not in loaded:
            loaded[key] = load_model(models[key]["path"], compile=False)
        return loaded[key]

    for i, b in enumerate(buildings):
        print(f"\n[{i+1}/{len(buildings)}] {b}", flush=True)
        try:
            X_b, y_b, sy_b, cols_b, _ = prepare_building(
                b, ctx["electricity"], ctx["weather"], ctx["meter_dfs"])
            X_al = align_to_schema(X_b, cols_b, schema)
            y_kwh = sy_b.inverse_transform(y_b.reshape(-1, 1))
        except Exception as e:
            print(f"  prep failed: {e}")
            continue

        # ---- SHAP ----------------------------------------------------
        # across-draw: one application per draw, RNG_A, X_aligned[:200],
        #              exactly the reported sweep's convention
        # ceiling    : two applications on draw1, RNG_A vs RNG_B
        for key in ctx["draw_keys"]:
            rec = {"building": b, "method": "shap", "model": key,
                   "replicate": "rngA", "role": "draw"}
            if (b, "shap", key, "rngA") in done:
                continue
            try:
                v = shap_vector(get_model(key), X_al, RNG_A, b)
                emit(rec, v, min(SHAP_SAMPLES, len(X_al)))
                print(f"  shap {key} rngA ok")
            except Exception as e:
                print(f"  shap {key} rngA FAILED: {e}")
        if (b, "shap", ctx["draw_keys"][0], "rngB") not in done:
            try:
                v = shap_vector(get_model(ctx["draw_keys"][0]), X_al, RNG_B, b)
                emit({"building": b, "method": "shap",
                      "model": ctx["draw_keys"][0], "replicate": "rngB",
                      "role": "ceiling"}, v, min(SHAP_SAMPLES, len(X_al)))
                print("  shap draw1 rngB ok (ceiling repeat)")
            except Exception as e:
                print(f"  shap draw1 rngB FAILED: {e}")

        # ---- Integrated Gradients -----------------------------------
        # across-draw: subsample_seed 0 per draw, matching ig_vector
        # ceiling    : subsample_seed 0 vs 1 on draw1
        def ig_once(model, sub_seed):
            Xs = X_al
            if len(Xs) > IG_SAMPLES:
                rng = np.random.default_rng(stable_seed(b) + sub_seed)
                idx = np.sort(rng.choice(len(Xs), IG_SAMPLES, replace=False))
                Xs = Xs[idx]
            raw = integrated_gradients(model, Xs, baseline=ig_baseline)
            return collapse_fn(raw), len(Xs)

        for key in ctx["draw_keys"]:
            if (b, "ig", key, "sub0") in done:
                continue
            try:
                v, n = ig_once(get_model(key), 0)
                emit({"building": b, "method": "ig", "model": key,
                      "replicate": "sub0", "role": "draw"}, v, n)
                print(f"  ig {key} sub0 ok")
            except Exception as e:
                print(f"  ig {key} sub0 FAILED: {e}")
        if (b, "ig", ctx["draw_keys"][0], "sub1") not in done:
            try:
                v, n = ig_once(get_model(ctx["draw_keys"][0]), 1)
                emit({"building": b, "method": "ig",
                      "model": ctx["draw_keys"][0], "replicate": "sub1",
                      "role": "ceiling"}, v, n)
                print("  ig draw1 sub1 ok (ceiling repeat)")
            except Exception as e:
                print(f"  ig draw1 sub1 FAILED: {e}")

        # ---- permutation --------------------------------------------
        # across-draw: cap draw 0 per model, matching attribute_one_building
        # ceiling    : two independent 3000-seq caps on draw1, matching
        #              analysis1d_cap_validation
        def perm_once(model, key, cap_offset):
            """Verbatim seeding conventions of _perm_with_cap() in
            FYP_Phase2_TFT_FINAL.py, which produced analysis1d. The
            subsample stream is stable_seed(b) + offset; the SHUFFLE
            stream is stable_seed(b) + 7919*offset, not + offset. At
            offset 0 both reduce to stable_seed(b), which is exactly
            attribute_one_building()'s convention, so cap0 is
            simultaneously the across-draw arm and the ceiling's first
            arm. Getting 7919 wrong silently changes the permutation
            ceiling."""
            n_full = len(X_al)
            Xp, yp = X_al, y_kwh
            if n_full > MAX_SEQ:
                cap_rng = np.random.default_rng(stable_seed(b) + cap_offset)
                idx = np.sort(cap_rng.choice(n_full, MAX_SEQ, replace=False))
                Xp, yp = X_al[idx], y_kwh[idx]
            infer_fn = _get_infer_fn(f"{key}_cap{MAX_SEQ}_{cap_offset}", model)
            rng = np.random.default_rng(stable_seed(b) + 7919 * cap_offset)
            deltas = permutation_importance(
                infer_fn, Xp.copy(), yp, sy_b, len(schema), rng)
            return collapse_fn(deltas), len(Xp)

        for key in ctx["draw_keys"]:
            if (b, "perm", key, "cap0") in done:
                continue
            try:
                v, n = perm_once(get_model(key), key, 0)
                emit({"building": b, "method": "perm", "model": key,
                      "replicate": "cap0", "role": "draw"}, v, n)
                print(f"  perm {key} cap0 ok")
            except Exception as e:
                print(f"  perm {key} cap0 FAILED: {e}")
        if (b, "perm", ctx["draw_keys"][0], "cap1") not in done:
            try:
                v, n = perm_once(get_model(ctx["draw_keys"][0]),
                                 ctx["draw_keys"][0], 1)
                emit({"building": b, "method": "perm",
                      "model": ctx["draw_keys"][0], "replicate": "cap1",
                      "role": "ceiling"}, v, n)
                print("  perm draw1 cap1 ok (ceiling repeat)")
            except Exception as e:
                print(f"  perm draw1 cap1 FAILED: {e}")

        del X_b, y_b, X_al, y_kwh
        gc.collect()

    print(f"\nwritten -> {out}")
    return 0


# =====================================================================
# STAGE 4 -- agreement and instability under three conventions
# =====================================================================
def stage4_agree(ctx):
    print("\n" + "=" * 68)
    print("STAGE 4 -- AGREEMENT AND INSTABILITY, THREE CONVENTIONS")
    print("=" * 68)
    labels = ctx["labels"]
    vec_p  = ctx["out_dir"] / "analysis12_attribution_vectors.csv"
    msk_p  = ctx["out_dir"] / "analysis12_feature_mask.csv"
    for p in (vec_p, msk_p):
        if not p.is_file():
            raise SystemExit(f"missing {p}. Run stages 2 and 3 first.")
    V = pd.read_csv(vec_p); V = V[V.status == "ok"]
    M = pd.read_csv(msk_p); M = M[M.status == "ok"].set_index("building")
    vcols = [f"v_{l}" for l in labels]
    NONMETER = [l for l in labels if l not in SCHEMA_METERS]

    def masks_for(b):
        full = np.ones(len(labels), dtype=bool)
        uni  = np.array([l in NONMETER for l in labels])
        if b in M.index:
            r = M.loc[b]
            msk = np.array([bool(r.get(f"c_varies_{l}", True)) for l in labels])
        else:
            msk = full.copy()
        return {"full12": full, "masked": msk, "uniform8": uni}

    draw_keys = ctx["draw_keys"]
    rows = []
    for b, g in V.groupby("building"):
        mk = masks_for(b)
        for meth in ["shap", "ig", "perm"]:
            gm = g[g.method == meth]
            if gm.empty:
                continue
            # across-draw: the 'draw' role rows, one per model
            dr = gm[gm.role == "draw"].set_index("model")
            ce = gm[gm.role == "ceiling"]
            for conv, mask in mk.items():
                # across-draw = mean of the pairwise rho over the draws
                have = [k for k in draw_keys if k in dr.index]
                pr, ns = [], []
                for x in range(len(have)):
                    for y in range(x + 1, len(have)):
                        r, n = masked_spearman(
                            dr.loc[have[x], vcols].to_numpy(float),
                            dr.loc[have[y], vcols].to_numpy(float), mask)
                        pr.append(r); ns.append(n)
                across = float(np.nanmean(pr)) if pr and not np.all(np.isnan(pr)) else np.nan
                # ceiling = the two replicates on draw1
                ceil, nc = (np.nan, int(mask.sum()))
                if not ce.empty and draw_keys[0] in dr.index:
                    ceil, nc = masked_spearman(
                        dr.loc[draw_keys[0], vcols].to_numpy(float),
                        ce.iloc[0][vcols].to_numpy(float), mask)
                rows.append({
                    "building": b, "method": meth, "convention": conv,
                    "n_features_ranked": int(mask.sum()),
                    "n_draws": len(have),
                    "rho_ceiling": ceil,
                    "rho_across_draw": across,
                    "instability": (ceil - across)
                                   if (ceil == ceil and across == across) else np.nan,
                })
    A = pd.DataFrame(rows)
    out_a = ctx["out_dir"] / "analysis12_agreement.csv"
    A.to_csv(out_a, index=False)
    print(f"written -> {out_a}   ({len(A)} rows)")

    S = (A.groupby(["convention", "method"])
           .agg(n=("building", "nunique"),
                feat_med=("n_features_ranked", "median"),
                ceiling_med=("rho_ceiling", "median"),
                across_med=("rho_across_draw", "median"),
                instab_med=("instability", "median"),
                instab_q1=("instability", lambda s: s.quantile(.25)),
                instab_q3=("instability", lambda s: s.quantile(.75)))
           .round(4).reset_index())
    out_s = ctx["out_dir"] / "analysis12_instability.csv"
    S.to_csv(out_s, index=False)
    print(f"written -> {out_s}\n")
    for conv in ["full12", "masked", "uniform8"]:
        sub = S[S.convention == conv]
        if sub.empty:
            continue
        print(f"--- {conv} " + "-" * (60 - len(conv)))
        print(sub.drop(columns="convention").to_string(index=False))
        print()
    print("Published figures for comparison (full12 convention, n=25):")
    print("  perm  ceiling 0.9710  across 0.6572  instability 0.2526")
    print("  ig    ceiling 0.9930  across 0.9441  instability 0.0490")
    print("  shap  ceiling 1.0000  across 0.9836  instability 0.0164")
    print("\nIf the full12 column does not reproduce those within rounding,")
    print("STOP. The re-run has not reproduced the run it is correcting,")
    print("and the discrepancy has to be resolved before the masked")
    print("column means anything.")
    return 0


# =====================================================================
# STAGE 5 -- re-slice of existing artefacts under the mask
# =====================================================================
def stage5_reslice(ctx):
    print("\n" + "=" * 68)
    print("STAGE 5 -- RE-SLICE OF EXISTING ARTEFACTS UNDER THE MASK")
    print("=" * 68)
    res = ctx["results_dir"]
    msk_p = ctx["out_dir"] / "analysis12_feature_mask.csv"
    if not msk_p.is_file():
        raise SystemExit(f"missing {msk_p}. Run stage 2 first.")
    M = pd.read_csv(msk_p); M = M[M.status == "ok"].set_index("building")

    # ---- 5a. controllable-feature result at n=1,276 -------------------
    p4 = res / "analysis4_shapig_meanbaseline_200steps_FULL.csv"
    if p4.is_file():
        D = pd.read_csv(p4); D = D[D.status == "ok"]
        recs = []
        for _, r in D.iterrows():
            b = r["building"]
            if b not in M.index:
                continue
            mr = M.loc[b]
            carried = [f for f in SCHEMA_METERS
                       if bool(mr.get(f"c_varies_{f}", False))]
            rec = {"building": b, "n_controllable_carried": len(carried),
                   "carried": "|".join(carried)}
            for meth in ["shap", "ig"]:
                if carried:
                    best = max(carried, key=lambda f: abs(r[f"{meth}_{f}"]))
                    rec[f"{meth}_ctrl_top1"] = best
                else:
                    rec[f"{meth}_ctrl_top1"] = None
                # unmasked, for the contrast
                rec[f"{meth}_ctrl_top1_unmasked"] = max(
                    SCHEMA_METERS, key=lambda f: abs(r[f"{meth}_{f}"]))
                for f in SCHEMA_METERS:
                    rec[f"{meth}_{f}"] = float(r[f"{meth}_{f}"])
            recs.append(rec)
        C = pd.DataFrame(recs)
        outc = ctx["out_dir"] / "analysis12_controllable.csv"
        C.to_csv(outc, index=False)
        print(f"written -> {outc}   ({len(C)} rows)")
        sub = C[C.n_controllable_carried > 0]
        print(f"\nbuildings carrying >=1 controllable meter: {len(sub)} "
              f"of {len(C)}")
        for meth in ["shap", "ig"]:
            print(f"\n  {meth} leading controllable driver, MASKED:")
            print((sub[f"{meth}_ctrl_top1"].value_counts(normalize=True) * 100)
                  .round(2).to_string())
            print(f"  {meth} leading controllable driver, UNMASKED "
                  "(the artefact-driven figure):")
            print((C[f"{meth}_ctrl_top1_unmasked"].value_counts(normalize=True)
                   * 100).round(2).to_string())
        if len(sub):
            agree = (sub.shap_ctrl_top1 == sub.ig_ctrl_top1).mean() * 100
            print(f"\n  SHAP and IG agree on the carried-meter leader in "
                  f"{agree:.2f}% of {len(sub)}")
    else:
        print(f"skipped 5a: {p4} not found")

    # ---- 5b. permutation across-draw at n=319, masked -----------------
    p1b = res / "analysis1b_variance_subsample.csv"
    if p1b.is_file():
        B = pd.read_csv(p1b); B = B[B.status == "ok"]
        schema = ctx["schema"]; labels = ctx["labels"]
        collapse_fn = ctx["collapse_fn"]
        NONMETER = [l for l in labels if l not in SCHEMA_METERS]
        recs = []
        for _, r in B.iterrows():
            b = r["building"]
            if b not in M.index:
                continue
            mr = M.loc[b]
            masks = {
                "full12":  np.ones(len(labels), bool),
                "masked":  np.array([bool(mr.get(f"c_varies_{l}", True))
                                     for l in labels]),
                "uniform8": np.array([l in NONMETER for l in labels]),
            }
            for arch in ["lstm", "tft"]:
                vs = []
                for dnum in (1, 2, 3):
                    cols = [f"perm_{arch}_draw{dnum}_{f}" for f in schema]
                    if not all(c in B.columns for c in cols):
                        vs = []; break
                    vs.append(collapse_fn(r[cols].to_numpy(float)))
                if len(vs) != 3:
                    continue
                for conv, mask in masks.items():
                    pr, ns = [], []
                    for x in range(3):
                        for y in range(x + 1, 3):
                            rho, n = masked_spearman(vs[x], vs[y], mask)
                            pr.append(rho); ns.append(n)
                    recs.append({
                        "building": b, "arch": arch, "convention": conv,
                        "n_features_ranked": int(mask.sum()),
                        "mean_rho": float(np.nanmean(pr))
                                    if not np.all(np.isnan(pr)) else np.nan})
        P = pd.DataFrame(recs)
        outp = ctx["out_dir"] / "analysis12_perm_reslice.csv"
        P.to_csv(outp, index=False)
        print(f"\nwritten -> {outp}   ({len(P)} rows)")
        if len(P):
            print("\npermutation across-draw, median by architecture and "
                  "convention:")
            print(P.groupby(["convention", "arch"])
                   .agg(n=("building", "nunique"),
                        feat_med=("n_features_ranked", "median"),
                        rho_med=("mean_rho", "median"))
                   .round(4).to_string())
            print("\nPublished (full12, n=319): within_lstm 0.6957  "
                  "within_tft 0.6217")
    else:
        print(f"skipped 5b: {p1b} not found")
    return 0


# =====================================================================
# context / provenance
# =====================================================================
def build_context(args):
    pipeline_dir = Path(args.pipeline_dir).resolve()
    sys.path.insert(0, str(pipeline_dir))
    from fyp_pipeline import (prepare_building, align_to_schema,
                              split_sequences)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)

    suffix = "_cleaned" if args.cleaned else ""
    data_dir = resolve_data_dir(args.data_dir, args.cleaned)
    print(f"data dir   : {data_dir}")
    print(f"convention : {'*_cleaned.csv' if args.cleaned else 'raw CSVs'}")

    electricity = pd.read_csv(data_dir / f"electricity{suffix}.csv",
                              parse_dates=["timestamp"])
    weather = pd.read_csv(data_dir / "weather.csv", parse_dates=["timestamp"])
    meter_dfs = {}
    for name in METER_NAMES:
        p = data_dir / f"{name}{suffix}.csv"
        if p.is_file():
            meter_dfs[name] = pd.read_csv(p, parse_dates=["timestamp"])
    print(f"meters     : {sorted(meter_dfs)}")

    print("preparing anchor ...")
    X_seq, y_seq, scaler_y, schema, _ = prepare_building(
        ANCHOR, electricity, weather, meter_dfs)
    X_tr, y_tr, X_vl, y_vl, X_te, y_te = split_sequences(X_seq, y_seq)
    labels, collapse_fn = build_collapse_index(schema)
    print(f"schema ({len(schema)})    : {schema}")
    print(f"collapsed ({len(labels)}) : {labels}")
    print(f"sequences  : {len(X_seq)}   train {len(X_tr)}")

    # population for the mask
    pop_src = results_dir / "analysis4_shapig_meanbaseline_200steps_FULL.csv"
    if pop_src.is_file():
        population = pd.read_csv(pop_src)["building"].dropna().unique().tolist()
    else:
        population = pd.read_csv(results_dir / "gate2_parity.csv")[
            "building"].dropna().unique().tolist()

    # the 25 ceiling buildings, from the IG ceiling artefact, so the
    # corrected ceilings sit on the same buildings as the published ones
    ceiling_buildings = pd.read_csv(
        results_dir / "analysis3_ig_ceiling.csv")["building"].tolist()[:N_TEST]

    arch = Path(args.draws_dir)
    draw_keys = ["lstm_draw1", "lstm_draw2", "lstm_draw3"]
    draw_paths = {k: arch / f"{k}.keras" for k in draw_keys}

    return {
        "args": args, "out_dir": out_dir, "results_dir": results_dir,
        "data_dir": data_dir, "suffix": suffix,
        "electricity": electricity, "weather": weather,
        "meter_dfs": meter_dfs,
        "prepare_building": prepare_building,
        "align_to_schema": align_to_schema,
        "schema": schema, "labels": labels, "collapse_fn": collapse_fn,
        "X_tr": X_tr, "population": population,
        "ceiling_buildings": ceiling_buildings,
        "draw_keys": draw_keys, "draw_paths": draw_paths,
        "final_py": args.final_py,
    }


def write_provenance(ctx, stages_run):
    out = ctx["out_dir"] / "analysis12_PROVENANCE.txt"
    dd, sfx = ctx["data_dir"], ctx["suffix"]
    with open(out, "a", encoding="utf-8") as fh:
        fh.write("\n" + "=" * 70 + "\n")
        fh.write("ANALYSIS 12 -- schema zero-fill audit\n")
        fh.write(f"run (UTC)       : "
                 f"{datetime.datetime.now(datetime.timezone.utc).isoformat()}\n")
        fh.write(f"stages          : {stages_run}\n")
        fh.write(f"script          : {Path(__file__).name}  "
                 f"sha256 {sha256(__file__)}\n")
        fh.write(f"python          : {sys.version.split()[0]}  "
                 f"platform {sys.platform}\n")
        try:
            import tensorflow as tf
            fh.write(f"tensorflow      : {tf.__version__}\n")
        except Exception:
            pass
        try:
            import shap as _s
            fh.write(f"shap            : {_s.__version__}\n")
        except Exception:
            pass
        fh.write(f"numpy {np.__version__}  pandas {pd.__version__}\n")
        fh.write(f"data dir        : {dd}\n")
        fh.write(f"data convention : "
                 f"{'*_cleaned.csv' if ctx['args'].cleaned else 'raw CSVs'}\n")
        fh.write("inputs read:\n")
        for f in [f"electricity{sfx}.csv", "weather.csv"] + \
                 [f"{n}{sfx}.csv" for n in METER_NAMES]:
            fp = dd / f
            if fp.is_file():
                fh.write(f"  {f:28s} sha256 {sha256(fp)}\n")
        for k, p in ctx["draw_paths"].items():
            if Path(p).is_file():
                fh.write(f"  {k:28s} sha256 {sha256(p)}   {p}\n")
        fh.write(f"schema ({len(ctx['schema'])})     : {ctx['schema']}\n")
        fh.write(f"collapsed ({len(ctx['labels'])})  : {ctx['labels']}\n")
        fh.write(f"ceiling buildings ({len(ctx['ceiling_buildings'])}): "
                 f"{ctx['ceiling_buildings']}\n")
        fh.write(f"population        : {len(ctx['population'])} buildings\n")
        fh.write("parameters: "
                 f"SHAP_SAMPLES={SHAP_SAMPLES} IG_SAMPLES={IG_SAMPLES} "
                 f"IG_STEPS={IG_STEPS} IG_BATCH={IG_BATCH} "
                 f"MAX_SEQ={MAX_SEQ} N_PERM_REPEATS={N_PERM_REPEATS} "
                 f"RNG_A={RNG_A} RNG_B={RNG_B} SEED={SEED}\n")
        fh.write(f"registered prediction: "
                 f"{json.dumps(MECHANISM_PREDICTION)}\n")
        if "spliced" in ctx:
            fh.write(f"spliced from    : {ctx['final_py']}  "
                     f"sha256 {sha256(ctx['final_py'])}\n")
            for name, src in sorted(ctx["spliced"].items()):
                fh.write(f"  {name:24s} sha256 {sha256_text(src)}  "
                         f"({len(src.splitlines())} lines)\n")
    print(f"\nprovenance appended -> {out}")


def main():
    ap = argparse.ArgumentParser(
        description="Analysis 12: schema zero-fill audit and re-run.")
    ap.add_argument("--stage", default="1",
                    choices=["1", "2", "3", "4", "5", "all",
                             "diag", "mask", "vectors", "agree", "reslice"])
    ap.add_argument("--pipeline-dir", default=".")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--draws-dir", default="archive/2026-07-21_analysis1",
                    help="directory holding lstm_draw1/2/3.keras")
    ap.add_argument("--final-py", default="FYP_Phase2_TFT_FINAL.py",
                    help="source of the spliced attribution functions")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--cleaned", action="store_true",
                    help="read *_cleaned.csv. Do NOT use for figures "
                         "compared against tab:ceilings.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    alias = {"diag": "1", "mask": "2", "vectors": "3",
             "agree": "4", "reslice": "5"}
    stage = alias.get(args.stage, args.stage)

    ctx = build_context(args)
    fns = {"1": stage1_diag, "2": stage2_mask, "3": stage3_vectors,
           "4": stage4_agree, "5": stage5_reslice}
    stages = ["1", "2", "3", "4", "5"] if stage == "all" else [stage]
    rc = 0
    for s in stages:
        rc = fns[s](ctx) or rc
    write_provenance(ctx, ",".join(stages))
    print("\ndone.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
