#!/usr/bin/env python3
"""
a8_shap_ceiling.py
==================
ANALYSIS 8 -- SHAP ESTIMATION CEILING

THE GAP THIS PLUGS
------------------
Table~\\ref{tab:ceilings} in main.tex carries measured estimation
ceilings for two of the three attribution methods:

    permutation, LSTM   median rho = 0.9710   analysis1d_cap_validation.csv
    permutation, TFT    median rho = 0.9855   analysis1d_cap_validation_tft.csv
    Integrated Gradients, LSTM   median 0.9930 / mean 0.9938 / min 0.9650
                                          analysis3_ig_ceiling.csv
    SHAP, LSTM          [VERIFY -- no artefact exists]

The SHAP row has never been computed. The figure that was proposed for
it (median 0.9836, mean 0.9809, min 0.9487) comes from
analysis1e_shap_draws_lstm.csv, which holds ACROSS-DRAW correlations --
three differently seeded models -- not two applications of the method
to ONE frozen model. Using it as a ceiling would divide SHAP's
across-draw agreement by itself and manufacture an instability of
approximately zero. Confirmed by recomputation, 2026-09-03: that file's
median/mean/min of `mean_rho` are 0.98363 / 0.98093 / 0.94872, which
are exactly the disputed figures.

This script computes the missing ceiling.

DEFINITION BEING MEASURED
-------------------------
From main.tex sec:ceiling: the estimation ceiling is the rank
correlation between two independent applications of an attribution
method to a single frozen model, holding weights, building and data
fixed, and varying only the stochastic elements of the estimator.

For SHAP under this project's conventions the stochastic element is
GradientExplainer's own sampling. The reported SHAP sweep explains
X_aligned[:200] -- the FIRST 200 sequences, deterministically -- so the
input set is not a source of variation in the reported values, unlike
Integrated Gradients, whose reported values use a randomly drawn 200.
Two conditions are therefore computed and both are written out.

  A. rho_estimator_only   Same frozen model, same background, same
                          X_aligned[:200]. Two GradientExplainer calls
                          under different NumPy RNG state.
                          *** THIS IS THE CANONICAL CEILING. ***
                          It varies exactly what the reported SHAP
                          values vary and nothing else, so it is the
                          figure that belongs in tab:ceilings and the
                          one Equation (2) must be evaluated against.

  B. rho_matched_to_ig    Additionally draws two independent random
                          200-sequence subsamples, via the same
                          stable_seed(building) + {0,1} convention the
                          IG ceiling uses. Reported as a sensitivity
                          column so that the SHAP and IG ceilings can
                          be compared under one common convention.
                          Do NOT put this in tab:ceilings; it measures
                          a source of variation the reported SHAP
                          values do not contain.

Condition B will be the lower of the two. If it is not, say so and
investigate rather than reporting the more flattering one.

THE OUTCOME THAT IS NOT A CEILING
---------------------------------
If condition A returns rho = 1.000 on every building AND the raw
vectors are bit-identical, GradientExplainer is deterministic under
this project's conventions. That is a legitimate and reportable
result, but it is NOT a measured ceiling and must not be written into
tab:ceilings as though it were. It would mean SHAP's reported values
carry no estimator noise from this source, so SHAP's across-draw
disagreement is attributable to the model in full -- which STRENGTHENS
rather than weakens the instability argument, and changes the
interpretation of Equation (2) for SHAP. The script detects this case
explicitly and says so. Do not tune anything to avoid it.

SCOPE
-----
LSTM only, matching the IG ceiling. The 25 buildings are read from the
IG ceiling artefact so that the two ceilings sit on the same buildings
and are directly comparable. The frozen model is lstm_draw1, which is
the model the reported SHAP portfolio sweep
(analysis4_shapig_meanbaseline_200steps_FULL.csv) was computed on.

CONVENTIONS
-----------
Every parameter below is copied from FYP_Phase2_TFT_FINAL.ipynb
(cell "ANALYSIS 1e / SHAP across draws") and from a6_sameseed_rung.py.
Do not change them.

    background   X_train[:200], anchor training partition
    explainer    shap.GradientExplainer(model, background)
    explain      X_aligned[:200]   (condition A)
    reduce       mean(|.|) over (samples, timesteps) -> per-feature
    4-dim quirk  squeeze(-1) on (200,48,15,1)
    collapse     sum sin/cos pairs, keep singles, schema-derived
    agreement    Spearman rank correlation on the collapsed-12 vector

OUTPUTS
-------
    results/analysis8_shap_ceiling.csv    one row per building
    stdout                                summary + the finished
                                          tab:ceilings row, ready to paste

Resumable: re-running skips buildings already present in the output CSV.

Archive integrity: reads BDGP2 source data and one archived .keras
weight file. Writes only its own output CSV. No archive is modified,
no model is trained, no model is re-saved.

USAGE
-----
    python a8_shap_ceiling.py --pipeline-dir . \
        --data-dir "/mnt/e/Claude/FYP/ZEIT4500 - FYP/Data"

--data-dir is the only path that has to be given, and even that is
auto-discovered from a candidate list if omitted (the list includes the
E: and OneDrive locations this project has used, and $FYP_DATA_DIR).
Add --dry-run to process a single building and stop, as a cheap check
that everything loads before committing to the full 25.

DATA FILE CONVENTION
--------------------
Reads electricity.csv, weather.csv and the plain meter CSVs, matching
FYP_Phase2_TFT_FINAL.py and rerun_tft_confirmation.py -- the scripts that
produced the permutation and IG ceilings this one has to sit alongside.
It does NOT read the *_cleaned.csv variants. a6_sameseed_rung.py does read
those, which is a divergence in the project's own record; for the anchor in
2016 it happens not to matter for the target (both have the same 17 missing
hours and identical values after ffill/bfill) but the meter covariates
differ on 1.5-4.2% of hours. Checked 2026-09-03. Pass --cleaned to switch,
but do not do that for a ceiling that is compared against tab:ceilings.
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
from scipy.stats import spearmanr

# ---------------------------------------------------------------------
# Determinism controls BEFORE importing TensorFlow. SHAP imports TF
# internally, so anything set afterwards is silently ignored. Same
# ordering rule as a6_sameseed_rung.py and the notebook.
# ---------------------------------------------------------------------
SEED = 2016
os.environ.setdefault("PYTHONHASHSEED", str(SEED))

import tensorflow as tf                                     # noqa: E402
for _g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_g, True)

from tensorflow.keras.models import load_model              # noqa: E402
import shap                                                 # noqa: E402

# ---------------------------------------------------------------------
# Fixed parameters -- copied from the notebook, do not alter
# ---------------------------------------------------------------------
ANCHOR       = "Bobcat_education_Alissa"
SHAP_SAMPLES = 200
N_TEST       = 25

# The two independent RNG offsets. Arbitrary but recorded, so the run
# reproduces exactly.
RNG_A = 0
RNG_B = 1

PAIR_MAP = {
    "Hour of Day": ("hour_sin", "hour_cos"),
    "Day of Week": ("dow_sin", "dow_cos"),
    "Month":       ("month_sin", "month_cos"),
}

METER_NAMES = ["chilledwater", "gas", "hotwater", "solar",
               "steam", "water", "irrigation"]

# Where the BDGP2 CSVs have lived on the machines this project has used.
# Tried in order when --data-dir is not given.
DATA_DIR_CANDIDATES = [
    os.environ.get("FYP_DATA_DIR", ""),
    "../Data", "./Data", "../../Data",
    "/mnt/e/Claude/FYP/ZEIT4500 - FYP/Data",
    "/mnt/c/Users/Paul/OneDrive/Desktop/Dump/Uni/Y4/S1/ZEIT4500 - FYP/Data",
    "/mnt/c/Users/pauli/OneDrive/Desktop/Dump/Uni/Y4/S1/ZEIT4500 - FYP/Data",
]


def resolve_data_dir(given, cleaned):
    """Find the BDGP2 directory, or fail with something actionable."""
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
        f"Needed: {elec} and weather.csv in one directory.\n"
        "Tried:\n  " + "\n  ".join(tried) + "\n\n"
        "Pass it explicitly, e.g.\n"
        '  --data-dir "/mnt/e/Claude/FYP/ZEIT4500 - FYP/Data"\n'
        "or set FYP_DATA_DIR. Note this script needs the RAW CSVs "
        "(electricity.csv, gas.csv, ...), not the *_cleaned.csv "
        "variants -- see DATA FILE CONVENTION in the docstring.")


def stable_seed(building, base=SEED):
    """Deterministic per-building seed, independent of PYTHONHASHSEED.
    Copied verbatim from FYP_Phase2_TFT_FINAL.ipynb."""
    h = int(hashlib.md5(building.encode()).hexdigest()[:8], 16)
    return base + (h % 100000)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_collapse_index(schema):
    """15 -> 12 collapse, schema-derived. Copied from a6_sameseed_rung.py."""
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


# ---------------------------------------------------------------------
# One SHAP application. `rng_offset` is the ONLY thing that differs
# between the two applications in condition A.
# ---------------------------------------------------------------------
def shap_vector(model, background, X, collapse_fn, rng_offset, building):
    """Collapsed mean-|SHAP| vector. Phase 1 / Analysis 1e convention.

    GradientExplainer's TF backend samples background references and
    interpolation coefficients from NumPy's legacy global RNG, so
    seeding it here is what makes the two applications independent and
    the run reproducible. The explainer is rebuilt per call, matching
    the notebook.
    """
    np.random.seed(stable_seed(building) + rng_offset)
    tf.random.set_seed(stable_seed(building) + rng_offset)

    explainer = shap.GradientExplainer(model, background)
    raw = explainer.shap_values(X[: min(SHAP_SAMPLES, len(X))])

    arr = np.array(raw)
    if arr.ndim == 4:
        arr = arr.squeeze(-1)               # (200,48,15,1) -> (200,48,15)
    per_feature = np.mean(np.abs(arr), axis=(0, 1))    # -> (15,)
    del raw, arr
    gc.collect()
    return collapse_fn(per_feature)


def main(args):
    sys.path.insert(0, args.pipeline_dir)
    from fyp_pipeline import prepare_building, align_to_schema, split_sequences

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- model, with its digest on the record ------------------------
    model_digest = sha256(args.model)
    print(f"model      : {args.model}")
    print(f"sha256     : {model_digest}")
    if args.archive_model and Path(args.archive_model).exists():
        arc = sha256(args.archive_model)
        match = (arc == model_digest)
        print(f"archive    : {args.archive_model}")
        print(f"  digest match: {match}")
        if not match:
            print("  !! The working copy does not match the archived copy.")
            print("     Stop and resolve this before running. The ceiling")
            print("     must be measured on the weights the reported SHAP")
            print("     values were computed on.")
            return 2
    model = load_model(args.model, compile=False)

    # ---- anchor: schema, background ----------------------------------
    suffix = "_cleaned" if args.cleaned else ""
    data_dir = resolve_data_dir(args.data_dir, args.cleaned)
    print(f"\ndata dir   : {data_dir}")
    print(f"convention : {'*_cleaned.csv' if args.cleaned else 'raw CSVs'} "
          f"({'NOT' if args.cleaned else ''} matching the notebook)")
    print("Preparing anchor ...")
    electricity = pd.read_csv(data_dir / f"electricity{suffix}.csv",
                              parse_dates=["timestamp"])
    weather = pd.read_csv(data_dir / "weather.csv", parse_dates=["timestamp"])
    meter_dfs = {}
    for name in METER_NAMES:
        p = data_dir / f"{name}{suffix}.csv"
        if p.is_file():
            meter_dfs[name] = pd.read_csv(p, parse_dates=["timestamp"])
    print(f"  meters loaded: {sorted(meter_dfs)}")

    X_seq, y_seq, scaler_y, schema, _ = prepare_building(
        ANCHOR, electricity, weather, meter_dfs)
    X_tr, y_tr, X_vl, y_vl, X_te, y_te = split_sequences(X_seq, y_seq)
    background = X_tr[:SHAP_SAMPLES]

    labels, collapse_fn = build_collapse_index(schema)

    # ---- provenance sidecar: what was actually read, and its digest ----
    prov = out_path.parent / (out_path.stem + "_PROVENANCE.txt")
    with open(prov, "w", encoding="utf-8") as fh:
        fh.write("ANALYSIS 8 -- SHAP estimation ceiling\n")
        fh.write(f"run (UTC)        : {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n")
        fh.write(f"script           : {Path(__file__).name}  sha256 {sha256(__file__)}\n")
        fh.write(f"model            : {args.model}  sha256 {model_digest}\n")
        fh.write(f"buildings from   : {args.buildings_from}\n")
        fh.write(f"data dir         : {data_dir}\n")
        fh.write(f"data convention  : {'*_cleaned.csv' if args.cleaned else 'raw CSVs'}\n")
        fh.write("inputs read:\n")
        for f in [f"electricity{suffix}.csv", "weather.csv"] + \
                 [f"{n}{suffix}.csv" for n in METER_NAMES]:
            fp = data_dir / f
            if fp.is_file():
                fh.write(f"  {f:28s} sha256 {sha256(fp)}\n")
        fh.write(f"schema ({len(schema)}) : {schema}\n")
        fh.write(f"collapsed ({len(labels)}) : {labels}\n")
    print(f"  provenance -> {prov}")
    print(f"  schema ({len(schema)}): {schema}")
    print(f"  sequences: {len(X_seq)}  train {len(X_tr)}")
    print(f"  background: X_train[:{SHAP_SAMPLES}]  shape {background.shape}")
    print(f"  collapsed ({len(labels)}): {labels}")

    # ---- the 25 buildings, taken from the IG ceiling artefact --------
    src = pd.read_csv(args.buildings_from)
    buildings = src["building"].tolist()[: args.n_test]
    print(f"\nbuildings  : {len(buildings)} from {args.buildings_from}")
    if args.dry_run:
        buildings = buildings[:1]
        print("  DRY RUN -- one building only")

    done = set()
    if out_path.exists() and not args.dry_run:
        done = set(pd.read_csv(out_path)["building"])
        print(f"  resuming, {len(done)} already done")

    rows = []
    n_identical = 0
    for i, b in enumerate(buildings):
        if b in done:
            continue
        try:
            X_b, y_b, sy_b, cols_b, _ = prepare_building(
                b, electricity, weather, meter_dfs)
            X_al = align_to_schema(X_b, cols_b, schema)

            # -- condition A: canonical ceiling -------------------------
            # identical inputs, independent explainer RNG
            vA1 = shap_vector(model, background, X_al, collapse_fn,
                              RNG_A, b)
            vA2 = shap_vector(model, background, X_al, collapse_fn,
                              RNG_B, b)
            bit_identical = np.array_equal(vA1, vA2)
            n_identical += int(bit_identical)

            # -- condition B: matched to the IG convention --------------
            # independent random 200-sequence subsamples as well
            vB = []
            for off in (RNG_A, RNG_B):
                rng = np.random.default_rng(stable_seed(b) + off)
                if len(X_al) > SHAP_SAMPLES:
                    idx = np.sort(rng.choice(len(X_al), SHAP_SAMPLES,
                                             replace=False))
                    Xs = X_al[idx]
                else:
                    Xs = X_al
                vB.append(shap_vector(model, background, Xs, collapse_fn,
                                      off, b))

            rows.append({
                "building": b,
                "data_convention": "cleaned" if args.cleaned else "raw",
                "rho_estimator_only": spearmanr(vA1, vA2)[0],
                "rho_matched_to_ig": spearmanr(vB[0], vB[1])[0],
                "top1_a": labels[int(np.argmax(vA1))],
                "top1_b": labels[int(np.argmax(vA2))],
                "bit_identical": bit_identical,
                "max_abs_diff": float(np.max(np.abs(vA1 - vA2))),
                "n_seq_available": int(len(X_al)),
                "status": "ok",
            })
            del X_b, y_b, X_al
            gc.collect()
        except Exception as e:
            rows.append({"building": b,
                         "status": f"{type(e).__name__}: {e}"})

        # write ONLY the newest row, concatenated onto what is on disk.
        # Writing the whole accumulated `rows` list each iteration and
        # concatenating produced triangular duplication (1+2+...+n rows).
        if rows:
            df_partial = pd.DataFrame([rows[-1]])
            if out_path.exists() and not args.dry_run:
                df_partial = pd.concat(
                    [pd.read_csv(out_path), df_partial], ignore_index=True)
            df_partial = df_partial.drop_duplicates(
                subset="building", keep="last")
            df_partial.to_csv(out_path, index=False)
        print(f"  {i+1}/{len(buildings)}  {b}", flush=True)

    # ---- summary -----------------------------------------------------
    df = pd.read_csv(out_path)
    ok = df[df.status == "ok"]
    print(f"\n{'='*66}\nSHAP ESTIMATION CEILING (lstm_draw1, n={len(ok)})\n{'='*66}")

    if len(ok) == 0:
        print("  no successful buildings -- nothing to report")
        return 1

    a = ok["rho_estimator_only"]
    b_ = ok["rho_matched_to_ig"]
    print(f"  A. estimator only   median {a.median():.4f}  "
          f"mean {a.mean():.4f}  min {a.min():.4f}")
    print(f"  B. matched to IG    median {b_.median():.4f}  "
          f"mean {b_.mean():.4f}  min {b_.min():.4f}")
    print(f"  top-1 identical under A: "
          f"{100*(ok.top1_a==ok.top1_b).mean():.0f}%")
    print(f"  largest per-feature disagreement under A: "
          f"{ok.max_abs_diff.max():.3e}")

    if bool(ok["bit_identical"].all()):
        print("\n  *** GradientExplainer returned BIT-IDENTICAL vectors on")
        print("      every building under condition A. Under this project's")
        print("      conventions the estimator is DETERMINISTIC, and A is")
        print("      1.000 by construction rather than by measurement.")
        print("      Report it as such. Do not write 1.000 into")
        print("      tab:ceilings as a measured ceiling. The consequence is")
        print("      that SHAP's reported across-draw disagreement carries")
        print("      no estimator noise from this source and is")
        print("      attributable to the model in full -- which strengthens")
        print("      the instability argument. Condition B remains a valid")
        print("      measured quantity and is the figure to discuss.")
    elif b_.median() > a.median():
        print("\n  !! Condition B is HIGHER than condition A, which is not")
        print("     expected -- B varies strictly more. Investigate before")
        print("     reporting either. Do not report the more flattering one.")

    print("\n  Reference ceilings, for comparison:")
    print("    permutation LSTM  0.9710   permutation TFT  0.9855")
    print("    Integrated Gradients LSTM  median 0.9930  mean 0.9938  "
          "min 0.9650")

    print(f"\n  tab:ceilings row, ready to paste:")
    print(f"    SHAP & LSTM & {a.median():.4f} & {a.mean():.4f} & "
          f"{a.min():.4f} \\\\")
    print(f"\n  Instability for SHAP under Equation (2), against the")
    print(f"  across-draw figure of 0.9836 "
          f"(analysis1e_shap_draws_lstm.csv):")
    print(f"    {a.median():.4f} - 0.9836 = {a.median()-0.9836:+.4f}")
    print("    A value at or below zero means the across-draw agreement")
    print("    sits at or above the ceiling, i.e. no measurable model")
    print("    instability for SHAP under this metric. Report that")
    print("    outcome plainly if it occurs.")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline-dir", required=True,
                    help="directory containing fyp_pipeline.py")
    ap.add_argument("--data-dir", default=None,
                    help="directory holding the BDGP2 CSVs; auto-discovered "
                         "from a candidate list (and $FYP_DATA_DIR) if omitted")
    ap.add_argument("--cleaned", action="store_true",
                    help="read *_cleaned.csv instead of the raw CSVs. Do NOT "
                         "use for a ceiling compared against tab:ceilings")
    ap.add_argument("--model", default="draws/lstm_draw1.keras",
                    help="frozen model; must be the draw the reported SHAP "
                         "sweep used")
    ap.add_argument("--archive-model",
                    default="archive/2026-07-21_analysis1/lstm_draw1.keras",
                    help="archived copy; digest is compared, run aborts on "
                         "mismatch")
    ap.add_argument("--buildings-from",
                    default="results/analysis3_ig_ceiling.csv",
                    help="artefact whose building column defines the set, so "
                         "the SHAP and IG ceilings sit on the same buildings")
    ap.add_argument("--n-test", type=int, default=N_TEST)
    ap.add_argument("--out", default="results/analysis8_shap_ceiling.csv")
    ap.add_argument("--dry-run", action="store_true")
    sys.exit(main(ap.parse_args()))
