#!/usr/bin/env python3
"""
a13_input_identity.py
=====================
HOW MUCH BUILDING-SPECIFIC INFORMATION IS IN THE MODEL'S INPUT?

WHY THIS EXISTS
---------------
`analysis4_shapig_meanbaseline_200steps_FULL.csv` contains 58 pairs of
DISTINCT buildings carrying bit-identical 12-feature SHAP vectors. 32 of
those pairs span different building types. There are no duplicated IG
vectors at all.

Reading `fyp_pipeline.prepare_building`, the feature matrix is

    feature_cols = weather_present + TIME_COLS + meters_included

The building's own electricity series is the TARGET (`y_data`) and never
an input. Site weather is shared by every building at a site; the six
cyclical encodings and `is_weekend` are functions of the calendar alone.
So building identity enters X only through `meters_included`.

If that reading is right, two buildings at the same site with the same
meter set present a bit-identical input tensor, and a deterministic
explainer must return a bit-identical attribution for them. That is not
a bug in the sweep; it is a property of the feature design, and it bears
directly on whether the stratification-by-type test in `sec:strat` could
have returned a positive result even in principle.

Every one of the 58 duplicate pairs shares a meter set and 54 of 58
share a site, which is consistent. What is NOT yet explained is why the
other same-site same-meter-set buildings do NOT collide -- 272 Rat
buildings with no meters produce 218 distinct SHAP vectors, not one.
Either the input tensors are not in fact identical (something in
preparation varies per building that the code reading missed), or the
explainer is not deterministic across calls in the way the SHAP ceiling
suggested. This script settles which, by comparing the arrays directly
rather than reasoning about them.

WHAT IT DOES
------------
No model, no attribution, no TensorFlow. Prepares buildings and compares
aligned input tensors.

  1. For a sample of (site, meter-set) groups, computes a SHA-256 of
     each building's X_aligned and counts distinct tensors per group.
  2. For every one of the 58 known duplicate-SHAP pairs present in the
     population, reports whether X_aligned is bit-identical, and if not,
     where and by how much it differs.
  3. Reports how many buildings in the whole population carry no
     building-specific input at all (no meter covariate in the schema),
     since for those the input is a pure function of site and calendar.

OUTPUTS
-------
    results/analysis13_input_identity.csv        per building
    results/analysis13_group_summary.csv         per (site, meter set)
    results/analysis13_dup_pairs.csv             the 58 pairs, checked
    results/analysis13_PROVENANCE.txt

USAGE
-----
    python a13_input_identity.py --data-dir "../Data"
    python a13_input_identity.py --data-dir "../Data" --limit 200

Safe to run WHILE a12 stage 3 is running: it loads no model, uses no
GPU, and writes only analysis13_* files. It is single-threaded pandas
and will simply compete for CPU, so if stage 3 is time-critical, run
this afterwards.
"""

import argparse
import datetime
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ANCHOR = "Bobcat_education_Alissa"
METER_NAMES = ["chilledwater", "gas", "hotwater", "solar",
               "steam", "water", "irrigation"]
SCHEMA_METERS = ["chilledwater", "gas", "hotwater", "solar"]

DATA_DIR_CANDIDATES = [
    os.environ.get("FYP_DATA_DIR", ""),
    "../Data", "./Data", "../../Data",
    "/mnt/e/Claude/FYP/ZEIT4500 - FYP/Data",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def arr_digest(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32)
                          .tobytes()).hexdigest()


def resolve_data_dir(given, cleaned=False):
    elec = "electricity_cleaned.csv" if cleaned else "electricity.csv"
    for cand in ([given] if given else DATA_DIR_CANDIDATES):
        if not cand:
            continue
        d = Path(cand)
        if (d / elec).is_file() and (d / "weather.csv").is_file():
            return d
    raise SystemExit("Could not locate the BDGP2 data directory. "
                     'Pass --data-dir "../Data".')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline-dir", default=".")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--cleaned", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N buildings (0 = all)")
    args = ap.parse_args()

    sys.path.insert(0, args.pipeline_dir)
    from fyp_pipeline import prepare_building, align_to_schema

    res = Path(args.results_dir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    dd = resolve_data_dir(args.data_dir, args.cleaned)
    sfx = "_cleaned" if args.cleaned else ""
    print(f"data dir : {dd}")

    electricity = pd.read_csv(dd / f"electricity{sfx}.csv",
                              parse_dates=["timestamp"])
    weather = pd.read_csv(dd / "weather.csv", parse_dates=["timestamp"])
    meter_dfs = {}
    for n in METER_NAMES:
        p = dd / f"{n}{sfx}.csv"
        if p.is_file():
            meter_dfs[n] = pd.read_csv(p, parse_dates=["timestamp"])

    print("preparing anchor for the schema ...")
    _, _, _, schema, _ = prepare_building(ANCHOR, electricity, weather,
                                          meter_dfs)
    print(f"schema ({len(schema)}): {schema}")

    sweep = pd.read_csv(res / "analysis4_shapig_meanbaseline_200steps_FULL.csv")
    sweep = sweep[sweep.status == "ok"].reset_index(drop=True)
    buildings = sweep["building"].tolist()
    if args.limit:
        buildings = buildings[: args.limit]
    print(f"buildings: {len(buildings)}")

    # the 58 known duplicate-SHAP pairs, recomputed here rather than
    # hardcoded, so this script stands alone
    FEAT = [c[len("shap_"):] for c in sweep.columns
            if c.startswith("shap_")
            and c not in ("shap_rank1", "shap_rank2", "shap_rank3",
                          "shap_dom_ratio", "shap_ig_rho")]
    SH = ["shap_" + f for f in FEAT]
    key = sweep[SH].astype(str).agg("|".join, axis=1)
    dupkeys = key.value_counts()
    dupkeys = set(dupkeys[dupkeys > 1].index)
    pairs = []
    for k, g in sweep[key.isin(dupkeys)].groupby(key[key.isin(dupkeys)]):
        bs = g["building"].tolist()
        for x in range(len(bs)):
            for y in range(x + 1, len(bs)):
                pairs.append((bs[x], bs[y]))
    print(f"duplicate-SHAP pairs found in the sweep: {len(pairs)}")

    rows = []
    digests = {}
    for i, b in enumerate(buildings):
        try:
            X_b, y_b, sy_b, cols_b, _ = prepare_building(
                b, electricity, weather, meter_dfs)
            X_al = align_to_schema(X_b, cols_b, schema)
            d_full = arr_digest(X_al)
            d_head = arr_digest(X_al[:200])
            digests[b] = X_al
            carried = [m for m in SCHEMA_METERS if m in cols_b]
            rows.append({
                "building": b,
                "site": b.split("_")[0],
                "meter_set": "|".join(sorted(m for m in cols_b
                                             if m in METER_NAMES)),
                "schema_meters_carried": "|".join(carried),
                "n_schema_meters": len(carried),
                "n_seq": int(len(X_al)),
                "digest_full": d_full,
                "digest_head200": d_head,
                "status": "ok",
            })
            # keep memory bounded: only retain arrays for paired buildings
            if b not in {x for p in pairs for x in p}:
                digests.pop(b, None)
            del X_b, y_b
        except Exception as e:
            rows.append({"building": b, "status": f"{type(e).__name__}: {e}"})
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(buildings)}", flush=True)

    D = pd.DataFrame(rows)
    D.to_csv(out / "analysis13_input_identity.csv", index=False)
    print(f"\nwritten -> {out/'analysis13_input_identity.csv'}")

    ok = D[D.status == "ok"]
    G = (ok.groupby(["site", "meter_set", "n_seq"])
           .agg(n_buildings=("building", "size"),
                distinct_full=("digest_full", "nunique"),
                distinct_head200=("digest_head200", "nunique"))
           .reset_index()
           .sort_values("n_buildings", ascending=False))
    G.to_csv(out / "analysis13_group_summary.csv", index=False)
    print(f"written -> {out/'analysis13_group_summary.csv'}")

    print("\n=== INPUT TENSOR IDENTITY BY (site, meter set) ===")
    print(G[G.n_buildings > 1].head(15).to_string(index=False))
    coll = G[(G.n_buildings > 1) & (G.distinct_full == 1)]
    print(f"\ngroups of >1 building sharing ONE input tensor: "
          f"{len(coll)} of {int((G.n_buildings>1).sum())}")

    print("\n=== THE DUPLICATE-SHAP PAIRS ===")
    prow = []
    for a, b in pairs:
        if a in digests and b in digests:
            Xa, Xb = digests[a], digests[b]
            same_shape = Xa.shape == Xb.shape
            ident = bool(same_shape and np.array_equal(Xa, Xb))
            ident200 = bool(same_shape
                            and np.array_equal(Xa[:200], Xb[:200]))
            maxdiff = float(np.max(np.abs(Xa - Xb))) if same_shape else np.nan
            worst = (schema[int(np.argmax(np.max(np.abs(Xa - Xb),
                                                 axis=(0, 1))))]
                     if same_shape and maxdiff > 0 else "")
        else:
            ident = ident200 = same_shape = None
            maxdiff = np.nan; worst = "not retained"
        prow.append({"a": a, "b": b,
                     "same_site": a.split("_")[0] == b.split("_")[0],
                     "X_identical_full": ident,
                     "X_identical_head200": ident200,
                     "max_abs_diff": maxdiff,
                     "worst_feature": worst})
    P = pd.DataFrame(prow)
    P.to_csv(out / "analysis13_dup_pairs.csv", index=False)
    print(f"written -> {out/'analysis13_dup_pairs.csv'}")
    if len(P):
        print(f"  pairs checked            : {len(P)}")
        print(f"  X identical (full array) : {int(P.X_identical_full.sum())}")
        print(f"  X identical (first 200)  : {int(P.X_identical_head200.sum())}")
        print(P.head(10).to_string(index=False))

    print("\n=== BUILDING-SPECIFIC INPUT CONTENT ===")
    print(ok.n_schema_meters.value_counts().sort_index().to_string())
    none_ = int((ok.n_schema_meters == 0).sum())
    print(f"\nbuildings carrying NO schema meter covariate: {none_} of {len(ok)}"
          f"  ({100*none_/len(ok):.1f}%)")
    print("For these the input tensor is a function of site weather and the")
    print("calendar only. No feature distinguishes one such building from")
    print("another at the same site, so no attribution method can, and a")
    print("stratification-by-type test over them is testing a distinction")
    print("the input does not carry.")

    with open(out / "analysis13_PROVENANCE.txt", "a", encoding="utf-8") as fh:
        fh.write("\n" + "=" * 70 + "\n")
        fh.write("ANALYSIS 13 -- input tensor identity\n")
        fh.write(f"run (UTC) : "
                 f"{datetime.datetime.now(datetime.timezone.utc).isoformat()}\n")
        fh.write(f"script    : {Path(__file__).name}  sha256 {sha256(__file__)}\n")
        fh.write(f"python {sys.version.split()[0]}  numpy {np.__version__}  "
                 f"pandas {pd.__version__}\n")
        fh.write(f"data dir  : {dd}  "
                 f"({'cleaned' if args.cleaned else 'raw'})\n")
        for f in [f"electricity{sfx}.csv", "weather.csv"] + \
                 [f"{n}{sfx}.csv" for n in METER_NAMES]:
            fp = dd / f
            if fp.is_file():
                fh.write(f"  {f:28s} sha256 {sha256(fp)}\n")
        fh.write(f"schema ({len(schema)}) : {schema}\n")
        fh.write(f"buildings : {len(ok)}  duplicate-SHAP pairs: {len(pairs)}\n")
    print(f"\nprovenance -> {out/'analysis13_PROVENANCE.txt'}")
    print("done.")


if __name__ == "__main__":
    main()
