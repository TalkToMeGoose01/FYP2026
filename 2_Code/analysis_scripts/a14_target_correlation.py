#!/usr/bin/env python3
"""
a14_target_correlation.py
=========================
CORRELATION OF EACH METER COVARIATE WITH ITS BUILDING'S OWN ELECTRICITY,
AND THE LAG-1 AUTOCORRELATION OF ELECTRICITY ITSELF

WHY THIS EXISTS
---------------
Section~\\ref{sec:method} records that the model's input comprises
weather, calendar encodings and meter covariates, and that the
building's own electricity series is the prediction target and is never
an input. That design is unusual: of the eighteen time-series energy
forecasting studies tabulated by Arabzadeh and Frank, fifteen include
the target's own history among their input features.

Two questions follow, and this script answers the measurable one.

  1. How strongly would a lagged target term have competed with the
     features actually used? Measured as the lag-1 autocorrelation of
     each building's electricity series, against the correlation of
     each meter covariate with that same series.

  2. What would that have done to the reported attributions? NOT
     ANSWERED HERE, and not answerable without retraining. The figures
     below support an expectation, not a result, and must be reported
     as such.

An earlier plan for this script was to add a lagged column and
recompute the input-tensor digests of Analysis 13, to show that the
461-tensor collapse would dissolve. That was abandoned: every
building's electricity series is unique, so the outcome is derivable by
inspection and running it would confirm arithmetic rather than measure
anything. The Limitations section states it as arithmetic instead.

WHAT IT COMPUTES
----------------
For each meter in {chilledwater, gas, hotwater, solar} and each
building carrying both that meter and electricity in 2016, the Pearson
correlation between the two hourly series over their common non-missing
timestamps. Buildings with fewer than 2,000 common hours, or with a
constant series on either side, are excluded and counted.

Separately, the lag-1 autocorrelation of electricity for every building
in the file, as the scale against which those correlations should be
read.

No model, no attribution, no scaling, no windowing. Raw meter series as
distributed in BDGP2, filtered to 2016 only. This is deliberately
upstream of prepare_building: the question is what signal existed in
the data, not what the pipeline made of it.

OUTPUTS
-------
    results/analysis14_target_correlation.csv        per meter, summary
    results/analysis14_per_building_corr.csv         per building, raw
    results/analysis14_PROVENANCE.txt

USAGE
-----
    python a14_target_correlation.py --data-dir "../Data"

Reads the raw CSVs, matching the convention of every other analysis in
this work. Writes only its own outputs.
"""

import argparse
import datetime
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

YEAR = 2016
MIN_HOURS = 2000
METERS = ["chilledwater", "gas", "hotwater", "solar"]

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


def resolve_data_dir(given):
    for cand in ([given] if given else DATA_DIR_CANDIDATES):
        if not cand:
            continue
        d = Path(cand)
        if (d / "electricity.csv").is_file():
            return d
    raise SystemExit('Could not locate the BDGP2 data directory. '
                     'Pass --data-dir "../Data".')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    dd = resolve_data_dir(args.data_dir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"data dir : {dd}")

    e = pd.read_csv(dd / "electricity.csv", parse_dates=["timestamp"])
    e = e[e.timestamp.dt.year == YEAR]
    print(f"electricity: {len(e)} rows, {e.shape[1]-1} buildings")

    rows, summary = [], []
    for m in METERS:
        p = dd / f"{m}.csv"
        if not p.is_file():
            print(f"  {m}: file absent, skipped"); continue
        d = pd.read_csv(p, parse_dates=["timestamp"])
        d = d[d.timestamp.dt.year == YEAR]
        shared = [c for c in d.columns if c != "timestamp" and c in e.columns]
        rs, skipped = [], 0
        for b in shared:
            x = e[["timestamp", b]].merge(d[["timestamp", b]], on="timestamp",
                                          suffixes=("_e", "_m")).dropna()
            if len(x) < MIN_HOURS:
                skipped += 1; continue
            a, bb = x[f"{b}_e"], x[f"{b}_m"]
            if a.std() == 0 or bb.std() == 0:
                skipped += 1; continue
            r = float(np.corrcoef(a, bb)[0, 1])
            rs.append(r)
            rows.append({"building": b, "meter": m, "n_hours": int(len(x)),
                         "pearson_r": r})
        rs = np.array(rs)
        summary.append({
            "meter": m, "n_buildings": int(len(rs)),
            "n_skipped": int(skipped),
            "median_r": float(np.nanmedian(rs)) if len(rs) else np.nan,
            "q1_r": float(np.nanpercentile(rs, 25)) if len(rs) else np.nan,
            "q3_r": float(np.nanpercentile(rs, 75)) if len(rs) else np.nan,
            "min_r": float(np.nanmin(rs)) if len(rs) else np.nan,
            "max_r": float(np.nanmax(rs)) if len(rs) else np.nan,
        })
        print(f"  {m:14s} n={len(rs):4d} skipped={skipped:4d} "
              f"median r={summary[-1]['median_r']:.3f}")

    # lag-1 autocorrelation of the target itself, for scale
    ac = []
    for b in [c for c in e.columns if c != "timestamp"]:
        s = e[b].dropna()
        if len(s) < MIN_HOURS or s.std() == 0:
            continue
        v = s.autocorr(lag=1)
        if v == v:
            ac.append(float(v))
            rows.append({"building": b, "meter": "electricity_lag1",
                         "n_hours": int(len(s)), "pearson_r": float(v)})
    ac = np.array(ac)
    summary.append({
        "meter": "electricity_lag1", "n_buildings": int(len(ac)),
        "n_skipped": 0,
        "median_r": float(np.nanmedian(ac)), "q1_r": float(np.nanpercentile(ac, 25)),
        "q3_r": float(np.nanpercentile(ac, 75)), "min_r": float(np.nanmin(ac)),
        "max_r": float(np.nanmax(ac)),
    })
    print(f"  {'electricity lag-1':14s} n={len(ac):4d} "
          f"median r={summary[-1]['median_r']:.3f}")

    S = pd.DataFrame(summary)
    S.to_csv(out / "analysis14_target_correlation.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "analysis14_per_building_corr.csv",
                              index=False)
    print(f"\nwritten -> {out/'analysis14_target_correlation.csv'}")
    print(f"written -> {out/'analysis14_per_building_corr.csv'}")
    print()
    print(S.round(4).to_string(index=False))

    with open(out / "analysis14_PROVENANCE.txt", "a", encoding="utf-8") as fh:
        fh.write("\n" + "=" * 70 + "\n")
        fh.write("ANALYSIS 14 -- target correlation of the meter covariates\n")
        fh.write(f"run (UTC) : "
                 f"{datetime.datetime.now(datetime.timezone.utc).isoformat()}\n")
        fh.write(f"script    : {Path(__file__).name}  sha256 {sha256(__file__)}\n")
        fh.write(f"python {sys.version.split()[0]}  numpy {np.__version__}  "
                 f"pandas {pd.__version__}\n")
        fh.write(f"data dir  : {dd}   (raw CSVs, {YEAR} only)\n")
        for f in ["electricity.csv"] + [f"{m}.csv" for m in METERS]:
            fp = dd / f
            if fp.is_file():
                fh.write(f"  {f:22s} sha256 {sha256(fp)}\n")
        fh.write(f"min common hours: {MIN_HOURS}; constant series excluded\n")
        fh.write("NOTE: these figures support an expectation about what a "
                 "lagged target term would have done to the reported "
                 "attributions. They do not measure it. Measuring it "
                 "requires retraining.\n")
    print(f"\nprovenance -> {out/'analysis14_PROVENANCE.txt'}")


if __name__ == "__main__":
    main()
