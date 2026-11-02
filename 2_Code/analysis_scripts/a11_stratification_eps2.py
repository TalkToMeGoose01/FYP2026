#!/usr/bin/env python3
"""
a11_stratification_eps2.py
==========================
ANALYSIS 11 -- BUILDING-TYPE STRATIFICATION, EFFECT SIZES

WHY THIS EXISTS
---------------
`sec:strat` specifies the stratified analysis in detail: Kruskal-Wallis per
collapsed feature, Holm-Bonferroni across the tests for that method,
restricted to types with at least 30 buildings, share-normalised attribution
as the canonical statistic, unnormalised as a sensitivity check, and
"effect size is reported as epsilon^2 and is the basis for all
interpretation". The negative result this produces is named in the title.

No artefact contains it. The only Kruskal-Wallis in the project is a cell of
FYP_Phase2_TFT_FINAL.py which (a) covers SHAP and IG only, not permutation,
(b) reports H and p but never computes epsilon^2, (c) runs on raw magnitudes
rather than the share-normalised canonical statistic, and (d) prints its
results without writing them. Audited 2026-09-03.

This script computes the specified analysis and saves it. It performs no
training and reads only existing result files.

WHAT IT COMPUTES
----------------
For each method, for each of the twelve collapsed features:
    Kruskal-Wallis H across building types with n >= 30
    epsilon^2 = (H - k + 1) / (n - k)
    Holm-Bonferroni adjusted p across the features tested for that method

Under two normalisations:
    share   each building's attribution vector normalised to sum to one
            -- the CANONICAL statistic per sec:strat
    raw     unnormalised magnitudes -- the sensitivity check

Methods:
    shap    gradient, from the full sweep
    ig      gradient, from the full sweep
    perm    permutation importance, LSTM only, collapsed and clipped in the
            order sec:strat specifies (sum sin/cos pairs with signs retained,
            then clip the twelve-vector at zero)

Degenerate columns are reported rather than silently dropped: a feature that
is constant across the population has no defined Kruskal-Wallis statistic
under the tie correction, and the reason is recorded per row.

INPUTS  (read only)
    results/analysis4_shapig_meanbaseline_200steps_FULL.csv
    results/analysis1_perm_both_models.csv
    results/gate2_parity.csv

OUTPUTS
    results/analysis11_stratification_eps2.csv
    results/analysis11_rank_composition.csv
    results/analysis11_stratification_PROVENANCE.txt
"""
import datetime, hashlib, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import kruskal

R = Path("results")
MIN_N = 30
FEATURES = ["Hour of Day", "Day of Week", "Month", "airTemperature",
            "dewTemperature", "windDirection", "windSpeed", "is_weekend",
            "chilledwater", "gas", "hotwater", "solar"]
BASE = ["airTemperature", "dewTemperature", "windDirection", "windSpeed",
        "is_weekend", "chilledwater", "gas", "hotwater", "solar"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def holm(ps):
    """Holm-Bonferroni, step-down with the monotonicity constraint enforced."""
    ps = np.asarray(ps, dtype=float)
    order = np.argsort(ps)
    n = len(ps)
    adj = np.empty(n)
    run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (n - i) * ps[idx])
        adj[idx] = min(run, 1.0)
    return adj


def collapse_perm(df, pre):
    """15 -> 12 for permutation. sec:strat: sum each sin/cos pair with signs
    retained, THEN clip the twelve-vector at zero. The two do not commute."""
    d = {b: df[pre + b] for b in BASE}
    d["Hour of Day"] = df[pre + "hour_sin"] + df[pre + "hour_cos"]
    d["Day of Week"] = df[pre + "dow_sin"] + df[pre + "dow_cos"]
    d["Month"] = df[pre + "month_sin"] + df[pre + "month_cos"]
    return pd.DataFrame(d)[FEATURES].clip(lower=0)


def eps2_table(X, types, method, norm):
    """X: DataFrame of the twelve features, one row per building."""
    X = X.copy()
    if norm == "share":
        s = X.sum(axis=1)
        X = X.div(s.replace(0, np.nan), axis=0)
    rows, ps, idx = [], [], []
    for f in FEATURES:
        col = X[f]
        groups = [g.dropna().to_numpy(float)
                  for _, g in col.groupby(types) if len(g.dropna()) > 0]
        vals = np.concatenate(groups) if groups else np.array([])
        rec = dict(method=method, normalisation=norm, feature=f,
                   n=int(len(vals)), k=len(groups))
        if len(groups) < 2 or len(vals) == 0 or np.allclose(vals, vals[0]):
            rec.update(H=np.nan, eps2=np.nan, p=np.nan, p_holm=np.nan,
                       note="degenerate: constant across the population, "
                            "Kruskal-Wallis undefined under the tie correction")
            rows.append(rec)
            continue
        try:
            H, p = kruskal(*groups)
        except ValueError as e:
            rec.update(H=np.nan, eps2=np.nan, p=np.nan, p_holm=np.nan,
                       note=f"undefined: {e}")
            rows.append(rec)
            continue
        k, n = len(groups), len(vals)
        rec.update(H=H, eps2=(H - k + 1) / (n - k), p=p, p_holm=np.nan, note="")
        rows.append(rec)
        ps.append(p)
        idx.append(len(rows) - 1)
    if ps:
        for j, a in zip(idx, holm(ps)):
            rows[j]["p_holm"] = a
    return rows


def main():
    g2 = pd.read_csv(R / "gate2_parity.csv")[["building", "Type"]]
    a4 = pd.read_csv(R / "analysis4_shapig_meanbaseline_200steps_FULL.csv")
    pm = pd.read_csv(R / "analysis1_perm_both_models.csv")

    a4 = a4.merge(g2, on="building", how="left")
    pm = pm.merge(g2, on="building", how="left")
    keep = a4.Type.value_counts()[lambda s: s >= MIN_N].index
    a4s, pms = a4[a4.Type.isin(keep)], pm[pm.Type.isin(keep)]
    print(f"types with n>={MIN_N}: {sorted(keep)}")
    print(f"stratified population: gradient n={len(a4s)}, permutation n={len(pms)}")

    rows = []
    for norm in ("share", "raw"):
        for m in ("shap", "ig"):
            X = a4s[[f"{m}_{f}" for f in FEATURES]].abs()
            X.columns = FEATURES
            rows += eps2_table(X, a4s.Type.values, m, norm)
        Xp = collapse_perm(pms, "perm_lstm_")
        rows += eps2_table(Xp, pms.Type.values, "perm_lstm", norm)

    out = pd.DataFrame(rows)
    out.to_csv(R / "analysis11_stratification_eps2.csv", index=False)

    for norm in ("share", "raw"):
        print(f"\n{'='*74}\nepsilon^2 by building type -- {norm} "
              f"({'CANONICAL' if norm=='share' else 'sensitivity check'})\n{'='*74}")
        piv = out[out.normalisation == norm].pivot(
            index="feature", columns="method", values="eps2").reindex(FEATURES)
        sig = out[out.normalisation == norm].pivot(
            index="feature", columns="method", values="p_holm").reindex(FEATURES)
        for f in FEATURES:
            line = f"  {f:16s}"
            for m in ("shap", "ig", "perm_lstm"):
                e, p = piv.loc[f, m], sig.loc[f, m]
                if pd.isna(e):
                    line += f"  {m}: {'--':>9s}      "
                else:
                    line += f"  {m}: {e:9.4f}{'*' if p < 0.05 else ' '}     "
            print(line)
        mx = out[(out.normalisation == norm)].groupby("method").eps2.max()
        print(f"  max eps2: " + "  ".join(f"{m} {v:.4f}" for m, v in mx.items()))

    comp = []
    for m in ("shap", "ig"):
        for rk in ("rank1", "rank2", "rank3"):
            ct = pd.crosstab(a4s.Type, a4s[f"{m}_{rk}"], normalize="index") * 100
            ct = ct.reset_index().melt(id_vars="Type", var_name="feature",
                                       value_name="pct")
            ct["method"], ct["rank"] = m, rk
            comp.append(ct[ct.pct > 0])
    pd.concat(comp).to_csv(R / "analysis11_rank_composition.csv", index=False)

    with open(R / "analysis11_stratification_PROVENANCE.txt", "w") as fh:
        fh.write("ANALYSIS 11 -- building-type stratification effect sizes\n")
        fh.write(f"run (UTC): {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n")
        fh.write(f"script   : {Path(__file__).name}  sha256 {sha256(__file__)}\n")
        fh.write(f"min type size: {MIN_N}   types: {sorted(keep)}\n")
        fh.write(f"n (gradient): {len(a4s)}   n (permutation): {len(pms)}\n")
        fh.write("inputs:\n")
        for f in ["gate2_parity.csv",
                  "analysis4_shapig_meanbaseline_200steps_FULL.csv",
                  "analysis1_perm_both_models.csv"]:
            fh.write(f"  {f:52s} sha256 {sha256(R / f)}\n")
        fh.write("eps2 = (H - k + 1)/(n - k); Holm step-down with monotonicity\n")
        fh.write("permutation: LSTM only, sin/cos summed with signs then clipped\n")
    print(f"\nwrote {R}/analysis11_stratification_eps2.csv and rank composition")
    return 0


if __name__ == "__main__":
    sys.exit(main())
