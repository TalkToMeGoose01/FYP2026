# -*- coding: utf-8 -*-
"""ANALYSIS 15 -- archive the three quantities that were reported without a
result file (review items B1-B3).  Nothing is recomputed differently: this
re-derives, from artefacts that already exist, the figures already in the
report, and writes them with a provenance sidecar so that every reported
number resolves to a named file as Section IV requires.

  B1  between-architecture permutation agreement, n=319, all three
      conventions, with the paired tests           -> analysis15_between_arch*
  B2a de-duplicated building-type stratification    -> analysis15_dedup_stratification.csv
  B2b within-cluster vs between-type dispersion     -> analysis15_cluster_noise.csv
  B3  attribution concentration, both methods on
      one sample and one convention                 -> analysis15_concentration.csv

No model is loaded and no training is performed.
"""
import hashlib, json, os, sys, datetime
import numpy as np, pandas as pd
from scipy import stats

R   = sys.argv[1] if len(sys.argv) > 1 else "."
RES = os.path.join(R, "results")
OUT = sys.argv[2] if len(sys.argv) > 2 else RES
INPUTS = {}

def rd(name):
    p = os.path.join(RES, name)
    INPUTS[name] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return pd.read_csv(p)

SCHEMA = ['airTemperature','dewTemperature','windDirection','windSpeed','hour_sin','hour_cos',
          'dow_sin','dow_cos','month_sin','month_cos','is_weekend','chilledwater','gas','hotwater','solar']
LAB = ['Hour of Day','Day of Week','Month','airTemperature','dewTemperature','windDirection',
       'windSpeed','is_weekend','chilledwater','gas','hotwater','solar']
TYPES5 = ['Education','Office','Entertainment/public assembly','Public services','Lodging/residential']

def collapse(v):
    d = dict(zip(SCHEMA, v))
    return np.array([d['hour_sin']+d['hour_cos'], d['dow_sin']+d['dow_cos'], d['month_sin']+d['month_cos'],
                     d['airTemperature'], d['dewTemperature'], d['windDirection'], d['windSpeed'],
                     d['is_weekend'], d['chilledwater'], d['gas'], d['hotwater'], d['solar']])

def masked_spearman(a, b, mask):
    a, b = np.asarray(a)[mask], np.asarray(b)[mask]
    if mask.sum() < 3 or np.all(a == a[0]) or np.all(b == b[0]): return np.nan
    return float(stats.spearmanr(a, b)[0])

def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m); run = 0.0
    for i, idx in enumerate(o):
        run = max(run, (m - i) * p[idx]); adj[idx] = min(run, 1.0)
    return adj

B   = rd("analysis1b_variance_subsample.csv"); B = B[B.status == "ok"]
M   = rd("analysis12_feature_mask.csv"); M = M[M.status == "ok"].set_index("building")
G   = rd("gate2_parity.csv").set_index("building")
F   = rd("analysis4_shapig_meanbaseline_200steps_FULL.csv"); F = F[F.status == "ok"].set_index("building")
P   = rd("analysis1_perm_both_models.csv"); P = P[P.status == "ok"].set_index("building")
I   = rd("analysis13_input_identity.csv").set_index("building")
NONMETER = [l for l in LAB if l not in ('chilledwater','gas','hotwater','solar')]

# ---------------------------------------------------------------- B1
rows = []
for _, r in B.iterrows():
    b = r.building
    if b not in M.index: continue
    mr = M.loc[b]
    masks = {"full12":  np.ones(12, bool),
             "masked":  np.array([bool(mr.get(f"c_varies_{l}", True)) for l in LAB]),
             "uniform8": np.array([l in NONMETER for l in LAB])}
    V = {a: [collapse(r[[f"perm_{a}_draw{d}_{f}" for f in SCHEMA]].to_numpy(float)) for d in (1,2,3)]
         for a in ("lstm","tft")}
    for conv, mk in masks.items():
        wl = np.nanmean([masked_spearman(V['lstm'][i], V['lstm'][j], mk) for i in range(3) for j in range(i+1,3)])
        wt = np.nanmean([masked_spearman(V['tft'][i],  V['tft'][j],  mk) for i in range(3) for j in range(i+1,3)])
        bt = np.nanmean([masked_spearman(V['lstm'][i], V['tft'][j],  mk) for i in range(3) for j in range(3)])
        rows.append({"building": b, "convention": conv, "n_features_ranked": int(mk.sum()),
                     "within_lstm": wl, "within_tft": wt, "between_arch": bt})
BA = pd.DataFrame(rows)
BA.to_csv(os.path.join(OUT, "analysis15_between_arch.csv"), index=False)
srows = []
for conv, d in BA.groupby("convention"):
    d = d.dropna()
    srows.append({"convention": conv, "n": len(d),
        "med_within_lstm": d.within_lstm.median(), "med_within_tft": d.within_tft.median(),
        "med_between": d.between_arch.median(),
        "drop_lstm_med": (d.within_lstm - d.between_arch).median(),
        "drop_lstm_p": stats.wilcoxon(d.within_lstm, d.between_arch).pvalue,
        "drop_tft_med": (d.within_tft - d.between_arch).median(),
        "drop_tft_p": stats.wilcoxon(d.within_tft, d.between_arch).pvalue,
        "lstm_more_selfconsistent_frac": float((d.within_lstm > d.within_tft).mean()),
        "selfconsistency_diff_med": (d.within_lstm - d.within_tft).median(),
        "selfconsistency_p": stats.wilcoxon(d.within_lstm, d.within_tft).pvalue})
SA = pd.DataFrame(srows)
SA.to_csv(os.path.join(OUT, "analysis15_between_arch_summary.csv"), index=False)
print("B1 between-architecture agreement, n=%d per convention" % BA.building.nunique())
print(SA.round(4).to_string(index=False))

# ---------------------------------------------------------------- B2
pop = [b for b in F.index if G.loc[b, "Type"] in TYPES5 and bool(G.loc[b, "pass_primary"])]
def perm_vec(b):
    return np.clip(collapse(P.loc[b, [f"perm_lstm_{f}" for f in SCHEMA]].to_numpy(float)), 0, None)
MATS = {"shap": pd.DataFrame({b: [F.loc[b, f"shap_{l}"] for l in LAB] for b in pop}, index=LAB).T,
        "ig":   pd.DataFrame({b: [F.loc[b, f"ig_{l}"]   for l in LAB] for b in pop}, index=LAB).T,
        "perm_lstm": pd.DataFrame({b: perm_vec(b) for b in pop}, index=LAB).T}
dedup = I.loc[pop].reset_index().drop_duplicates("digest_full").building.tolist()
recs = []
for meth, mat in MATS.items():
    m = mat.loc[dedup]; m = m[m.sum(axis=1) > 0]; m = m.div(m.sum(axis=1), axis=0)
    ty = G.loc[m.index, "Type"]; k = ty.nunique(); n = len(m); sub = []
    for l in LAB:
        if meth == "shap" and l == "Month":
            sub.append((l, np.nan, np.nan, "degenerate: constant across the population")); continue
        groups = [m.loc[ty == t, l].values for t in ty.unique()]
        try: H, p = stats.kruskal(*groups)
        except ValueError:
            sub.append((l, np.nan, np.nan, "degenerate: constant, Kruskal-Wallis undefined")); continue
        sub.append((l, (H - k + 1) / (n - k), p, ""))
    ok = [s for s in sub if not np.isnan(s[1])]
    adj = holm([s[2] for s in ok]); j = 0
    for l, e, p, note in sub:
        if np.isnan(e): recs.append({"method": meth, "dedup_key": "digest_full", "feature": l, "n": n,
                                     "k": k, "eta2_H": np.nan, "p": np.nan, "p_holm": np.nan, "note": note})
        else:
            recs.append({"method": meth, "dedup_key": "digest_full", "feature": l, "n": n, "k": k,
                         "eta2_H": e, "p": p, "p_holm": adj[j], "note": ""}); j += 1
D = pd.DataFrame(recs)
D.to_csv(os.path.join(OUT, "analysis15_dedup_stratification.csv"), index=False)
print("\nB2a de-duplicated stratification (one building per distinct full-year tensor)")
for meth, d in D.groupby("method"):
    dd = d.dropna(subset=["eta2_H"])
    print("  %-10s n=%d  max eta2 %.4f (%s)  significant %d of %d"
          % (meth, d.n.iloc[0], dd.eta2_H.max(), dd.loc[dd.eta2_H.idxmax(), "feature"],
             int((dd.p_holm < .05).sum()), len(dd)))

S = MATS["shap"]; Sh = S.div(S.sum(axis=1), axis=0)
ty_all = G.loc[pop, "Type"]
big = I.loc[pop].digest_full.value_counts().idxmax()
cl = [b for b in pop if I.loc[b, "digest_full"] == big]
nrows = []
for l in LAB:
    for nm, X in (("share", Sh), ("raw", S)):
        w = X.loc[cl, l].std(ddof=1); bt = X[l].groupby(ty_all).mean().std(ddof=1)
        nrows.append({"feature": l, "normalisation": nm, "n_cluster": len(cl),
                      "sd_within_largest_cluster": w, "sd_between_type_means": bt,
                      "ratio_within_over_between": w / bt if bt else np.nan})
N = pd.DataFrame(nrows)
N.to_csv(os.path.join(OUT, "analysis15_cluster_noise.csv"), index=False)
print("\nB2b within-cluster vs between-type dispersion, SHAP, largest cluster n=%d" % len(cl))
print(N[N.normalisation == "share"].set_index("feature")
      [["sd_within_largest_cluster","sd_between_type_means","ratio_within_over_between"]].round(5).to_string())

# ---------------------------------------------------------------- B3
crows = []
for b in B.building:
    if b not in F.index: continue
    pv = np.clip(collapse(P.loc[b, [f"perm_lstm_{f}" for f in SCHEMA]].to_numpy(float)), 0, None)
    sv = np.abs(F.loc[b, [f"shap_{l}" for l in LAB]].values.astype(float))
    r = {"building": b}
    for nm, v in (("perm_lstm", pv), ("shap", sv)):
        s = np.sort(v)[::-1]
        r[f"{nm}_rank1"] = s[0]; r[f"{nm}_rank2"] = s[1]
        r[f"{nm}_dominance"] = np.inf if s[1] == 0 else s[0] / s[1]
        r[f"{nm}_rank1_share"] = s[0] / v.sum() if v.sum() > 0 else np.nan
    crows.append(r)
C = pd.DataFrame(crows)
C.to_csv(os.path.join(OUT, "analysis15_concentration.csv"), index=False)
print("\nB3 concentration, n=%d, permutation clipped at zero, SHAP on |values|" % len(C))
for nm in ("perm_lstm", "shap"):
    d = C[f"{nm}_dominance"]; fin = d[np.isfinite(d)]
    print("  %-10s median dominance (all) %.3f  unbounded %d  median (finite only) %.3f  median rank-1 share %.3f"
          % (nm, d.median(), int(np.isinf(d).sum()), fin.median(), C[f"{nm}_rank1_share"].median()))

with open(os.path.join(OUT, "analysis15_PROVENANCE.txt"), "w") as fh:
    fh.write("ANALYSIS 15 -- amendments B1-B3: archive quantities reported without a result file\n")
    fh.write("run (UTC) : %s\n" % datetime.datetime.now(datetime.timezone.utc).isoformat())
    fh.write("script    : a15_amendments.py  sha256 %s\n" % hashlib.sha256(open(__file__,"rb").read()).hexdigest())
    fh.write("python %s  numpy %s  pandas %s  scipy %s\n" % (sys.version.split()[0], np.__version__,
             pd.__version__, __import__("scipy").__version__))
    fh.write("no model loaded, no training performed; all inputs are existing result files\n\ninputs read:\n")
    for k in sorted(INPUTS): fh.write("  %-52s sha256 %s\n" % (k, INPUTS[k]))
    fh.write("\nconventions:\n")
    fh.write("  permutation collapsed sin/cos with signs retained, then clipped at zero\n")
    fh.write("  stratification share-normalised; de-duplication on the full-year input tensor digest\n")
    fh.write("  eta2_H = (H-k+1)/(n-k); Holm step-down with monotonicity, within method\n")
    fh.write("\noutputs:\n")
    for o in ["analysis15_between_arch.csv","analysis15_between_arch_summary.csv",
              "analysis15_dedup_stratification.csv","analysis15_cluster_noise.csv",
              "analysis15_concentration.csv"]:
        fh.write("  %s\n" % o)
print("\nwritten to", OUT)
