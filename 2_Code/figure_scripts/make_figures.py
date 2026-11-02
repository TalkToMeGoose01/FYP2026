# -*- coding: utf-8 -*-
"""Generate every figure for the ZEIT4500 report from archived artefacts.

Design constraints (IEEEtran conference, two columns):
  single-column figures are drawn at 3.45 in wide, double-column at 7.16 in,
  i.e. at final size, so no LaTeX scaling shrinks the type. Base font 8 pt,
  ticks 7 pt, legends 7 pt -- all >= 7 pt in the printed paper.
  Output is vector PDF. Categorical palette is Okabe-Ito, validated
  colour-blind-safe; every multi-series figure also carries a non-colour
  encoding (marker, hatch or direct label).

No new model runs. Every input is a named result file; SHA-256 digests of
all inputs are written to figures/PROVENANCE.txt.
"""
import hashlib, io, json, os, re, sys, datetime
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch

R = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(OUT, exist_ok=True)
RES = os.path.join(R, "results")
ARC = os.path.join(R, "archive", "2026-08-17_analysis6")

BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"
GREY, LGREY, INK = "#5c5c5c", "#c9c9c9", "#1a1a1a"
COL1, COL2 = 3.50, 7.16   # IEEEtran conference \columnwidth and \textwidth

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "lines.linewidth": 1.2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.edgecolor": GREY, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": GREY, "ytick.color": GREY,
    "legend.frameon": False, "figure.dpi": 200,
    "savefig.bbox": None,
    "pdf.fonttype": 42,
})

INPUTS, MADE, FAILED = {}, [], []

def rd(rel, base=RES):
    p = os.path.join(base, rel)
    with open(p, "rb") as f:
        INPUTS[os.path.relpath(p, R)] = hashlib.sha256(f.read()).hexdigest()
    return pd.read_csv(p)

def tidy(ax, grid="y"):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    if grid: ax.grid(axis=grid, color=LGREY, alpha=.7, zorder=0); ax.set_axisbelow(True)

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf")); plt.close(fig); MADE.append(name)
    print("  wrote", name)

def fig(name, w=COL1, h=2.2):
    return plt.subplots(figsize=(w, h), layout="constrained")

def panels(n, w, h, **kw):
    f, ax = plt.subplots(1, n, figsize=(w, h), layout="constrained", **kw)
    return f, ax

def canvas(w, h):
    """Full-bleed canvas for the schematic figures."""
    f, ax = plt.subplots(figsize=(w, h))
    f.subplots_adjust(0, 0, 1, 1)
    ax.axis("off")
    return f, ax

def guard(fn):
    try:
        fn()
    except Exception as e:
        FAILED.append((fn.__name__, repr(e))); print("  FAILED", fn.__name__, repr(e))

LAB12 = ['Hour of Day','Day of Week','Month','airTemperature','dewTemperature',
         'windDirection','windSpeed','is_weekend','chilledwater','gas','hotwater','solar']
SCHEMA = ['airTemperature','dewTemperature','windDirection','windSpeed','hour_sin','hour_cos',
          'dow_sin','dow_cos','month_sin','month_cos','is_weekend','chilledwater','gas','hotwater','solar']
SHORT = {'airTemperature':'Air temp.','dewTemperature':'Dew temp.','windDirection':'Wind dir.',
         'windSpeed':'Wind speed','is_weekend':'Weekend','chilledwater':'Chilled water',
         'gas':'Gas','hotwater':'Hot water','solar':'Solar','Hour of Day':'Hour of day',
         'Day of Week':'Day of week','Month':'Month'}
TYPES5 = ['Education','Office','Entertainment/public assembly','Public services','Lodging/residential']

def collapse(v):
    d = dict(zip(SCHEMA, v))
    return np.array([d['hour_sin']+d['hour_cos'], d['dow_sin']+d['dow_cos'], d['month_sin']+d['month_cos'],
                     d['airTemperature'], d['dewTemperature'], d['windDirection'], d['windSpeed'],
                     d['is_weekend'], d['chilledwater'], d['gas'], d['hotwater'], d['solar']])

def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra, rb = pd.Series(a).rank().values, pd.Series(b).rank().values
    if np.all(ra == ra[0]) or np.all(rb == rb[0]): return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])

# ---------------------------------------------------------------- F01
def f01_instability():
    a = rd("analysis12_agreement.csv"); a = a[a.convention == "masked"]
    order, names = ["shap", "ig", "perm"], ["SHAP", "Int. Gradients", "Permutation"]
    cols = [GREEN, BLUE, ORANGE]
    f, ax = fig("f", COL1, 2.10); rng = np.random.default_rng(7)
    for i, (m, c) in enumerate(zip(order, cols)):
        v = a[a.method == m].instability.values
        ax.scatter(v, np.full_like(v, i) + rng.uniform(-.13, .13, v.size), s=9, color=c,
                   alpha=.75, linewidths=.4, edgecolors="white", zorder=3)
        q1, md, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([q1, q3], [i - .30] * 2, color=INK, lw=1.0, zorder=4, solid_capstyle="butt")
        ax.plot([md, md], [i - .37, i - .23], color=INK, lw=1.4, zorder=5)
        ax.text(q3 + .022, i - .30, f"median {md:.3f}", ha="left", va="center", fontsize=7, color=INK)
    ax.set_yticks(range(3)); ax.set_yticklabels(names)
    ax.set_xlabel("Instability (ceiling $-$ across-draw agreement)")
    ax.set_xlim(-.04, 1.02); ax.set_ylim(-.62, 2.45)
    tidy(ax, grid="x")
    save(f, "f01_instability_by_method")

# ---------------------------------------------------------------- F02
def f02_ladder():
    B = rd("analysis1b_variance_subsample.csv"); B = B[B.status == "ok"]
    cl = rd("analysis1d_cap_validation.csv").set_index("building")
    ct = rd("analysis1d_cap_validation_tft.csv").set_index("building")
    keep = [b for b in B.building if b in cl.index]
    rows = []
    for _, r in B[B.building.isin(keep)].iterrows():
        V = {a: [collapse(r[[f"perm_{a}_draw{d}_{f}" for f in SCHEMA]].to_numpy(float))
                 for d in (1, 2, 3)] for a in ("lstm", "tft")}
        wl = np.nanmean([spearman(V['lstm'][i], V['lstm'][j]) for i in range(3) for j in range(i+1, 3)])
        wt = np.nanmean([spearman(V['tft'][i], V['tft'][j]) for i in range(3) for j in range(i+1, 3)])
        bt = np.nanmean([spearman(V['lstm'][i], V['tft'][j]) for i in range(3) for j in range(3)])
        rows.append((r.building, cl.loc[r.building, "rho_same_model_same_cap"],
                     ct.loc[r.building, "rho_same_model_same_cap"], wl, wt, bt))
    D = pd.DataFrame(rows, columns=["b", "ceilL", "ceilT", "wl", "wt", "bt"])
    f, ax = fig("f", COL1, 2.35)
    for cc, ww, col, dx in [("ceilL", "wl", BLUE, -.045), ("ceilT", "wt", ORANGE, .045)]:
        for _, r in D.iterrows():
            ax.plot([0+dx, 1+dx, 2+dx], [r[cc], r[ww], r.bt], color=col, alpha=.16, lw=.6, zorder=2)
    for cc, ww, col, nm, mk, dx in [("ceilL", "wl", BLUE, "LSTM", "o", -.045),
                                    ("ceilT", "wt", ORANGE, "TFT", "s", .045)]:
        med = [D[cc].median(), D[ww].median(), D.bt.median()]
        ax.plot([0+dx, 1+dx, 2+dx], med, color=col, lw=1.8, marker=mk, ms=5, zorder=5,
                markerfacecolor="white", markeredgecolor=col, markeredgewidth=1.4, label=nm)
        for k, (x, y) in enumerate(zip([0+dx, 1+dx, 2+dx], med)):
            if k == 2 and col == ORANGE:
                continue                      # both architectures land on 0.550
            off = (0, -14) if col == BLUE else (0, 9)
            if k == 2: off = (0, -14)
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=off,
                        ha="center", fontsize=7, color=col if k != 2 else INK, zorder=6)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Estimation\nceiling", "Same arch.,\nnew seed", "Different\narchitecture"])
    ax.set_xlim(-.35, 2.35); ax.set_ylim(0, 1.10)
    ax.set_ylabel("Spearman rank correlation")
    ax.legend(loc="lower left", handlelength=1.4, ncol=2, columnspacing=1.0)
    tidy(ax)
    save(f, "f02_variance_ladder")

# ---------------------------------------------------------------- F03
def f03_degeneracy():
    I = rd("analysis13_input_identity.csv").set_index("building")
    g = rd("gate2_parity.csv").set_index("building")
    pop = [b for b in I.index if g.loc[b, "Type"] in TYPES5 and g.loc[b, "pass_primary"]]
    S = I.loc[pop].copy(); S["Type"] = g.loc[pop, "Type"]
    sz = S.groupby("digest_full").size().sort_values(ascending=False)
    multi = sz[sz > 1]
    sites = {d: "+".join(sorted(set(S[S.digest_full == d].site))) for d in multi.index}
    f, ax = fig("f", COL1, 2.45)
    xs = np.arange(len(multi))
    ax.bar(xs, multi.values, .74, color=ORANGE, linewidth=0)
    for x, v in zip(xs, multi.values):
        ax.text(x, v + 5, str(v), ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([sites[d] for d in multi.index], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Buildings sharing one tensor"); ax.set_ylim(0, 300)
    ax.text(len(multi) - .4, 250,
            "%d tensors shared by %d buildings;\nthe other %d each hold\na tensor of their own"
            % (len(multi), int(multi.sum()), len(S) - int(multi.sum())),
            fontsize=7, color=GREY, ha="right", va="top")
    tidy(ax)
    save(f, "f03_input_degeneracy")

# ---------------------------------------------------------------- F04
def f04_eta2():
    e = rd("analysis11_stratification_eps2.csv"); e = e[e.normalisation == "share"]
    I = rd("analysis13_input_identity.csv").set_index("building")
    g = rd("gate2_parity.csv").set_index("building")
    F = rd("analysis4_shapig_meanbaseline_200steps_FULL.csv"); F = F[F.status == "ok"].set_index("building")
    P = rd("analysis1_perm_both_models.csv"); P = P[P.status == "ok"].set_index("building")
    pop = [b for b in F.index if g.loc[b, "Type"] in TYPES5 and g.loc[b, "pass_primary"]]
    from scipy import stats as st
    mats = {"shap": pd.DataFrame({b: [F.loc[b, f"shap_{l}"] for l in LAB12] for b in pop}, index=LAB12).T,
            "ig":   pd.DataFrame({b: [F.loc[b, f"ig_{l}"] for l in LAB12] for b in pop}, index=LAB12).T,
            "perm": pd.DataFrame({b: np.clip(collapse(P.loc[b, [f"perm_lstm_{f}" for f in SCHEMA]].to_numpy(float)), 0, None)
                                  for b in pop}, index=LAB12).T}
    dd = I.loc[pop].reset_index().drop_duplicates("digest_full").building.tolist()
    def eta(mat, bl, meth):
        m = mat.loc[bl]; m = m[m.sum(axis=1) > 0]; m = m.div(m.sum(axis=1), axis=0)
        ty = g.loc[m.index, "Type"]; k = ty.nunique(); n = len(m); out = {}
        for l in LAB12:
            if meth == "shap" and l == "Month": continue
            try: H, _ = st.kruskal(*[m.loc[ty == t, l].values for t in ty.unique()])
            except ValueError: continue
            out[l] = (H - k + 1) / (n - k)
        return out
    full = {m: eta(mats[m], pop, m) for m in ("perm", "shap", "ig")}
    ded = {m: eta(mats[m], dd, m) for m in ("perm", "shap", "ig")}
    f, axes = panels(2, COL2, 2.55, sharey=True)
    x = np.arange(len(LAB12)); w = .26
    for ax, dat, ttl in zip(axes, [full, ded], ["All buildings ($n=1{,}157$)", "One building per distinct input"]):
        for k, (m, nm, c, hat) in enumerate([("perm", "Permutation", ORANGE, ""),
                                             ("shap", "SHAP", GREEN, "///"),
                                             ("ig", "Int. Gradients", BLUE, "...")]):
            v = [dat[m].get(l, np.nan) for l in LAB12]
            ax.bar(x + (k - 1) * w, v, w * .92, label=nm, color=c, linewidth=0, hatch=hat)
        ax.axhline(.06, color=GREY, lw=.7, ls=(0, (4, 3)))
        ax.text(len(LAB12) - .4, .062, "conventional medium effect", fontsize=7, color=GREY, ha="right", va="bottom")
        ax.set_xticks(x); ax.set_xticklabels([SHORT.get(l, l) for l in LAB12], rotation=55, ha="right")
        ax.set_title(ttl, fontsize=8, pad=3); tidy(ax)
    axes[0].set_ylabel(r"$\eta^2_H$ (share-normalised)"); axes[0].set_ylim(0, .105)
    axes[0].legend(loc="upper right", ncol=1, handlelength=1.1)
    save(f, "f04_type_discrimination")

# ---------------------------------------------------------------- F05
def f05_attrition():
    g = rd("gate2_parity.csv"); I = rd("analysis13_input_identity.csv").set_index("building")
    gi = g.set_index("building")
    pop = [b for b in I.index if gi.loc[b, "Type"] in TYPES5 and gi.loc[b, "pass_primary"]]
    steps = [("BDGP2 buildings", 1636), ("Scored, both architectures", len(g)),
             ("Inside Gate 2 primary band", int(g.pass_primary.sum())),
             ("Stratified population", len(pop)),
             ("Distinct model inputs", I.loc[pop].digest_full.nunique())]
    f, ax = canvas(COL1, 2.45)
    P = .205
    for i, (nm, v) in enumerate(steps):
        y = 1 - i * P
        w = .40 + .60 * (v / steps[0][1])
        ax.add_patch(FancyBboxPatch((.47 - w / 2, y - .052), w, .104,
                                    boxstyle="round,pad=0.008,rounding_size=0.02",
                                    facecolor=ORANGE if i == 4 else BLUE, edgecolor="none", alpha=.93))
        ax.text(.47, y, f"{nm}: {v:,}", ha="center", va="center", color="white", fontsize=7.2)
        if i:
            ax.add_patch(FancyArrowPatch((.47, y + .152), (.47, y + .060),
                                         arrowstyle="-|>", mutation_scale=7, lw=.9, color=GREY))
            ax.text(.51 + w / 2 * 0, y + .104, f"$-${steps[i-1][1]-v:,}", ha="left",
                    va="center", fontsize=7, color=ORANGE)
    ax.set_xlim(-.02, 1.02); ax.set_ylim(.90 - 4 * P, 1.09)
    save(f, "f05_population_attrition")

# ---------------------------------------------------------------- F06
def f06_search():
    txt = open(os.path.join(R, "screening", "screening_SUMMARY.txt"), "rb").read()
    INPUTS["screening/screening_SUMMARY.txt"] = hashlib.sha256(txt).hexdigest()
    stages = [("Records retrieved", 1113, "Scopus 855, Web of Science 258"),
              ("After DOI de-duplication", 865, "within and across databases"),
              ("Unique across four searches", 817, "searches A-D combined"),
              ("Flagged by triage", 174, "re-initialisation $\\times$ attribution $\\times$ stability"),
              ("Met inclusion criterion", 11, "all from Search D, none in energy")]
    f, ax = canvas(COL1, 2.75)
    P = .255
    for i, (nm, v, note) in enumerate(stages):
        y = 1 - i * P
        wdt = .46 + .54 * (np.log10(v) / np.log10(1113))
        ax.add_patch(FancyBboxPatch((.48 - wdt / 2, y - .058), wdt, .116,
                                    boxstyle="round,pad=0.008,rounding_size=0.02",
                                    facecolor=BLUE if i < 4 else ORANGE, edgecolor="none", alpha=.93))
        ax.text(.48, y, f"{nm}: {v:,}", ha="center", va="center", color="white", fontsize=7.2)
        ax.text(.48, y - .072, note, ha="center", va="top", fontsize=7, color=GREY)
        if i:
            ax.add_patch(FancyArrowPatch((.48, y + .152), (.48, y + .065),
                                         arrowstyle="-|>", mutation_scale=7, lw=.9, color=GREY))
            ax.text(.52, y + .108, f"$-${stages[i-1][1]-v:,}", ha="left", va="center",
                    fontsize=7, color=ORANGE)
    ax.set_xlim(-.02, 1.02); ax.set_ylim(.85 - 4 * P, 1.10)
    save(f, "f06_search_flow")

# ---------------------------------------------------------------- F07
def f07_zerofill():
    F = rd("analysis4_shapig_meanbaseline_200steps_FULL.csv"); F = F[F.status == "ok"].set_index("building")
    M = rd("analysis12_feature_mask.csv").set_index("building")
    meters = ["chilledwater", "gas", "hotwater", "solar"]
    f, axes = plt.subplots(2, 1, figsize=(COL1, 3.05), sharex=True, layout="constrained")
    for ax, meth, nm in zip(axes, ["shap", "ig"], ["SHAP", "Integrated Gradients"]):
        pres, absn = [], []
        for mt in meters:
            has = M.loc[F.index, f"supplied_{mt}"].astype(bool).values
            v = F[f"{meth}_{mt}"].abs().values
            pres.append(np.median(v[has]) if has.any() else 0); absn.append(np.median(v[~has]))
        x = np.arange(len(meters)); w = .36
        ax.bar(x - w/2, pres, w, label="Carries the meter", color=BLUE, linewidth=0)
        ax.bar(x + w/2, absn, w, label="Does not", color=ORANGE, hatch="///", linewidth=0)
        hod = F[f"{meth}_Hour of Day"].abs().median()
        ax.axhline(hod, color=INK, lw=.8, ls=(0, (4, 2)))
        ax.text(3.45, hod, "median Hour of Day", fontsize=7, color=INK, ha="right", va="bottom")
        ax.set_ylabel(nm + "\nmedian $|$attr.$|$", fontsize=8, linespacing=1.4)
        ax.set_ylim(0, hod * 1.42)
        if meth == "shap":
            ax.annotate("exactly zero\nin all 911", (w/2, 0), xytext=(0.62, hod * 0.46),
                        fontsize=7, color=INK, ha="center",
                        arrowprops=dict(arrowstyle="-|>", mutation_scale=6, lw=.7, color=GREY))
            ax.legend(loc="upper left", handlelength=1.1, ncol=2, columnspacing=.8)
        tidy(ax)
    axes[1].set_xticks(np.arange(len(meters)))
    axes[1].set_xticklabels([SHORT[m] for m in meters], rotation=18, ha="right")
    save(f, "f07_zerofill_artefact")

# ---------------------------------------------------------------- F08
def f08_gate2():
    g = rd("gate2_parity.csv"); g = g[g.status == "ok"]
    f, ax = fig("f", COL1, 2.00)
    bins = np.linspace(.6, 1.6, 90)
    ax.hist(np.clip(g.ratio, .6, 1.6), bins=bins, color=BLUE, linewidth=0)
    ax.axvspan(.80, 1.25, color=GREEN, alpha=.10, zorder=0)
    for v, ls in [(.80, "-"), (1.25, "-"), (.667, (0, (3, 2))), (1.50, (0, (3, 2)))]:
        ax.axvline(v, color=GREY, lw=.7, ls=ls)
    ax.axvline(g.ratio.median(), color=ORANGE, lw=1.2)
    ax.annotate(f"median {g.ratio.median():.4f}", (g.ratio.median(), ax.get_ylim()[1]*.92),
                xytext=(6, 0), textcoords="offset points", fontsize=7, color=ORANGE)
    ax.annotate("18 buildings\nexcluded", (1.33, 18), xytext=(1.40, 90), fontsize=7, color=INK,
                arrowprops=dict(arrowstyle="-", lw=.7, color=GREY))
    ax.text(.815, ax.get_ylim()[1]*.60, "primary band\n0.80-1.25", fontsize=7, color=GREY, ha="left")
    ax.text(.688, ax.get_ylim()[1]*.16, "sensitivity\nband\n0.667-1.50", fontsize=7, color=GREY, ha="left")
    ax.set_xlabel("MAE% (TFT) / MAE% (LSTM)")
    ax.set_ylabel("Buildings"); ax.set_xlim(.6, 1.6)
    tidy(ax)
    save(f, "f08_gate2_parity")

# ---------------------------------------------------------------- F09
def f09_tenseed():
    t = rd("analysis5_tenseed_predictive.csv")
    f, ax = fig("f", COL1, 1.98); rng = np.random.default_rng(3)
    for i, (a, nm, c) in enumerate([("lstm", "LSTM", BLUE), ("tft", "TFT", ORANGE)]):
        v = t[t.arch == a].mae_pct.values
        ax.scatter(np.full_like(v, i) + rng.uniform(-.10, .10, v.size), v, s=16, color=c,
                   alpha=.85, linewidths=.4, edgecolors="white", zorder=3)
        ax.plot([i - .27, i + .27], [v.mean()] * 2, color=INK, lw=1.4, zorder=4)
        ax.text(i + .31, v.mean(), f"{v.mean():.2f}%", fontsize=7, color=INK, va="center")
    ax.axhline(26, color=ORANGE, lw=1.0, ls=(0, (4, 2)))
    ax.text(1.45, 26, "Gate 1 threshold 26%", fontsize=7, color=ORANGE, ha="right", va="bottom")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["LSTM", "TFT"]); ax.set_xlim(-.55, 1.75)
    ax.set_ylabel("Anchor test MAE%"); ax.set_ylim(12, 28)
    tidy(ax)
    save(f, "f09_tenseed_predictive")

# ---------------------------------------------------------------- F10
def f10_selection():
    def runs(s): return [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", str(s).replace("np.float64", ""))]
    jul = rd("tft_top5_confirmation_log.csv", base=R)
    sep = rd("tft_top5_confirmation_RERUN_2026-09-02T064610Z.csv", base=R)
    f, ax = fig("f", COL1, 2.08)
    for d, nm, c, mk, dx in [(jul, "21 July", BLUE, "o", -.09), (sep, "2 September", ORANGE, "s", .09)]:
        x = d["rank"].values + dx
        ax.errorbar(x, d.confirmed_mean, yerr=d.confirmed_std, fmt=mk, ms=4.5, color=c,
                    lw=0, elinewidth=1.0, capsize=2.2, capthick=.8, label=nm, zorder=3)
        ax.plot(x, d.confirmed_mean, color=c, lw=.8, alpha=.5, zorder=2)
    sel = jul[jul["rank"] == 2]
    ax.annotate("configuration\ncarried forward", (2 - .09, float(sel.confirmed_mean.iloc[0])),
                xytext=(2.6, 16.6), fontsize=7, color=INK,
                arrowprops=dict(arrowstyle="-", lw=.7, color=GREY))
    ax.set_xticks([1, 2, 3, 4, 5]); ax.set_xlabel("Configuration (screening rank)")
    ax.set_ylabel("Confirmed anchor test MAE%")
    ax.legend(loc="upper left", handlelength=1.2, ncol=2, columnspacing=1.0)
    ax.set_ylim(15.8, 24.2); tidy(ax)
    save(f, "f10_selection_instability")

# ---------------------------------------------------------------- F11
def f11_negatives():
    P = rd("analysis1_perm_both_models.csv"); P = P[P.status == "ok"].set_index("building")
    g = rd("gate2_parity.csv").set_index("building")
    pop = [b for b in P.index if g.loc[b, "Type"] in TYPES5 and g.loc[b, "pass_primary"]]
    n = []
    for b in pop:
        n.append(int((collapse(P.loc[b, [f"perm_lstm_{f}" for f in SCHEMA]].to_numpy(float)) < 0).sum()))
    n = np.array(n)
    f, ax = fig("f", COL1, 1.95)
    vals, cnt = np.unique(n, return_counts=True)
    ax.bar(vals, cnt, .72, color=ORANGE, linewidth=0)
    ax.axvline(n.mean(), color=INK, lw=1.1)
    ax.text(n.mean() + .18, cnt.max() * .92, f"mean {n.mean():.2f} of 12", fontsize=7, color=INK)
    ax.set_xlabel("Collapsed features with a negative $\\Delta$MAE (of 12)")
    ax.set_ylabel("Buildings"); ax.set_xticks(range(0, 12))
    tidy(ax)
    save(f, "f11_permutation_negatives")

# ---------------------------------------------------------------- F12
def f12_crossmethod():
    F = rd("analysis4_shapig_meanbaseline_200steps_FULL.csv"); F = F[F.status == "ok"]
    rho = np.array([spearman(r[[f"shap_{l}" for l in LAB12]].values.astype(float),
                             r[[f"ig_{l}" for l in LAB12]].values.astype(float)) for _, r in F.iterrows()])
    f, ax = fig("f", COL1, 1.92)
    ax.hist(rho, bins=np.linspace(-.2, 1, 55), color=BLUE, linewidth=0)
    ax.axvline(np.nanmedian(rho), color=ORANGE, lw=1.3)
    ax.text(np.nanmedian(rho) - .03, ax.get_ylim()[1] * .9, f"median {np.nanmedian(rho):.3f}",
            fontsize=7, color=ORANGE, ha="right")
    ax.set_xlabel("Spearman correlation, SHAP vs Integrated Gradients")
    ax.set_ylabel("Buildings"); ax.set_xlim(-.2, 1)
    tidy(ax)
    save(f, "f12_cross_method_agreement")

# ---------------------------------------------------------------- F13
def f13_lag():
    d = rd("analysis14_per_building_corr.csv")
    order = ["electricity_lag1", "gas", "chilledwater", "solar", "hotwater"]
    lbl = ["Electricity\nlag-1", "Gas", "Chilled\nwater", "Solar", "Hot\nwater"]
    data = [d[d.meter == m].pearson_r.dropna().values for m in order]
    f, ax = fig("f", COL1, 2.00)
    bp = ax.boxplot(data, widths=.55, showfliers=False, patch_artist=True,
                    medianprops=dict(color=INK, lw=1.3), whiskerprops=dict(color=GREY, lw=.7),
                    capprops=dict(color=GREY, lw=.7), boxprops=dict(linewidth=0))
    for i, p in enumerate(bp["boxes"]):
        p.set_facecolor(ORANGE if i == 0 else BLUE); p.set_alpha(.85)
    for i, v in enumerate(data):
        ax.text(i + 1, np.median(v) + .05, f"{np.median(v):.2f}", ha="center", fontsize=7, color=INK)
        ax.text(i + 1, -1.02, f"$n$={len(v)}", ha="center", fontsize=7, color=GREY)
    ax.set_xticklabels(lbl); ax.set_ylabel("Pearson $r$ with electricity")
    ax.axhline(0, color=GREY, lw=.6); ax.set_ylim(-1.1, 1.12)
    tidy(ax)
    save(f, "f13_target_correlation")

# ---------------------------------------------------------------- F14
def f14_grid():
    g = rd("tft_extended_grid_log.csv", base=R)
    p = g.pivot_table(index=["d_model", "n_heads", "dropout", "lstm_units"],
                      columns="batch_size", values="val_rmse")
    d = (p[256] - p[64]).sort_values().values
    f, ax = fig("f", COL1, 2.00)
    x = np.arange(len(d))
    ax.bar(x, d, .78, color=[BLUE if v < 0 else ORANGE for v in d], linewidth=0)
    ax.axhline(0, color=GREY, lw=.8)
    ax.axhline(np.median(d), color=INK, lw=1.0, ls=(0, (4, 2)))
    ax.text(len(d) - .5, np.median(d) - .5, f"median {np.median(d):+.2f}", fontsize=7,
            color=INK, ha="right", va="top")
    nb = int((d < 0).sum())
    ax.text(1, 6.4, f"{nb} of {len(d)} pairs favour\nbatch 256 (below zero)", fontsize=7,
            color=BLUE, ha="left", va="top", linespacing=1.35)
    ax.annotate(f"worst cell at batch 256:\n{p[256].max():.2f} vs {p[64].max():.2f} at batch 64",
                (len(d) - 1, d[-1]), xytext=(len(d) - 3, d[-1] - 1.0), fontsize=7, color=INK,
                ha="right", arrowprops=dict(arrowstyle="-", lw=.7, color=GREY))
    ax.set_xlabel("Configuration pair, ordered by difference")
    ax.set_ylabel("RMSE difference,\nbatch 256 $-$ batch 64", linespacing=1.35)
    ax.set_xlim(-1, len(d)); ax.set_xticks([])
    tidy(ax)
    save(f, "f14_batch_dispersion")

# ---------------------------------------------------------------- F15
def f15_pipeline():
    f, ax = canvas(COL2, 1.72)
    def box(x, y, w, h, t, c, fs=6.8, tc="white"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.02",
                                    facecolor=c, edgecolor="none", alpha=.93))
        ax.text(x + w/2, y + h/2, t, ha="center", va="center", fontsize=fs, color=tc)
    def arr(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7,
                                     lw=.8, color=GREY))
    box(.00, .58, .17, .26, "BDGP2 hourly\nweather + meters", BLUE)
    box(.00, .16, .17, .26, "Anchor building\nBobcat_education_Alissa", BLUE)
    box(.22, .37, .16, .26, "15-feature\ntensor,\n48 h window", GREEN, 6.5)
    box(.43, .37, .155, .26, "LSTM 64/32\nTFT $d=64$,\n4 heads", GREEN, 6.5)
    box(.645, .58, .175, .26, "Zero-shot to\n1,276 buildings", ORANGE)
    box(.645, .16, .175, .26, "Permutation,\nSHAP, IG", ORANGE)
    box(.865, .37, .135, .26, "12 collapsed\nfeatures", GREY)
    for a in [(.17, .71, .215, .57), (.17, .29, .215, .43), (.38, .50, .425, .50),
              (.598, .50, .64, .68), (.598, .50, .64, .29), (.82, .71, .86, .56),
              (.82, .29, .86, .44)]:
        arr(*a)
    ax.text(.30, .685, "80/10/10\nchronological", ha="center", fontsize=7, color=GREY, linespacing=1.3)
    ax.text(.5025, .685, "Adam, MSE,\nearly stopping", ha="center", fontsize=7, color=GREY, linespacing=1.3)
    ax.set_xlim(-.02, 1.02); ax.set_ylim(.09, .91)
    save(f, "f15_pipeline")

# ---------------------------------------------------------------- F16
def f16_collapse():
    f, ax = canvas(COL1, 1.55)
    for i, dz in enumerate([0, .028, .056]):
        ax.add_patch(Rectangle((.015 + dz, .30 + dz), .16, .36, facecolor=BLUE,
                               edgecolor="white", lw=.8, alpha=.55 + .15 * i))
    ax.text(.115, .14, "3,000 $\\times$ 48 $\\times$ 15\nsamples, steps, features",
            ha="center", fontsize=7, color=GREY, linespacing=1.35)
    ax.add_patch(FancyArrowPatch((.325, .50), (.415, .50), arrowstyle="-|>", mutation_scale=7, lw=.8, color=GREY))
    ax.text(.395, .655, "mean $|\\cdot|$ over\nsamples, steps", ha="center", fontsize=7, color=INK, linespacing=1.35)
    ax.add_patch(Rectangle((.445, .31), .085, .38, facecolor=GREEN, edgecolor="white", lw=.8, alpha=.85))
    ax.text(.4875, .14, "15 model\ninputs", ha="center", fontsize=7, color=GREY, linespacing=1.35)
    ax.add_patch(FancyArrowPatch((.565, .50), (.655, .50), arrowstyle="-|>", mutation_scale=7, lw=.8, color=GREY))
    ax.text(.610, .655, "sum sin/cos\npairs", ha="center", fontsize=7, color=INK, linespacing=1.35)
    ax.add_patch(Rectangle((.685, .34), .075, .32, facecolor=ORANGE, edgecolor="white", lw=.8, alpha=.9))
    ax.text(.7225, .14, "12 collapsed\nfeatures", ha="center", fontsize=7, color=GREY, linespacing=1.35)
    ax.text(.80, .50, "Hour of day\nDay of week\nMonth\n$+$ 9 others", fontsize=7, color=INK, va="center", linespacing=1.35)
    ax.set_xlim(-.045, 1.0); ax.set_ylim(.04, .86)
    save(f, "f16_tensor_collapse")

for fn in [f01_instability, f02_ladder, f03_degeneracy, f04_eta2, f05_attrition, f06_search,
           f07_zerofill, f08_gate2, f09_tenseed, f10_selection, f11_negatives, f12_crossmethod,
           f13_lag, f14_grid, f15_pipeline, f16_collapse]:
    guard(fn)

with open(os.path.join(OUT, "PROVENANCE.txt"), "w") as fh:
    fh.write("FIGURES -- generated from archived artefacts, no new model runs\n")
    fh.write("run (UTC) : %s\n" % datetime.datetime.now(datetime.timezone.utc).isoformat())
    fh.write("script    : make_figures.py  sha256 %s\n" %
             hashlib.sha256(open(__file__, "rb").read()).hexdigest())
    fh.write("python %s  numpy %s  pandas %s  matplotlib %s\n" %
             (sys.version.split()[0], np.__version__, pd.__version__, matplotlib.__version__))
    fh.write("figures written: %d\n\ninputs read:\n" % len(MADE))
    for k in sorted(INPUTS): fh.write("  %-58s sha256 %s\n" % (k, INPUTS[k]))
    fh.write("\nfigures:\n")
    for m in MADE: fh.write("  %s.pdf\n" % m)
    if FAILED:
        fh.write("\nFAILED:\n")
        for n, e in FAILED: fh.write("  %s  %s\n" % (n, e))
print("\n%d made, %d failed" % (len(MADE), len(FAILED)))
