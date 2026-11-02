#!/usr/bin/env python3
"""
a21_ceiling_across_types.py
===========================
ANALYSIS 21 -- ESTIMATION CEILINGS AND INSTABILITY OUTSIDE EDUCATION

WHY THIS EXISTS
---------------
Every estimation ceiling and every instability figure in
Table~\\ref{tab:methodeffect} comes from the 25 calibration buildings of
analysis3_ig_ceiling.csv, all of which are education buildings. This
analysis repeats the same measurement on buildings of the four other
types with at least 40 buildings in the 319-building variance subsample,
to test whether the method ordering and its size hold outside education.

WHAT IT RUNS
------------
Nothing new is implemented. The script imports a12_zerofill_audit.py
and calls its stage 3 (raw attribution vectors: SHAP, Integrated
Gradients and permutation importance on the three archived LSTM draws,
plus the ceiling repeats) and stage 4 (ceiling, across-draw agreement
and instability under the full12, masked and uniform8 conventions) on a
different building list. The attribution arithmetic is the code a12
splices from FYP_Phase2_TFT_FINAL.py, so every convention matches
Table~\\ref{tab:methodeffect} exactly.

BUILDINGS
---------
  * 5 per type for Office, Public services, Entertainment/public
    assembly and Lodging/residential, drawn from the 319-building
    variance subsample with numpy default_rng(2016), sorted list,
    no replacement. 20 buildings.
  * 1 replication building, the first of the original 25
    (Rat_education_Barbara), whose vectors must reproduce
    analysis12_attribution_vectors.csv. The run says so if they do not.

OUTPUTS (results/)
------------------
  analysis21_attribution_vectors.csv   raw collapsed vectors
  analysis21_agreement.csv             per building, method, convention
  analysis21_summary.txt               by type, against the 25 education
  analysis21_PROVENANCE.txt            inputs, digests, building list
  analysis21_work/                     a12's working files (kept)

Resumable: re-running skips (building, method, model, replicate) rows
already written. Reads only; no model is trained or re-saved.

USAGE (WSL, fyp_env, GPU), run from the Phase2 folder
    python a21_ceiling_across_types.py --dry-run
    python a21_ceiling_across_types.py
"""
import argparse, datetime, hashlib, importlib.util, platform, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu

TYPES = ["Office", "Public services", "Entertainment/public assembly",
         "Lodging/residential"]
PER_TYPE, SEED = 5, 2016
REPLICATION = "Rat_education_Barbara"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def select_buildings(results):
    v = pd.read_csv(results / "analysis1b_variance_subsample.csv")
    v = v[v.status == "ok"][["building"]]
    g = pd.read_csv(results / "gate2_parity.csv")[["building", "Type"]]
    v = v.merge(g, on="building", how="left")
    rng = np.random.default_rng(SEED)
    picked, types = [], {}
    for t in TYPES:
        pool = sorted(v.loc[v.Type == t, "building"].tolist())
        sel = sorted(rng.choice(pool, PER_TYPE, replace=False).tolist())
        picked += sel
        types.update({b: t for b in sel})
    return picked, types


def summarise(phase2, results, work, types, out_txt):
    A = pd.read_csv(work / "analysis12_agreement.csv")
    E = pd.read_csv(results / "analysis12_agreement.csv")       # the 25 education
    A["group"] = A.building.map(lambda b: types.get(b, "replication"))
    E["group"] = "Education (calibration 25)"
    L = []
    def p(s=""):
        print(s); L.append(s)
    p("=" * 76)
    p("ANALYSIS 21 -- ceilings and instability outside education (LSTM, masked)")
    p("=" * 76)

    # replication check
    rep_new = A[A.building == REPLICATION]
    rep_old = E[E.building == REPLICATION]
    if len(rep_new):
        j = rep_new.merge(rep_old, on=["method", "convention"], suffixes=("_new", "_old"))
        d = max((j.rho_ceiling_new - j.rho_ceiling_old).abs().max(),
                (j.rho_across_draw_new - j.rho_across_draw_old).abs().max())
        p(f"replication on {REPLICATION}: max |difference| against analysis12 = {d:.4f}"
          + ("   OK" if d < 1e-6 else "   ** DOES NOT REPRODUCE -- resolve before reading on **"))
        p("")

    new = A[A.group != "replication"]
    both = pd.concat([E, new], ignore_index=True)
    m = both[both.convention == "masked"]
    p(f"{'group':<32}{'method':<7}{'n':>3}{'ceiling':>9}{'across':>9}{'instab':>9}{'IQR':>16}")
    for grp in ["Education (calibration 25)"] + TYPES + ["All four other types"]:
        sub = m[m.group.isin(TYPES)] if grp == "All four other types" else m[m.group == grp]
        for meth in ["perm", "ig", "shap"]:
            s = sub[sub.method == meth]
            if s.empty:
                continue
            q1, q3 = s.instability.quantile(.25), s.instability.quantile(.75)
            p(f"{grp[:31]:<32}{meth:<7}{s.building.nunique():>3}{s.rho_ceiling.median():>9.4f}"
              f"{s.rho_across_draw.median():>9.4f}{s.instability.median():>9.4f}"
              f"{q1:>8.3f}-{q3:<7.3f}")
        p("")

    def ratios(sub, label):
        w = sub.pivot_table(index="building", columns="method", values="instability")
        r1 = w.perm.median() / w.ig.median(); r2 = w.perm.median() / w.shap.median()
        f1 = (w.perm > w.ig).mean(); f2 = (w.perm > w.shap).mean()
        try:
            p1 = wilcoxon(w.perm, w.ig).pvalue; p2 = wilcoxon(w.perm, w.shap).pvalue
        except ValueError:
            p1 = p2 = float("nan")
        p(f"{label}: permutation {r1:.1f}x IG and {r2:.1f}x SHAP; less reproducible than IG in "
          f"{100*f1:.0f}% (p={p1:.3g}), than SHAP in {100*f2:.0f}% (p={p2:.3g})")
    ratios(m[m.group == "Education (calibration 25)"], "Education 25   ")
    ratios(m[m.group.isin(TYPES)], "Other types 20 ")
    p("")
    p("Instability, other types against education (Mann-Whitney, two-sided):")
    for meth in ["perm", "ig", "shap"]:
        a = m[(m.group.isin(TYPES)) & (m.method == meth)].instability.dropna()
        b = m[(m.group == "Education (calibration 25)") & (m.method == meth)].instability.dropna()
        p(f"  {meth:<5} median {a.median():.4f} vs {b.median():.4f}, p = {mannwhitneyu(a, b).pvalue:.3g}")
    p("")
    p("Reference, Table tab:methodeffect (masked, n=25): perm 0.9762/0.6833/0.2111, "
      "IG 1.0000/0.9524/0.0444, SHAP 1.0000/0.9603/0.0397")
    Path(out_txt).write_text("\n".join(L) + "\n", encoding="utf-8")


def main(a):
    phase2 = Path(a.phase2).resolve()
    results = phase2 / "results"
    work = results / "analysis21_work"
    work.mkdir(parents=True, exist_ok=True)
    shutil.copy2(results / "analysis12_feature_mask.csv", work / "analysis12_feature_mask.csv")

    spec = importlib.util.spec_from_file_location("a12", phase2 / "a12_zerofill_audit.py")
    a12 = importlib.util.module_from_spec(spec)
    sys.argv = [sys.argv[0]]
    spec.loader.exec_module(a12)

    buildings, types = select_buildings(results)
    run_list = [REPLICATION] + buildings
    args = argparse.Namespace(
        stage="3", pipeline_dir=str(phase2), results_dir=str(results),
        out_dir=str(work), draws_dir=str(phase2 / "archive" / "2026-07-21_analysis1"),
        final_py=str(phase2 / "FYP_Phase2_TFT_FINAL.py"), data_dir=a.data,
        cleaned=False, dry_run=a.dry_run)
    ctx = a12.build_context(args)
    ctx["ceiling_buildings"] = run_list
    print("buildings:")
    for b in run_list:
        print(f"  {b:45s} {types.get(b, 'replication (education)')}")

    with open(results / "analysis21_PROVENANCE.txt", "w", encoding="utf-8") as fh:
        fh.write("ANALYSIS 21 -- ceilings and instability outside education\n")
        fh.write(f"run (UTC) : {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n")
        fh.write(f"host      : {platform.node()} {platform.platform()} python {platform.python_version()}\n")
        for f in ["a21_ceiling_across_types.py", "a12_zerofill_audit.py",
                  "FYP_Phase2_TFT_FINAL.py", "fyp_pipeline.py"]:
            fh.write(f"{f:32s} sha256 {sha256(phase2 / f)}\n")
        for k, pth in ctx["draw_paths"].items():
            fh.write(f"{k:32s} sha256 {sha256(pth)}\n")
        fh.write(f"selection : {PER_TYPE} per type, default_rng({SEED}), from analysis1b_variance_subsample.csv\n")
        for b in run_list:
            fh.write(f"  {b} | {types.get(b, 'replication')}\n")

    a12.stage3_vectors(ctx)
    if a.dry_run:
        print("\nDRY RUN complete (replication building only). Compare its rows in "
              "results/analysis21_work/analysis12_attribution_vectors.csv against "
              "results/analysis12_attribution_vectors.csv.")
        return 0
    a12.stage4_agree(ctx)
    shutil.copy2(work / "analysis12_attribution_vectors.csv", results / "analysis21_attribution_vectors.csv")
    shutil.copy2(work / "analysis12_agreement.csv", results / "analysis21_agreement.csv")
    summarise(phase2, results, work, types, results / "analysis21_summary.txt")
    print("\nwritten: results/analysis21_attribution_vectors.csv, analysis21_agreement.csv, "
          "analysis21_summary.txt, analysis21_PROVENANCE.txt")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase2", default=".")
    ap.add_argument("--data", default=None, help="BDGP2 Data folder (auto-found if omitted)")
    ap.add_argument("--dry-run", action="store_true")
    sys.exit(main(ap.parse_args()))
