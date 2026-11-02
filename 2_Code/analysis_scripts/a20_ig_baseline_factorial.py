#!/usr/bin/env python3
"""
a20_ig_baseline_factorial.py
============================
ANALYSIS 20 -- INTEGRATED GRADIENTS: BASELINE x STEP-COUNT FACTORIAL

WHY THIS EXISTS
---------------
sec:baseline reports that moving IG from the training-mean baseline to a
zero baseline leaves the estimation ceiling unmoved (0.9930 vs 0.9929)
and raises across-draw agreement (0.9441 -> 0.9669). The two archived
cells behind that comparison differ in TWO settings at once:

    analysis3_ig_*.csv               mean baseline, 200 steps
    analysis3_ig_*_ZEROBASELINE.csv  zero baseline,  50 steps
                                     (archive FYP_Phase2_TFT_FINAL.py In[63]
                                      sets IG_STEPS = 50)

so the effect cannot be assigned to the baseline. This script fills the
two missing cells of the 2x2 design on the same 25 buildings, the same
frozen LSTM draws and the same subsample convention:

    zero baseline, 200 steps    (isolates the baseline at 200 steps)
    mean baseline,  50 steps    (isolates the step count at the mean baseline)

It also records the IG completeness error for every cell it runs, so
the completeness figures stop depending on notebook comments.

Optional --replicate re-runs the two archived cells as well. Their
per-building rho values are compared against the archived artefacts,
which checks that this environment reproduces the archived numbers.

CONVENTIONS (copied from archive/2026-07-24_analysis4/FYP_Phase2_TFT_FINAL.py,
cells In[63] and In[50]; do not change)
    IG_SAMPLES  200 sequences, drawn by default_rng(stable_seed(b) + offset)
                (offset 0 for every draw, offset 1 for the ceiling's second run)
    IG_BATCH    25
    path        Riemann right-endpoint, alpha = k/steps, k = 1..steps
    reduce      mean |IG| over samples then timesteps -> 15 values
    collapse    sum sin/cos pairs -> 12 values (schema-derived)
    ceiling     Spearman(lstm_draw1 offset 0, lstm_draw1 offset 1)
    across-draw mean Spearman over the three draw pairs, offset 0
    sub_rho     the same with Hour of Day removed
    baselines   zero  = np.zeros((1,48,15))
                mean  = anchor X_train.mean(axis=0, keepdims=True)
    completeness (per building, lstm_draw1, offset-0 subsample)
                lhs = sum over (timesteps, features) of signed IG
                rhs = F(x) - F(baseline)
                rel_err = mean|lhs - rhs| / mean|rhs|
                (same definition as archive In[47])
    instability ceiling median - across-draw median (as in sec:baseline)

OUTPUTS
    results/analysis20_ig_factorial.csv            one row per building x cell
    results/analysis20_ig_factorial_PROVENANCE.txt input digests
    results/analysis20_summary.txt                 the summary printed at the end

Resumable: (building, cell) pairs already in the output CSV are skipped.
Reads only. No model is trained or re-saved. No archived file is written.

USAGE (WSL, fyp_env, GPU)
    python a20_ig_baseline_factorial.py --phase2 /home/pie/fyp/phase2 \
        --data "/home/pie/fyp/ZEIT4500 - FYP/Data" --dry-run
    python a20_ig_baseline_factorial.py --phase2 /home/pie/fyp/phase2 \
        --data "/home/pie/fyp/ZEIT4500 - FYP/Data"
    add --replicate to also re-run mean/200 and zero/50
"""
import argparse, datetime, gc, hashlib, os, platform, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

SEED = 2016
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
import tensorflow as tf                                      # noqa: E402
for _g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_g, True)
from tensorflow.keras.models import load_model               # noqa: E402

ANCHOR, IG_SAMPLES, IG_BATCH, N_TEST = "Bobcat_education_Alissa", 200, 25, 25
PAIR_MAP = {"Hour of Day": ("hour_sin", "hour_cos"),
            "Day of Week": ("dow_sin", "dow_cos"),
            "Month": ("month_sin", "month_cos")}
METER_NAMES = ["chilledwater", "gas", "hotwater", "solar", "steam", "water", "irrigation"]
NEW_CELLS = [("zero", 200), ("mean", 50)]
ARCHIVED_CELLS = {("mean", 200): ("analysis3_ig_ceiling.csv", "analysis3_ig_draws_lstm.csv"),
                  ("zero", 50): ("analysis3_ig_ceiling_ZEROBASELINE.csv",
                                 "analysis3_ig_draws_lstm_ZEROBASELINE.csv")}


def stable_seed(building, base=SEED):
    h = int(hashlib.md5(building.encode()).hexdigest()[:8], 16)
    return base + (h % 100000)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_collapse_index(schema):
    paired, labels, ops = set(), [], []
    for label, (s, c) in PAIR_MAP.items():
        if s in schema and c in schema:
            labels.append(label); ops.append(("sum", schema.index(s), schema.index(c)))
            paired.update([s, c])
    for i, f in enumerate(schema):
        if f not in paired:
            labels.append(f); ops.append(("keep", i))

    def collapse_fn(arr):
        return np.array([arr[o[1]] + arr[o[2]] if o[0] == "sum" else arr[o[1]] for o in ops])
    return labels, collapse_fn


def integrated_gradients(model, X, baseline, steps, completeness=False):
    """Per-feature mean |IG| over (samples, timesteps), exactly as In[50]/In[63].
    With completeness=True also returns (mean|lhs-rhs|, mean|rhs|)."""
    X = np.asarray(X, dtype=np.float32)
    baseline = np.asarray(baseline, dtype=np.float32)
    total = np.zeros(X.shape[1:], dtype=np.float64)
    n, err_abs, rhs_abs = 0, [], []
    for s in range(0, len(X), IG_BATCH):
        xb = X[s:s + IG_BATCH]
        bl = np.repeat(baseline, len(xb), axis=0)
        diff = xb - bl
        acc = np.zeros_like(xb, dtype=np.float64)
        for k in range(1, steps + 1):
            pt = tf.convert_to_tensor(bl + (k / steps) * diff)
            with tf.GradientTape() as tape:
                tape.watch(pt)
                out = model(pt, training=False)
            acc += tape.gradient(out, pt).numpy()
        ig = (acc / steps) * diff
        total += np.abs(ig).sum(axis=0)
        if completeness:
            lhs = ig.sum(axis=(1, 2))
            rhs = (model(xb, training=False).numpy().ravel()
                   - model(bl, training=False).numpy().ravel())
            err_abs.extend(np.abs(lhs - rhs)); rhs_abs.extend(np.abs(rhs))
        n += len(xb)
        del acc, ig; gc.collect()
    per_feature = (total / n).mean(axis=0)
    if completeness:
        return per_feature, float(np.mean(err_abs)), float(np.mean(rhs_abs))
    return per_feature


def subsample(X_al, building, offset):
    if len(X_al) > IG_SAMPLES:
        rng = np.random.default_rng(stable_seed(building) + offset)
        idx = np.sort(rng.choice(len(X_al), IG_SAMPLES, replace=False))
        return X_al[idx]
    return X_al


def summarise(df, out_txt):
    lines = []
    def p(s=""):
        print(s); lines.append(s)
    ok = df[df.status == "ok"]
    p("=" * 72)
    p("ANALYSIS 20 -- IG baseline x steps, LSTM, 25 buildings")
    p("=" * 72)
    p(f"{'cell':<14}{'source':<10}{'n':>4}{'ceiling':>10}{'across':>10}{'instab.':>10}{'complete':>11}")
    cells = {}
    for (bl, st), g in ok.groupby(["baseline", "steps"]):
        cells[(bl, st)] = g.set_index("building")
        ce = g.ceiling_rho.median(); ac = g.mean_rho.median()
        comp = g.completeness_rel_err.median() if "completeness_rel_err" in g else np.nan
        p(f"{bl + '/' + str(st):<14}{g.source.iloc[0]:<10}{len(g):>4}{ce:>10.4f}{ac:>10.4f}{ce - ac:>10.4f}{comp:>11.4f}")
    p("")
    p("paired comparisons of per-building across-draw rho (Wilcoxon, two-sided)")
    for a, b, what in [(("zero", 200), ("mean", 200), "baseline at 200 steps"),
                       (("zero", 50), ("mean", 50), "baseline at 50 steps"),
                       (("mean", 50), ("mean", 200), "steps at mean baseline"),
                       (("zero", 50), ("zero", 200), "steps at zero baseline")]:
        if a in cells and b in cells:
            j = cells[a][["mean_rho"]].join(cells[b][["mean_rho"]], lsuffix="_a", rsuffix="_b").dropna()
            d = j.mean_rho_a - j.mean_rho_b
            try:
                pv = wilcoxon(j.mean_rho_a, j.mean_rho_b).pvalue
            except ValueError:
                pv = np.nan
            p(f"  {what:<26} {a[0]}/{a[1]} vs {b[0]}/{b[1]}: median diff {d.median():+.4f}, "
              f"first higher in {(d > 0).sum()}/{len(d)}, p = {pv:.4g}")
    Path(out_txt).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(a):
    phase2 = Path(a.phase2); data = Path(a.data)
    sys.path.insert(0, str(phase2))
    from fyp_pipeline import prepare_building, align_to_schema, split_sequences
    res = phase2 / "results"
    out = res / "analysis20_ig_factorial.csv"
    prov = res / "analysis20_ig_factorial_PROVENANCE.txt"

    models, digests = {}, {}
    for d in (1, 2, 3):
        mp = phase2 / "draws" / f"lstm_draw{d}.keras"
        ap_ = phase2 / "archive" / "2026-07-21_analysis1" / f"lstm_draw{d}.keras"
        digests[d] = sha256(mp)
        if ap_.exists() and sha256(ap_) != digests[d]:
            print(f"!! lstm_draw{d}.keras does not match its archived copy. Stop.")
            return 2
        models[d] = load_model(mp, compile=False)

    electricity = pd.read_csv(data / "electricity.csv", parse_dates=["timestamp"])
    weather = pd.read_csv(data / "weather.csv", parse_dates=["timestamp"])
    meter_dfs = {m: pd.read_csv(data / f"{m}.csv", parse_dates=["timestamp"])
                 for m in METER_NAMES if (data / f"{m}.csv").is_file()}
    X_seq, y_seq, _, schema, _ = prepare_building(ANCHOR, electricity, weather, meter_dfs)
    X_tr = split_sequences(X_seq, y_seq)[0]
    baselines = {"zero": np.zeros((1,) + X_tr.shape[1:], dtype=np.float32),
                 "mean": X_tr.mean(axis=0, keepdims=True).astype(np.float32)}
    labels, collapse_fn = build_collapse_index(schema)
    hod = labels.index("Hour of Day")

    src = pd.read_csv(res / "analysis3_ig_ceiling.csv")
    buildings = src["building"].tolist()[:N_TEST]
    cells = NEW_CELLS + (list(ARCHIVED_CELLS) if a.replicate else [])
    if a.dry_run:
        buildings = buildings[:1]

    with open(prov, "w", encoding="utf-8") as fh:
        fh.write("ANALYSIS 20 -- IG baseline x steps factorial\n")
        fh.write(f"run (UTC)   : {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n")
        fh.write(f"script      : {Path(__file__).name}  sha256 {sha256(__file__)}\n")
        fh.write(f"host        : {platform.node()}  {platform.platform()}  python {platform.python_version()}  tf {tf.__version__}\n")
        fh.write(f"gpu         : {[g.name for g in tf.config.list_physical_devices('GPU')]}\n")
        for d in (1, 2, 3):
            fh.write(f"lstm_draw{d}  : sha256 {digests[d]}\n")
        fh.write(f"pipeline    : fyp_pipeline.py  sha256 {sha256(phase2 / 'fyp_pipeline.py')}\n")
        for f in ["electricity.csv", "weather.csv"] + [f"{m}.csv" for m in METER_NAMES]:
            if (data / f).is_file():
                fh.write(f"  {f:18s} sha256 {sha256(data / f)}\n")
        fh.write(f"buildings   : {len(buildings)} from analysis3_ig_ceiling.csv\n")
        fh.write(f"cells       : {cells}\n")
        fh.write(f"schema      : {schema}\ncollapsed   : {labels}\n")

    done = set()
    if out.exists() and not a.dry_run:
        prev = pd.read_csv(out)
        done = set(zip(prev.building, prev.baseline, prev.steps))
    t_start = time.time()
    for i, b in enumerate(buildings):
        X_b, _, _, cols_b, _ = prepare_building(b, electricity, weather, meter_dfs)
        X_al = align_to_schema(X_b, cols_b, schema)
        X0, X1 = subsample(X_al, b, 0), subsample(X_al, b, 1)
        for bl, st in cells:
            if (b, bl, st) in done:
                continue
            t0 = time.time()
            row = {"building": b, "baseline": bl, "steps": st, "source": "a20"}
            try:
                raw1, e_abs, r_abs = integrated_gradients(models[1], X0, baselines[bl], st, True)
                v = {1: collapse_fn(raw1)}
                v1b = collapse_fn(integrated_gradients(models[1], X1, baselines[bl], st))
                for d in (2, 3):
                    v[d] = collapse_fn(integrated_gradients(models[d], X0, baselines[bl], st))
                pairs = [(1, 2), (1, 3), (2, 3)]
                m = np.ones(len(labels), bool); m[hod] = False
                row.update({
                    "ceiling_rho": spearmanr(v[1], v1b)[0],
                    "mean_rho": float(np.nanmean([spearmanr(v[x], v[y])[0] for x, y in pairs])),
                    "sub_rho": float(np.nanmean([spearmanr(v[x][m], v[y][m])[0] for x, y in pairs])),
                    "top1_draw1": labels[int(np.argmax(v[1]))],
                    "top1_draw2": labels[int(np.argmax(v[2]))],
                    "top1_draw3": labels[int(np.argmax(v[3]))],
                    "completeness_abs_err": e_abs, "completeness_mean_abs_rhs": r_abs,
                    "completeness_rel_err": e_abs / r_abs if r_abs > 0 else np.nan,
                    "seconds": round(time.time() - t0, 1), "status": "ok"})
                if (bl, st) in ARCHIVED_CELLS:
                    ce_f, dr_f = ARCHIVED_CELLS[(bl, st)]
                    ce = pd.read_csv(res / ce_f).set_index("building")
                    dr = pd.read_csv(res / dr_f).set_index("building")
                    row["archived_ceiling_rho"] = ce.loc[b, "rho"]
                    row["archived_mean_rho"] = dr.loc[b, "mean_rho"]
            except Exception as e:
                row["status"] = f"{type(e).__name__}: {e}"
            df_new = pd.DataFrame([row])
            if out.exists() and not a.dry_run:
                df_new = pd.concat([pd.read_csv(out), df_new], ignore_index=True)
            df_new.drop_duplicates(subset=["building", "baseline", "steps"], keep="last").to_csv(
                out if not a.dry_run else res / "analysis20_DRYRUN.csv", index=False)
            print(f"[{i + 1}/{len(buildings)}] {b} {bl}/{st}  {row['status']}  "
                  f"ceil {row.get('ceiling_rho', np.nan):.4f}  across {row.get('mean_rho', np.nan):.4f}  "
                  f"compl {row.get('completeness_rel_err', np.nan):.4f}  {row.get('seconds', 0)}s  "
                  f"elapsed {(time.time() - t_start) / 60:.1f} min", flush=True)
        del X_b, X_al, X0, X1; gc.collect()

    if a.dry_run:
        print("\nDRY RUN complete: one building, output in results/analysis20_DRYRUN.csv")
        return 0
    df = pd.read_csv(out)
    for (bl, st), (ce_f, dr_f) in ARCHIVED_CELLS.items():
        if not ((df.baseline == bl) & (df.steps == st)).any():
            ce = pd.read_csv(res / ce_f); dr = pd.read_csv(res / dr_f)
            arc = ce[["building", "rho"]].rename(columns={"rho": "ceiling_rho"}).merge(
                dr[["building", "mean_rho", "sub_rho"]], on="building")
            arc["baseline"], arc["steps"], arc["source"], arc["status"] = bl, st, "archived", "ok"
            df = pd.concat([df, arc], ignore_index=True)
    if a.replicate:
        rep = df[df.get("archived_mean_rho").notna()] if "archived_mean_rho" in df else df.iloc[0:0]
        if len(rep):
            print("\nREPLICATION against archived artefacts (max |difference| over buildings)")
            for (bl, st), g in rep.groupby(["baseline", "steps"]):
                print(f"  {bl}/{st}: ceiling {np.abs(g.ceiling_rho - g.archived_ceiling_rho).max():.4f}  "
                      f"across {np.abs(g.mean_rho - g.archived_mean_rho).max():.4f}")
    summarise(df, res / "analysis20_summary.txt")
    print(f"\nwrote {out}\n      {res / 'analysis20_summary.txt'}\n      {prov}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase2", default="/home/pie/fyp/phase2")
    ap.add_argument("--data", default="/home/pie/fyp/ZEIT4500 - FYP/Data")
    ap.add_argument("--replicate", action="store_true",
                    help="also re-run the two archived cells and compare")
    ap.add_argument("--dry-run", action="store_true")
    sys.exit(main(ap.parse_args()))
