#!/usr/bin/env python3
"""Analysis 16b: Shapiro-Wilk normality tests on the two gas-target 10-run
samples of Analysis 19 (Section IV-F), which the thesis reports as
mean +/- standard deviation.

Analysis 16 (a16_normality.py, 10 Sep 2026) tested the four electricity
samples (Gate 1 LSTM and TFT, TFT rank 2 and rank 5). The gas-target models
were trained later, on 13 Oct 2026, so Analysis 16 could not include them.
This script applies the same test to those two samples, in the same way:
scipy.stats.shapiro on mae_pct, rows with status=ok, sample SD with ddof=1.

No model is loaded and nothing is trained; the input is an existing result
file. Usage (from the working folder of the README):
  python3 a16b_normality_gas.py [results_dir]        default: results
"""
import sys, hashlib, datetime, platform, pathlib
import numpy as np, pandas as pd, scipy
from scipy import stats

R = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "results")
SRC = R / "analysis19_gas_predictive.csv"
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

df = pd.read_csv(SRC)
if (df["status"] != "ok").any():
    sys.exit("analysis19_gas_predictive.csv contains runs whose status is not ok")

rows = []
for arch in ("lstm", "tft"):
    v = df.loc[(df.target == "gas") & (df.arch == arch), "mae_pct"].to_numpy(float)
    if len(v) != 10:
        sys.exit(f"gas {arch}: expected 10 runs, found {len(v)}")
    w, p = stats.shapiro(v)
    q1, q3 = np.percentile(v, [25, 75])
    rows.append({"source_file": "analysis19_gas_predictive", "sample": f"gas_{arch}",
                 "metric": "mae_pct", "n": len(v), "mean": f"{v.mean():.6f}",
                 "sd_ddof1": f"{v.std(ddof=1):.6f}", "median": f"{np.median(v):.6f}",
                 "iqr": f"{q3 - q1:.6f}", "shapiro_W": f"{w:.6f}", "shapiro_p": f"{p:.6f}",
                 "normality_rejected_at_0.05": "yes" if p < 0.05 else "no"})
out = pd.DataFrame(rows)
out_path = R / "analysis16b_normality.csv"
out.to_csv(out_path, index=False)
out["shapiro_p"] = out["shapiro_p"].astype(float)

log = [
    "analysis16b_normality.csv",
    "=" * 60,
    "",
    "PURPOSE",
    "  Extends analysis16_normality (four electricity samples, 10 Sep 2026)",
    "  to the two gas-target 10-run samples of Analysis 19, trained on",
    "  13 Oct 2026 and reported in Section IV-F as mean +/- SD.",
    "",
    "METHOD",
    "  scipy.stats.shapiro on the mae_pct column, rows with status=ok.",
    "  Standard deviation is the sample SD (ddof=1). Same method and output",
    "  columns as analysis16_normality.",
    "",
    "INPUTS (SHA-256)",
    f"  {sha(SRC)}  {SRC.name}",
    "",
    "OUTPUT (SHA-256)",
    f"  {sha(out_path)}  {out_path.name}",
    "",
    "RESULT",
] + [f"  {r['sample']:<9} mean {float(r['mean']):.2f}  sd {float(r['sd_ddof1']):.2f}  "
     f"W {float(r['shapiro_W']):.4f}  p {float(r['shapiro_p']):.4f}" for r in rows] + [
    f"  Normality rejected at alpha = 0.05 in any sample: "
    f"{'YES' if (out.shapiro_p < 0.05).any() else 'NO'}",
    f"  Minimum p across these two samples: {out.shapiro_p.min():.4f}",
    "  Minimum p across analysis16 and analysis16b together: "
    f"{min(out.shapiro_p.min(), 0.2355):.4f} (analysis16 minimum 0.2355)",
    "",
    "ENVIRONMENT",
    f"  Python {platform.python_version()}",
    f"  numpy {np.__version__}  pandas {pd.__version__}  scipy {scipy.__version__}",
    f"  Run (UTC): {datetime.datetime.now(datetime.timezone.utc).isoformat()}",
    f"  script a16b_normality_gas.py sha256 {sha(__file__)}",
    "  Run outside the Conda environment of Appendix A. Shapiro-Wilk is",
    "  deterministic, so the machine does not affect W or p.",
    "  This analysis loads no model and trains nothing.",
]
(R / "analysis16b_PROVENANCE.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
print("\n".join(log))
