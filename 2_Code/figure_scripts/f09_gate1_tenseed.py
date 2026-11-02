#!/usr/bin/env python3
"""f09: Gate 1 at the anchor, ten seeds per architecture.
Reads results/analysis5_tenseed_predictive.csv only (test-partition MAE%).
Usage: python3 f09_gate1_tenseed.py <Phase2 dir>"""
import sys, pandas as pd, numpy as np
import figstyle as fs
import matplotlib.pyplot as plt

P2 = sys.argv[1] if len(sys.argv) > 1 else "."
a = pd.read_csv(f"{P2}/results/analysis5_tenseed_predictive.csv")
a = a[a.status == "ok"]
st = a.groupby("arch").mae_pct.agg(["mean", "std", "count"])
assert (st["count"] == 10).all()
assert round(st.loc["lstm", "mean"], 2) == 16.85 and round(st.loc["tft", "mean"], 2) == 18.42
THRESH = 1.5 * 17.29                     # Gate 1: 1.5 x anchor LSTM, Appendix impl

fs.setup()
fig, ax = plt.subplots(figsize=(fs.COL_W, 1.6))
for y, arch, col, lab in [(1, "lstm", fs.BLUE, "LSTM"), (0, "tft", fs.ORANGE, "TFT")]:
    v = a[a.arch == arch].mae_pct.values
    ax.scatter(v, np.full(len(v), y), s=16, color=col, alpha=0.8, zorder=3, linewidths=0)
    m, s = st.loc[arch, "mean"], st.loc[arch, "std"]
    ax.errorbar(m, y + 0.28, xerr=s, fmt="|", color=col, capsize=2, lw=1, ms=8)
    ax.text(m, y + 0.42, f"{m:.2f} $\\pm$ {s:.2f}", ha="center", va="bottom", fontsize=6.5, color=col)
ax.axvline(THRESH, color=fs.GREY, ls="--", lw=0.8)
ax.text(THRESH - 0.15, 0.5, "Gate 1\nthreshold\n26%", ha="right", va="center", fontsize=6.5, color=fs.GREY)
ax.set_yticks([0, 1], ["TFT", "LSTM"])
ax.set_ylim(-0.4, 1.75)
ax.set_xlim(13, 27)
ax.set_xlabel("Anchor test MAE% (test-partition mean)")
fig.tight_layout()
fs.save(fig, P2, "f09_gate1_tenseed")
