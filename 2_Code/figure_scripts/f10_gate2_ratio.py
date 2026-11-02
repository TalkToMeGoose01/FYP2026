#!/usr/bin/env python3
"""f10: Gate 2 parity ratio MAE%_TFT / MAE%_LSTM across the portfolio.
Reads results/gate2_parity.csv only. Usage: python3 f10_gate2_ratio.py <Phase2 dir>"""
import sys, pandas as pd, numpy as np
import figstyle as fs
import matplotlib.pyplot as plt

P2 = sys.argv[1] if len(sys.argv) > 1 else "."
g = pd.read_csv(f"{P2}/results/gate2_parity.csv")
g = g[g.status == "ok"]
r = g.ratio.values
n_hi, n_lo = int((r > 1.25).sum()), int((r < 0.80).sum())
assert len(g) == 1294 and int(g.pass_primary.sum()) == 1276 and n_hi == 18 and n_lo == 0
anchor = float(g.loc[g.building == "Bobcat_education_Alissa", "ratio"].iloc[0])

fs.setup()
fig, ax = plt.subplots(figsize=(fs.COL_W, 2.0))
ax.axvspan(0.667, 1.50, color=fs.LIGHT, alpha=0.35, lw=0, label="Sensitivity band 0.667 to 1.50")
ax.axvspan(0.80, 1.25, color=fs.LIGHT, alpha=1.0, lw=0, label="Primary band 0.80 to 1.25")
bins = np.exp(np.linspace(np.log(0.6), np.log(1.6), 61))
ax.hist(r, bins=bins, color=fs.BLUE, zorder=3)
ax.set_xscale("log")
ticks = [0.667, 0.8, 1.0, 1.25, 1.5]
ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
ax.minorticks_off()
ax.axvline(anchor, color=fs.ORANGE, lw=1, zorder=4)
ax.text(anchor, ax.get_ylim()[1] * 0.92, f" anchor\n {anchor:.3f}", color=fs.ORANGE,
        fontsize=6.5, va="top", ha="left")
ax.text(0.62, ax.get_ylim()[1] * 0.55, f"below 0.80: {n_lo}\nabove 1.25: {n_hi}",
        fontsize=6.5, color=fs.GREY, va="top")
ax.set_xlabel("MAE%$_{\\mathrm{TFT}}$ / MAE%$_{\\mathrm{LSTM}}$ (log scale)")
ax.set_ylabel("Buildings")
ax.legend(loc="upper left", handlelength=1.2)
fig.tight_layout()
fs.save(fig, P2, "f10_gate2_ratio")
