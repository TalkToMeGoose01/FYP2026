#!/usr/bin/env python3
"""f02: Integrated Gradients baseline x step count (Analysis 20), LSTM, 25 buildings.
(a) across-draw agreement per building in each cell; (b) completeness error.
Reads results/analysis20_ig_factorial.csv only. Usage: python3 f02_ig_baseline_steps.py <Phase2 dir>"""
import sys, pandas as pd, numpy as np
import figstyle as fs
import matplotlib.pyplot as plt

P2 = sys.argv[1] if len(sys.argv) > 1 else "."
d = pd.read_csv(f"{P2}/results/analysis20_ig_factorial.csv")
d = d[d.status == "ok"]
CELLS = [("mean", 50), ("mean", 200), ("zero", 50), ("zero", 200)]
LAB = ["Mean\n50 steps", "Mean\n200 steps", "Zero\n50 steps", "Zero\n200 steps"]
acr = d.pivot(index="building", columns=["baseline", "steps"], values="mean_rho")[CELLS]
com = d.pivot(index="building", columns=["baseline", "steps"], values="completeness_rel_err")[CELLS]
assert len(acr) == 25 and acr.notna().all().all()
assert round(acr[("mean", 200)].median(), 4) == 0.9441 and round(acr[("zero", 200)].median(), 4) == 0.9669

fs.setup()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(fs.TEXT_W, 2.3))
x = np.arange(4)
for ax, df, title, ylab in [(a1, acr, "(a) Across-draw agreement", "Spearman $\\rho$ across three draws"),
                            (a2, com, "(b) Completeness error", "Relative completeness error")]:
    for _, row in df.iterrows():
        ax.plot(x[:2], row.values[:2], color=fs.GREY, alpha=0.25, lw=0.6)
        ax.plot(x[2:], row.values[2:], color=fs.GREY, alpha=0.25, lw=0.6)
        ax.plot([1, 3], [row.values[1], row.values[3]], color=fs.BLUE, alpha=0.15, lw=0.6, ls=":")
    med = df.median().values
    cols = [fs.BLUE, fs.BLUE, fs.ORANGE, fs.ORANGE]
    ax.scatter(x, med, color=cols, s=28, zorder=4)
    for xi, m in zip(x, med):
        ax.text(xi + 0.13, m, f"{m:.4f}" if df is acr else f"{m:.3f}", fontsize=6.5, va="center",
                color=fs.GREY)
    ax.set_xticks(x, LAB)
    ax.set_xlim(-0.4, 3.6)
    ax.set_title(title, color=fs.GREY)
    ax.set_ylabel(ylab)
fig.tight_layout()
fs.save(fig, P2, "f02_ig_baseline_steps")
