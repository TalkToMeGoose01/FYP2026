#!/usr/bin/env python3
"""f11: TFT configuration selection is selection on noise.
(a) confirmed three-run means, 21 July against 2 September 2026
    (tft_top5_confirmation_log.csv and the two RERUN files);
(b) ten seeds each of screening ranks 2 and 5 (results/analysis9_rank2_vs_rank5.csv,
    test-partition MAE%). Usage: python3 f11_tft_selection.py <Phase2 dir>"""
import sys, pandas as pd, numpy as np
import figstyle as fs
import matplotlib.pyplot as plt

P2 = sys.argv[1] if len(sys.argv) > 1 else "."
jul = pd.read_csv(f"{P2}/tft_top5_confirmation_log.csv").set_index("rank")
sepA = pd.read_csv(f"{P2}/tft_top5_confirmation_RERUN_2026-09-02T052146Z.csv").set_index("rank")
sepB = pd.read_csv(f"{P2}/tft_top5_confirmation_RERUN_2026-09-02T064610Z.csv").set_index("rank")
t9 = pd.read_csv(f"{P2}/results/analysis9_rank2_vs_rank5.csv")
t9 = t9[t9.status == "ok"]
assert round(jul.loc[2, "confirmed_mean"], 2) == 18.82 and round(sepB.loc[2, "confirmed_mean"], 2) == 20.07
st = t9.groupby("rank").mae_pct_testpartition.agg(["mean", "std", "count"])
assert (st["count"] == 10).all() and round(st.loc[2, "mean"], 2) == 18.96 and round(st.loc[5, "mean"], 2) == 19.90

fs.setup()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(fs.TEXT_W, 2.3), gridspec_kw={"width_ratios": [1.2, 1]})
for rk in [1, 2, 3, 4]:
    col = fs.ORANGE if rk == 2 else fs.BLUE
    y0, y1 = jul.loc[rk, "confirmed_mean"], sepB.loc[rk, "confirmed_mean"]
    a1.plot([0, 1], [y0, y1], color=col, lw=1, marker="o", ms=4, zorder=3)
    a1.text(-0.06, y0, f"rank {rk}", ha="right", va="center", fontsize=6.5, color=col)
a1.scatter([0], [jul.loc[5, "confirmed_mean"]], color=fs.GREY, s=14, zorder=3)
a1.text(0.05, jul.loc[5, "confirmed_mean"] - 0.04, "rank 5, not re-run", ha="left", va="top", fontsize=6.5, color=fs.GREY)
a1.set_ylim(17.8, None)
a1.scatter([1.08], [sepA.loc[2, "confirmed_mean"]], facecolor="white", edgecolor=fs.ORANGE, s=16, zorder=3)
a1.text(1.13, sepA.loc[2, "confirmed_mean"], "rank 2, second\nSeptember session", fontsize=6.5,
        color=fs.ORANGE, va="center")
a1.set_xticks([0, 1], ["21 July 2026", "2 September 2026"])
a1.set_xlim(-0.35, 1.6)
a1.set_ylabel("Confirmed three-run mean MAE%")
a1.set_title("(a) Confirmation means do not hold their order", color=fs.GREY)

for x, rk, col in [(0, 2, fs.ORANGE), (1, 5, fs.BLUE)]:
    v = t9[t9["rank"] == rk].mae_pct_testpartition.values
    jit = (np.arange(len(v)) - 4.5) * 0.025
    a2.scatter(x + jit, v, color=col, s=14, alpha=0.85, linewidths=0, zorder=3)
    m, s = st.loc[rk, "mean"], st.loc[rk, "std"]
    a2.errorbar(x + 0.3, m, yerr=s, fmt="o", color=col, capsize=2, ms=3, lw=1)
    a2.text(x + 0.38, m, f"{m:.2f} $\\pm$ {s:.2f}", va="center", fontsize=6.5, color=col)
a2.set_xticks([0, 1], ["Rank 2, batch 64\n(selected)", "Rank 5, batch 256\n(rule's choice)"])
a2.set_xlim(-0.4, 1.9)
a2.set_ylabel("Anchor test MAE%, ten seeds")
a2.set_title("(b) Ten seeds each, Welch $p = 0.33$", color=fs.GREY)
fig.tight_layout()
fs.save(fig, P2, "f11_tft_selection")
