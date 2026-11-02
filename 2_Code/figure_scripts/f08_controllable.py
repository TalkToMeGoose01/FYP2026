#!/usr/bin/env python3
"""f08: leading controllable meter, before and after the zero-fill correction.
Reads results/analysis12_controllable.csv only. Asserts the figures quoted in
sec:ranking before drawing. Usage: python3 f08_controllable.py <Phase2 dir>"""
import sys, pandas as pd, numpy as np
import figstyle as fs
import matplotlib.pyplot as plt

P2 = sys.argv[1] if len(sys.argv) > 1 else "."
c = pd.read_csv(f"{P2}/results/analysis12_controllable.csv")
masked = c[c.shap_ctrl_top1.notna()]
assert len(c) == 1276 and len(masked) == 538
METERS = ["chilledwater", "gas", "hotwater", "solar"]
NAMES = ["Chilled water", "Gas", "Hot water", "Solar"]
share = lambda s: [100 * (s == m).mean() for m in METERS]
panels = [("All 1,276 buildings, every meter ranked (uncorrected)",
           share(c.shap_ctrl_top1_unmasked), share(c.ig_ctrl_top1_unmasked)),
          ("538 buildings with a usable meter, carried meters only (corrected)",
           share(masked.shap_ctrl_top1), share(masked.ig_ctrl_top1))]
assert round(panels[0][1][1], 1) == 88.6 and round(panels[0][2][1], 1) == 81.2
assert round(panels[1][1][0], 1) == 49.1 and round(panels[1][2][0], 1) == 58.4

fs.setup()
fig, axes = plt.subplots(1, 2, figsize=(fs.TEXT_W, 2.3), sharey=True)
x, w = np.arange(4), 0.38
for ax, (title, s, g) in zip(axes, panels):
    for off, vals, col, lab in [(-w / 2, s, fs.BLUE, "SHAP"), (w / 2, g, fs.ORANGE, "Integrated Gradients")]:
        bars = ax.bar(x + off, vals, w, color=col, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=6.5, color=fs.GREY)
    ax.set_xticks(x, NAMES)
    ax.set_title(title, color=fs.GREY)
    ax.set_ylim(0, 100)
axes[0].set_ylabel("Buildings where the meter ranks first (%)")
axes[0].legend(loc="upper right")
fig.tight_layout()
fs.save(fig, P2, "f08_controllable")
