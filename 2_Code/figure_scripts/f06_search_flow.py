#!/usr/bin/env python3
"""Figure f06: structured search and screening flow (Appendix figs).
Rebuilt 19 Oct 2026 after the author review (a10d) raised the included set
from 11 to 12. Counts below come from screening/screening_SUMMARY.txt and
screening/screening_ALL.csv; the script asserts the screening counts.
Usage: python3 f06_search_flow.py <Phase2 dir>
"""
import csv, sys, pathlib, collections
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

P2 = pathlib.Path(sys.argv[1])
rows = list(csv.DictReader(open(P2 / "screening" / "screening_ALL.csv", encoding="utf-8")))
dec = collections.Counter(r["decision"] for r in rows)
n_union = len(rows)
n_flag = sum(r["triage_flag"] == "True" for r in rows)
n_in = dec["IN"]
assert (n_union, n_flag, n_in) == (817, 174, 12), (n_union, n_flag, n_in)

RETRIEVED, DEDUP = 1113, 865
stages = [
    (f"Records retrieved: {RETRIEVED:,}", "Scopus 855, Web of Science 258", "#127bb7", 0.98),
    (f"After DOI de-duplication: {DEDUP}", "within and across databases", "#127bb7", 0.97),
    (f"Unique across four searches: {n_union}", "searches A-D combined", "#127bb7", 0.96),
    (f"Flagged by triage: {n_flag}", "re-initialisation × attribution × stability", "#127bb7", 0.85),
    (f"Met inclusion criterion: {n_in}", "all from Search D, one also from Search A", "#d76912", 0.63),
]
drops = [RETRIEVED - DEDUP, DEDUP - n_union, n_union - n_flag, n_flag - n_in]

plt.rcParams.update({"font.family": "STIXGeneral", "pdf.fonttype": 42})
fig = plt.figure(figsize=(3.5, 2.75))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
xc, h = 0.484, 0.10
ys = [0.924, 0.722, 0.520, 0.320, 0.120]
for i, ((label, sub, col, w), y) in enumerate(zip(stages, ys)):
    ax.add_patch(FancyBboxPatch((xc - w / 2, y - h / 2), w, h,
                 boxstyle="round,pad=0,rounding_size=0.015", fc=col, ec="none"))
    ax.text(xc, y, label, ha="center", va="center", color="white", fontsize=7.6)
    ax.text(xc, y - 0.069, sub, ha="center", va="center", color="#555555", fontsize=7.2)
    if i < len(drops):
        top, bot = y - 0.088, ys[i + 1] + h / 2 + 0.004
        ax.annotate("", xy=(xc, bot), xytext=(xc, top),
                    arrowprops=dict(arrowstyle="-|>", color="#5c5c5c", lw=1.3, mutation_scale=9))
        ax.text(xc + 0.04, (top + bot) / 2 + 0.008, f"−{drops[i]}", ha="left",
                va="center", color="#d76912", fontsize=7.2)
for out in [P2 / "figures" / "f06_search_flow.pdf", P2 / "FinalReport" / "figures" / "f06_search_flow.pdf"]:
    fig.savefig(out, metadata={"CreationDate": None})
print("written", n_union, n_flag, n_in, drops)
