"""Shared style for the appendix figures (matches f06_search_flow.py)."""
import pathlib
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

BLUE, ORANGE, GREY, LIGHT = "#127bb7", "#d76912", "#5c5c5c", "#dbe9f4"
COL_W, TEXT_W = 3.5, 7.16


def setup():
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "pdf.fonttype": 42,
        "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": GREY, "axes.linewidth": 0.6, "xtick.color": GREY,
        "ytick.color": GREY, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False})


def save(fig, phase2, name):
    for d in ["figures", "FinalReport/figures"]:
        fig.savefig(pathlib.Path(phase2) / d / f"{name}.pdf", bbox_inches="tight",
                    metadata={"CreationDate": None})
    print("written", name)
