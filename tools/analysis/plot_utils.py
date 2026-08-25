"""
Shared plot styling for all CloudSimLLM experimental figures (FGCS
double-column). Import this at the top of every case_study_*_cli.py and
make_sensitivity_figs.py so the paper's figures are visually consistent.

Design rules enforced here (figure-designer audit):
  - serif font family to match the elsarticle/Times body text;
  - small canvas so fonts scale up to >=8pt in a column;
  - ColorBrewer-safe qualitative palette with dual encoding (colour+marker);
  - subtle grey grid, no chartjunk, top/right spines off;
  - vector PDF output at 300 dpi PNG fallback.
"""
from __future__ import annotations
import string
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ColorBrewer Dark2 (colour-blind-safe qualitative). Index 0 reserved for the
# "protagonist" series when a figure argues for one.
PALETTE = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
# Per-workload consistent colour + marker (dual encoding, B/W-safe).
WORKLOAD_STYLE = {
    "short":  ("#1f77b4", "o"),
    "medium": ("#2ca02c", "s"),
    "long":   ("#d62728", "^"),
}
WORKLOADS = ["short", "medium", "long"]


def apply_paper_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,            # embed TrueType, editable text
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#cccccc",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.5,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
        "legend.frameon": False,
    })


def panel_label(ax, index: int, workload: str | None = None, loc: str = "left") -> None:
    """Put an "(a) short" identifier on a subplot. Panel identifiers are NOT
    chart titles (which the caption replaces) — they tell the reader which
    workload each panel shows, which the caption cannot do per-panel."""
    tag = f"({string.ascii_lowercase[index]})"
    text = f"{tag} {workload}" if workload else tag
    ax.set_title(text, fontsize=10.5, loc=loc, fontweight="bold", pad=4)
