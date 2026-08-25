"""
§6.2.1 cross-simulator request-characteristic validation figure (Table 7).

Replays three request shapes (short / medium / long) through BOTH CloudSimLLM
and Vidur on an identical single-A100 Llama-3-8B trace and overlays the
per-request results, showing the two simulators track each other across the
full request-shape range.

Panel (a): TTFT empirical CDF (log-x). Per shape, the Vidur (solid) and
           CloudSimLLM (dashed) curves overlap — agreement — while the three
           shapes separate ~6x horizontally — request-shape sensitivity.
Panel (b): TPOT grouped bars. Decode-per-token stays ~12 ms for every shape in
           both simulators — decode is memory-bound and shape-independent.

Reads committed per-request data from tools/analysis/data/vidur_shapes/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

from plot_utils import apply_paper_style, WORKLOAD_STYLE, WORKLOADS

# Vidur column names (per-request metrics dump).
V_TTFT = "prefill_e2e_time"                                   # s
V_TPOT = "decode_time_execution_plus_preemption_normalized"  # s / token


def _load(datadir: Path):
    """Return {shape: (vidur_ttft, cs_ttft, vidur_tpot_ms, cs_tpot_ms)}."""
    out = {}
    for wl in WORKLOADS:
        vd = pd.read_csv(datadir / f"vidur_{wl}.csv")
        cs = pd.read_csv(datadir / f"csllm_{wl}.csv")
        out[wl] = (
            vd[V_TTFT].to_numpy(),
            cs["ttft_sec"].to_numpy(),
            vd[V_TPOT].to_numpy() * 1000.0,
            cs["tpot_sec"].to_numpy() * 1000.0,
        )
    return out


def _cdf(ax, values, color, dashed):
    x = np.sort(values)
    y = np.arange(1, x.size + 1) / x.size
    ax.plot(x, y, color=color, linewidth=1.8,
            linestyle="--" if dashed else "-", alpha=0.95)


def make_figure(data, outdir: Path) -> None:
    fig, (axc, axb) = plt.subplots(1, 2, figsize=(11, 4.2))

    # ---- Panel (a): TTFT CDF overlay ------------------------------------
    for wl in WORKLOADS:
        color = WORKLOAD_STYLE[wl][0]
        v_ttft, c_ttft, _, _ = data[wl]
        _cdf(axc, v_ttft, color, dashed=False)   # Vidur solid
        _cdf(axc, c_ttft, color, dashed=True)    # CloudSimLLM dashed
    axc.set_xscale("log")
    axc.set_xlabel("TTFT (s, log scale)")
    axc.set_ylabel("Cumulative fraction of requests")
    axc.set_ylim(0, 1.02)
    axc.set_title("(a) TTFT distribution", fontsize=11, loc="left",
                  fontweight="bold", pad=4)
    # Two-part legend: colour = shape, line style = simulator.
    shape_handles = [mlines.Line2D([], [], color=WORKLOAD_STYLE[w][0],
                     linewidth=2.4, label=w) for w in WORKLOADS]
    sim_handles = [
        mlines.Line2D([], [], color="0.25", linewidth=2.0, linestyle="-",
                      label="Vidur"),
        mlines.Line2D([], [], color="0.25", linewidth=2.0, linestyle="--",
                      label="CloudSimLLM"),
    ]
    leg1 = axc.legend(handles=shape_handles, title="Request shape",
                      loc="lower right", fontsize=9, title_fontsize=9)
    axc.add_artist(leg1)
    axc.legend(handles=sim_handles, loc="upper left", fontsize=9)
    # Annotate the ~6x shape separation the two simulators agree on.
    axc.annotate("", xy=(0.23, 0.5), xytext=(0.037, 0.5),
                 arrowprops=dict(arrowstyle="<->", color="0.4", lw=1.1))
    axc.text(0.09, 0.54, r"$\approx$6$\times$", color="0.35", fontsize=9.5)

    # ---- Panel (b): TPOT grouped bars -----------------------------------
    x = np.arange(len(WORKLOADS))
    w = 0.36
    v_means = [data[wl][2].mean() for wl in WORKLOADS]
    c_means = [data[wl][3].mean() for wl in WORKLOADS]
    b1 = axb.bar(x - w / 2, v_means, w, label="Vidur", color="#7570b3",
                 edgecolor="k", linewidth=0.4)
    b2 = axb.bar(x + w / 2, c_means, w, label="CloudSimLLM", color="#d95f02",
                 edgecolor="k", linewidth=0.4)
    for bars in (b1, b2):
        for r in bars:
            axb.annotate(f"{r.get_height():.1f}",
                         (r.get_x() + r.get_width() / 2, r.get_height()),
                         ha="center", va="bottom", fontsize=8.5,
                         xytext=(0, 1), textcoords="offset points")
    axb.set_xticks(x)
    axb.set_xticklabels(WORKLOADS)
    axb.set_xlabel("Request shape")
    axb.set_ylabel("Mean TPOT (ms)")
    axb.set_ylim(0, max(max(v_means), max(c_means)) * 1.28)
    axb.grid(axis="x", visible=False)
    axb.set_title("(b) TPOT is shape-independent", fontsize=11, loc="left",
                  fontweight="bold", pad=4)
    axb.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"fig_vidur_shapes.{ext}",
                    dpi=200 if ext == "png" else None)
    plt.close(fig)
    print(f"[fig_vidur_shapes] {outdir/'fig_vidur_shapes.pdf'}")
    print(f"  TPOT Vidur {[round(v,1) for v in v_means]}  "
          f"CS {[round(v,1) for v in c_means]} ms")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/vidur_shapes")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    apply_paper_style()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    make_figure(_load(Path(args.data)), outdir)


if __name__ == "__main__":
    main()
