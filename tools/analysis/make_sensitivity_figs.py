"""
Extra figures for the revision:
  fig18 — autoscale SLO vs base arrival rate (the M3 sensitivity sweep)
  fig19 — DynamoLLM-style pool: size-aware vs round-robin routing SLO/energy
"""
from __future__ import annotations
import glob
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "tools/experiments/results"
OUT = ROOT / "tools/analysis/figures"
AUTOSCALE_COLS = ("label,policy,burst,startup_sec,max_pool,min_pool,initial_pool,"
                  "requests,workload,seed,n_finished,mean_ttft_s,p99_ttft_s,mean_tpot_s,"
                  "p99_tpot_s,mean_e2e_s,slo_attainment,cold_start_violations,"
                  "active_vm_hours,idle_vm_hours,cost_usd,total_energy_kwh,wall_ms").split(",")
POL_COLOR = {"STATIC": "#7570b3", "REACTIVE": "#1b9e77", "PREDICTIVE": "#d95f02"}
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})


def load_autoscale(f):
    df = pd.read_csv(f, header=None, names=AUTOSCALE_COLS)
    df = df[df.policy.isin(["STATIC", "REACTIVE", "PREDICTIVE"])]
    for c in ["slo_attainment", "cost_usd", "active_vm_hours"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["slo_attainment"])


def fig_sensitivity():
    rows = []
    for f in glob.glob(str(RES / "autoscale_sensitivity_r*.csv")):
        rate = int(f.split("_r")[-1].split(".")[0])
        df = load_autoscale(f)
        for pol in POL_COLOR:
            s = df[df.policy == pol]
            if s.empty:
                continue
            n = len(s)
            rows.append(dict(rate=rate, policy=pol,
                             slo=100 * s.slo_attainment.mean(),
                             slo_ci=100 * 1.96 * s.slo_attainment.std() / np.sqrt(n),
                             cost=s.cost_usd.mean()))
    d = pd.DataFrame(rows).sort_values("rate")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for pol in ["STATIC", "REACTIVE", "PREDICTIVE"]:
        s = d[d.policy == pol]
        ax1.errorbar(s.rate, s.slo, yerr=s.slo_ci, marker="o", color=POL_COLOR[pol],
                     label=pol, lw=1.8, capsize=2.5)
        ax2.plot(s.rate, s.cost, marker="s", color=POL_COLOR[pol], label=pol, lw=1.8)
    ax1.axhspan(0, 100, alpha=0)  # no-op to keep limits
    ax1.set_xlabel("Base arrival rate (req/s)")
    ax1.set_ylabel("SLO attainment (%)  [20 seeds, 95% CI]")
    ax1.axvspan(6, 12, color="gold", alpha=0.12, label="autoscaling-relevant")
    ax1.grid(alpha=0.3); ax1.legend(frameon=False, fontsize=9)
    ax2.set_xlabel("Base arrival rate (req/s)")
    ax2.set_ylabel("Cost (USD)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig18_autoscale_sensitivity.pdf")
    fig.savefig(OUT / "fig18_autoscale_sensitivity.png", dpi=200)
    plt.close(fig)
    print("[fig18] autoscale sensitivity written")
    return d


def fig_pool():
    df = pd.read_csv(RES / "pool_sweep.csv")
    df["rate"] = df.label.str.extract(r"-r(\d+)-").astype(int)
    df["key"] = df.config + "/" + df.routing.str.replace("ROUND_ROBIN", "RR").str.replace("SIZE_AWARE", "SA")
    g = df.groupby(["rate", "key"]).agg(slo=("slo_attainment", lambda x: 100 * x.mean()),
                                        jpt=("joules_per_token", "mean")).reset_index()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    markers = {"mixed/SA": ("o", "#d62728"), "mixed/RR": ("s", "#1f77b4"),
               "tp1x8/RR": ("^", "#2ca02c"), "tp4x2/RR": ("v", "#9467bd")}
    for key, (mk, col) in markers.items():
        s = g[g.key == key].sort_values("rate")
        if s.empty:
            continue
        lbl = {"mixed/SA": "mixed + size-aware", "mixed/RR": "mixed + round-robin",
               "tp1x8/RR": "8×TP1 uniform", "tp4x2/RR": "2×TP4 uniform"}[key]
        ax1.plot(s.rate, s.slo, marker=mk, color=col, label=lbl, lw=1.8)
        ax2.plot(s.rate, s.jpt, marker=mk, color=col, label=lbl, lw=1.8)
    ax1.set_xlabel("Arrival rate (req/s)"); ax1.set_ylabel("SLO attainment (%)")
    ax1.grid(alpha=0.3); ax1.legend(frameon=False, fontsize=8.5)
    ax2.set_xlabel("Arrival rate (req/s)"); ax2.set_ylabel("Energy per token (J/token)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig19_pool_routing.pdf")
    fig.savefig(OUT / "fig19_pool_routing.png", dpi=200)
    plt.close(fig)
    print("[fig19] pool routing written")


if __name__ == "__main__":
    fig_sensitivity()
    fig_pool()
