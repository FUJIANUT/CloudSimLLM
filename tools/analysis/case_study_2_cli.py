"""
§6.4 Case Study 2 — Heterogeneous GPU mix analysis.

Produces:
  Fig 9  — TTFT P99 vs estimated $/M-token cost across mixes
  Fig 10 — Energy/latency Pareto across (mix, policy)
  Fig 11 — Mix robustness across workloads
  + summary table
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WORKLOADS = ["short", "medium", "long"]
COLORS    = {"short": "#1f77b4", "medium": "#2ca02c", "long": "#d62728"}

# Approximate $/hour list-price for cloud GPU rental (2026 average across providers).
SKU_COST_PER_HOUR = {"a100": 1.85, "h100": 4.20, "l40s": 1.10}


def parse_mix(mix: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for tok in mix.split("_"):
        sku, n = tok.split("x")
        out[sku] = out.get(sku, 0) + int(n)
    return out


def cost_per_hour(mix: str) -> float:
    return sum(SKU_COST_PER_HOUR[sku] * n for sku, n in parse_mix(mix).items())


def cost_per_million_tokens(row) -> float:
    """Estimate $/M-token throughput cost from sim wall-clock and total output tokens."""
    # tokens served = requests × avg output ≈ output_mid based on workload
    avg_out = {"short": 128, "medium": 448, "long": 320}[row.workload]
    tokens = row.n_finished * avg_out
    sim_seconds = row.mean_e2e_s if row.mean_e2e_s > 0 else row.wall_ms / 1000.0
    if tokens == 0 or sim_seconds <= 0:
        return float("nan")
    cph = cost_per_hour(row.mix)
    cost_per_token = cph / 3600.0 * sim_seconds / tokens
    return cost_per_token * 1e6


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["cost_per_hour"] = df.mix.apply(cost_per_hour)
    df["cost_per_M_token"] = df.apply(cost_per_million_tokens, axis=1)
    return df


def fig9_ttft_vs_cost(df: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, w in zip(axes, WORKLOADS):
        sub = df[(df.workload == w) & (df.policy == "FREE_HBM")]
        if sub.empty: continue
        for _, r in sub.iterrows():
            ax.scatter(r.cost_per_hour, r.p99_ttft_s, s=100, edgecolor="k",
                       linewidth=0.5, color="steelblue")
            ax.annotate(r.mix.replace("_", "\n"),
                        (r.cost_per_hour, r.p99_ttft_s),
                        fontsize=7, xytext=(5, 4), textcoords="offset points")
        ax.set_xlabel("Cluster cost ($/hour)")
        ax.set_ylabel("P99 TTFT (s)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "fig9_ttft_vs_cost.pdf")
    fig.savefig(outdir / "fig9_ttft_vs_cost.png", dpi=200)
    plt.close(fig)


def fig10_energy_pareto(df: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, w in zip(axes, WORKLOADS):
        sub = df[(df.workload == w) & (df.policy == "FREE_HBM")]
        if sub.empty: continue
        sc = ax.scatter(sub.total_energy_kwh, sub.mean_e2e_s, c=sub.cost_per_hour,
                        cmap="viridis", s=100, edgecolor="k", linewidth=0.5)
        for _, r in sub.iterrows():
            ax.annotate(r.mix, (r.total_energy_kwh, r.mean_e2e_s),
                        fontsize=7, xytext=(5, 4), textcoords="offset points")
        ax.set_xlabel("Total energy (kWh)")
        ax.set_ylabel("Mean E2E latency (s)")
        ax.grid(alpha=0.3)
        if w == WORKLOADS[-1]:
            plt.colorbar(sc, ax=ax, fraction=0.045, label="$/hour")
    fig.tight_layout()
    fig.savefig(outdir / "fig10_energy_pareto.pdf")
    fig.savefig(outdir / "fig10_energy_pareto.png", dpi=200)
    plt.close(fig)


def fig11_mix_robustness(df: pd.DataFrame, outdir: Path) -> None:
    """For each mix, plot TTFT P99 across all workloads. Stable mixes are 'robust'."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sub = df[df.policy == "FREE_HBM"]
    mixes = sub.mix.unique()
    x = np.arange(len(mixes))
    width = 0.25
    for i, w in enumerate(WORKLOADS):
        vals = [sub[(sub.mix == m) & (sub.workload == w)].p99_ttft_s.iloc[0]
                if not sub[(sub.mix == m) & (sub.workload == w)].empty else 0.0
                for m in mixes]
        ax.bar(x + i*width - width, vals, width=width, label=w,
               color=COLORS[w], edgecolor="k", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(mixes, rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("P99 TTFT (s)")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "fig11_mix_robustness.pdf")
    fig.savefig(outdir / "fig11_mix_robustness.png", dpi=200)
    plt.close(fig)


def summary_table(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows = []
    for w in WORKLOADS:
        sub = df[(df.workload == w) & (df.policy == "FREE_HBM")]
        if sub.empty: continue
        # min-cost satisfying P99 TTFT <= some practical SLO (10s)
        feasible = sub[sub.p99_ttft_s <= 10.0]
        cheapest = feasible.loc[feasible.cost_per_hour.idxmin()] if not feasible.empty else None
        # min-latency overall
        fastest = sub.loc[sub.p99_ttft_s.idxmin()]
        # min-energy
        greenest = sub.loc[sub.total_energy_kwh.idxmin()]
        rows.append({
            "workload": w,
            "fastest_mix": fastest.mix,
            "fastest_p99_ttft": fastest.p99_ttft_s,
            "fastest_cost": fastest.cost_per_hour,
            "cheapest_feasible": cheapest.mix if cheapest is not None else "none",
            "cheapest_cost":     cheapest.cost_per_hour if cheapest is not None else float("nan"),
            "cheapest_p99_ttft": cheapest.p99_ttft_s if cheapest is not None else float("nan"),
            "greenest_mix":      greenest.mix,
            "greenest_kwh":      greenest.total_energy_kwh,
        })
    summary = pd.DataFrame(rows).set_index("workload")
    summary.to_csv(outdir / "table_case_study_2.csv")
    tex = summary.to_latex(float_format="%.2f")
    import re
    tex = re.sub(r"(?<!\\)_", r"\\_", tex)
    (outdir / "table_case_study_2.tex").write_text(tex)
    return summary


def policy_robustness(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    """Compare FREE_HBM vs EST_TTFT across all (mix, workload) cells."""
    pivot = df.pivot_table(index=["mix", "workload"], columns="policy",
                           values="p99_ttft_s", aggfunc="mean")
    pivot["pct_diff"] = 100 * (pivot["EST_TTFT"] - pivot["FREE_HBM"]) / pivot["FREE_HBM"]
    pivot.to_csv(outdir / "table_policy_robustness.csv")
    return pivot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300,
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = load(Path(args.results))
    print(f"[load] {len(df)} cells")

    fig9_ttft_vs_cost(df, outdir);    print(f"[fig9]  {outdir/'fig9_ttft_vs_cost.pdf'}")
    fig10_energy_pareto(df, outdir);  print(f"[fig10] {outdir/'fig10_energy_pareto.pdf'}")
    fig11_mix_robustness(df, outdir); print(f"[fig11] {outdir/'fig11_mix_robustness.pdf'}")

    summary = summary_table(df, outdir)
    print(f"\n[summary]\n{summary.to_string(float_format='%.3f')}\n")

    pol = policy_robustness(df, outdir)
    print(f"[policy] FREE_HBM vs EST_TTFT difference: median |%| = {pol.pct_diff.abs().median():.2f}%")


if __name__ == "__main__":
    main()
