"""
§6.5 Case Study 3 — Carbon-aware geo-distributed routing analysis.

Produces:
  Fig 12 — Carbon vs hour-of-day per policy, per workload
  Fig 13 — TTFT P99 vs total carbon Pareto (per workload)
  Fig 14 — Daily-mean reduction summary bars
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
POLICY_COLORS = {
    "LATENCY_GREEDY": "#7570b3",
    "CARBON_AWARE":   "#1b9e77",
    "BLENDED":        "#d95f02",
}


METRIC_COLS = ["n_finished", "mean_ttft_s", "p99_ttft_s", "mean_tpot_s",
               "p99_tpot_s", "mean_e2e_s", "slo_attainment", "lambda",
               "total_energy_kwh", "total_carbon_kg", "wall_ms"]


def aggregate_seeds(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    from scipy import stats as scstats
    metrics = [c for c in METRIC_COLS if c in df.columns]
    g = df.groupby(group_cols, dropna=False)
    agg = g[metrics].agg(["mean", "std", "count"]).reset_index()
    new_cols = []
    for col in agg.columns:
        metric, stat = col if isinstance(col, tuple) else (col, "")
        if stat == "" or stat == "mean":
            new_cols.append(metric)
        elif stat == "std":
            new_cols.append(f"{metric}_std")
        elif stat == "count":
            new_cols.append(f"{metric}_n")
    agg.columns = new_cols
    agg["n_seeds"] = g.size().reset_index(drop=True)
    for m in metrics:
        n = agg[f"{m}_n"]
        sd = agg[f"{m}_std"].fillna(0.0)
        tcrit = pd.Series(
            [scstats.t.ppf(0.975, max(1, k - 1)) for k in n.fillna(1).astype(int)],
            index=agg.index)
        agg[f"{m}_ci"] = tcrit * sd / np.sqrt(n.where(n > 0, 1))
    return agg


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return aggregate_seeds(df, ["policy", "workload", "hour"])


def fig12_carbon_vs_hour(df: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, w in zip(axes, WORKLOADS):
        sub = df[df.workload == w].sort_values("hour")
        for policy in ["LATENCY_GREEDY", "CARBON_AWARE", "BLENDED"]:
            ps = sub[sub.policy == policy]
            if ps.empty: continue
            ax.plot(ps.hour, ps.total_carbon_kg, marker="o",
                    color=POLICY_COLORS[policy], label=policy.replace("_", "\n"),
                    linewidth=1.6)
        ax.set_xlabel("Hour of day (UTC)")
        ax.set_ylabel("Total carbon (kg CO$_2$eq)")
        ax.grid(alpha=0.3)
        ax.set_xticks(range(0, 24, 6))
        if w == WORKLOADS[-1]:
            ax.legend(loc="upper right", frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "fig12_carbon_vs_hour.pdf")
    fig.savefig(outdir / "fig12_carbon_vs_hour.png", dpi=200)
    plt.close(fig)


def fig13_pareto_carbon_latency(df: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, w in zip(axes, WORKLOADS):
        sub = df[df.workload == w]
        for policy, marker in zip(["LATENCY_GREEDY", "CARBON_AWARE", "BLENDED"], "ovs"):
            ps = sub[sub.policy == policy]
            if ps.empty: continue
            ax.scatter(ps.total_carbon_kg, ps.p99_ttft_s, marker=marker, s=70,
                       color=POLICY_COLORS[policy], edgecolor="k", linewidth=0.4,
                       label=policy.replace("_", " "))
        ax.set_xlabel("Total carbon (kg CO$_2$eq)")
        ax.set_ylabel("P99 TTFT (s)")
        ax.grid(alpha=0.3)
        if w == WORKLOADS[-1]:
            ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fig13_pareto_carbon_latency.pdf")
    fig.savefig(outdir / "fig13_pareto_carbon_latency.png", dpi=200)
    plt.close(fig)


def fig14_daily_summary(df: pd.DataFrame, outdir: Path) -> None:
    """Bar chart of daily-mean carbon reduction per workload."""
    summary = df.groupby(["workload", "policy"]).agg(
        mean_carbon=("total_carbon_kg", "mean"),
        mean_ttft_p99=("p99_ttft_s", "mean"),
    ).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: carbon
    x = np.arange(len(WORKLOADS))
    width = 0.27
    for i, p in enumerate(["LATENCY_GREEDY", "CARBON_AWARE", "BLENDED"]):
        vals = [summary[(summary.workload == w) & (summary.policy == p)].mean_carbon.iloc[0]
                if not summary[(summary.workload == w) & (summary.policy == p)].empty else 0.0
                for w in WORKLOADS]
        axes[0].bar(x + i*width - width, vals, width=width, label=p,
                    color=POLICY_COLORS[p], edgecolor="k", linewidth=0.4)
    axes[0].set_xticks(x); axes[0].set_xticklabels(WORKLOADS)
    axes[0].set_ylabel("Mean daily carbon (kg)")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8)

    # Right: P99 TTFT
    for i, p in enumerate(["LATENCY_GREEDY", "CARBON_AWARE", "BLENDED"]):
        vals = [summary[(summary.workload == w) & (summary.policy == p)].mean_ttft_p99.iloc[0]
                if not summary[(summary.workload == w) & (summary.policy == p)].empty else 0.0
                for w in WORKLOADS]
        axes[1].bar(x + i*width - width, vals, width=width, label=p,
                    color=POLICY_COLORS[p], edgecolor="k", linewidth=0.4)
    axes[1].set_xticks(x); axes[1].set_xticklabels(WORKLOADS)
    axes[1].set_ylabel("Mean daily P99 TTFT (s)")
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "fig14_daily_summary.pdf")
    fig.savefig(outdir / "fig14_daily_summary.png", dpi=200)
    plt.close(fig)


def summary_table(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows = []
    for w in WORKLOADS:
        sub = df[df.workload == w]
        if sub.empty: continue
        greedy = sub[sub.policy == "LATENCY_GREEDY"]
        carbon = sub[sub.policy == "CARBON_AWARE"]
        blend  = sub[sub.policy == "BLENDED"]

        rows.append({
            "workload": w,
            "greedy_kg":      greedy.total_carbon_kg.mean(),
            "carbon_kg":      carbon.total_carbon_kg.mean(),
            "blended_kg":     blend.total_carbon_kg.mean(),
            "carbon_savings_pct":  100*(1 - carbon.total_carbon_kg.mean() / greedy.total_carbon_kg.mean()),
            "blended_savings_pct": 100*(1 - blend.total_carbon_kg.mean() / greedy.total_carbon_kg.mean()),
            "greedy_p99_ttft":  greedy.p99_ttft_s.mean(),
            "carbon_p99_ttft":  carbon.p99_ttft_s.mean(),
            "blended_p99_ttft": blend.p99_ttft_s.mean(),
            "ttft_penalty_pct":  100*(carbon.p99_ttft_s.mean() / greedy.p99_ttft_s.mean() - 1),
        })
    summary = pd.DataFrame(rows).set_index("workload")
    summary.to_csv(outdir / "table_case_study_3.csv")
    tex = summary.to_latex(float_format="%.2f")
    import re
    tex = re.sub(r"(?<!\\)_", r"\\_", tex)
    (outdir / "table_case_study_3.tex").write_text(tex)
    return summary


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

    fig12_carbon_vs_hour(df, outdir);          print(f"[fig12] {outdir/'fig12_carbon_vs_hour.pdf'}")
    fig13_pareto_carbon_latency(df, outdir);   print(f"[fig13] {outdir/'fig13_pareto_carbon_latency.pdf'}")
    fig14_daily_summary(df, outdir);           print(f"[fig14] {outdir/'fig14_daily_summary.pdf'}")

    summary = summary_table(df, outdir)
    print(f"\n[summary]\n{summary.to_string(float_format='%.3f')}")


if __name__ == "__main__":
    main()
