"""
§6.6 Case Study 4 — Bursty workload auto-scaling analysis.

Produces:
  Fig 15 — Cost vs SLO Pareto across (policy, burst, startup) cells
  Fig 16 — VM-hours saved vs static baseline, per startup-delay
  Fig 17 — Daily-mean policy comparison summary bars
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

POLICY_COLORS = {
    "STATIC":     "#7570b3",
    "REACTIVE":   "#1b9e77",
    "PREDICTIVE": "#d95f02",
}
BURSTS = ["low", "med", "high"]


def fig15_cost_slo_pareto(df: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, burst in zip(axes, BURSTS):
        sub = df[df.burst == burst]
        for policy, marker in zip(["STATIC", "REACTIVE", "PREDICTIVE"], "ovs"):
            ps = sub[sub.policy == policy]
            if ps.empty: continue
            ax.scatter(ps.cost_usd, 100*ps.slo_attainment, marker=marker, s=80,
                       color=POLICY_COLORS[policy], edgecolor="k", linewidth=0.4,
                       label=policy)
            for _, r in ps.iterrows():
                ax.annotate(f"{int(r.startup_sec)}s",
                            (r.cost_usd, 100*r.slo_attainment),
                            fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("Cost (USD)")
        ax.set_ylabel("SLO attainment (%)")
        ax.grid(alpha=0.3)
        if burst == BURSTS[-1]:
            ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fig15_cost_slo_pareto.pdf")
    fig.savefig(outdir / "fig15_cost_slo_pareto.png", dpi=200)
    plt.close(fig)


def fig16_savings_vs_startup(df: pd.DataFrame, outdir: Path) -> None:
    """For each (burst, startup), compute % VM-hour savings vs STATIC baseline."""
    rows = []
    for burst in BURSTS:
        for startup in df.startup_sec.unique():
            static = df[(df.policy == "STATIC") & (df.burst == burst) & (df.startup_sec == startup)]
            if static.empty: continue
            static_vh = static.active_vm_hours.iloc[0]
            for policy in ["REACTIVE", "PREDICTIVE"]:
                sub = df[(df.policy == policy) & (df.burst == burst) & (df.startup_sec == startup)]
                if sub.empty or static_vh == 0: continue
                savings_pct = 100 * (1 - sub.active_vm_hours.iloc[0] / static_vh)
                rows.append({"burst": burst, "startup": startup,
                            "policy": policy, "savings_pct": savings_pct})
    if not rows:
        print("[fig16] no data; skipping")
        return
    sav = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.30
    bursts = sav.burst.unique()
    x = np.arange(len(bursts))
    startups = sorted(sav.startup.unique())
    for i, startup in enumerate(startups):
        for policy in ["REACTIVE", "PREDICTIVE"]:
            vals = []
            for b in bursts:
                row = sav[(sav.burst == b) & (sav.startup == startup) & (sav.policy == policy)]
                vals.append(row.savings_pct.iloc[0] if not row.empty else 0)
            offset = i * width - width
            mark = "/" if policy == "REACTIVE" else "\\"
            ax.bar(x + offset, vals, width=width / 2,
                   color=POLICY_COLORS[policy], edgecolor="k", linewidth=0.4,
                   hatch=mark, label=f"{policy} (s={int(startup)})" if i == 0 else None)
    ax.set_xticks(x); ax.set_xticklabels(bursts)
    ax.set_xlabel("Burst level")
    ax.set_ylabel("VM-hour savings vs STATIC (%)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "fig16_savings_vs_startup.pdf")
    fig.savefig(outdir / "fig16_savings_vs_startup.png", dpi=200)
    plt.close(fig)


def fig17_summary(df: pd.DataFrame, outdir: Path) -> None:
    """Summary: cost, SLO, P99 TTFT bars per policy, averaged over bursts × startups."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    summary = df.groupby("policy").agg(
        cost=("cost_usd", "mean"),
        slo=("slo_attainment", "mean"),
        ttft_p99=("p99_ttft_s", "mean"),
        vmh=("active_vm_hours", "mean"),
    ).reindex(["STATIC", "REACTIVE", "PREDICTIVE"]).reset_index()
    panels = [("cost", "Mean cost (USD)"),
              ("vmh", "Mean active VM-hours"),
              ("slo", "Mean SLO attainment")]
    for ax, (col, ylabel) in zip(axes, panels):
        vals = summary[col].values
        if col == "slo": vals = vals * 100
        colors = [POLICY_COLORS[p] for p in summary.policy]
        ax.bar(range(len(summary)), vals, color=colors, edgecolor="k", linewidth=0.4)
        ax.set_xticks(range(len(summary))); ax.set_xticklabels(summary.policy)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "fig17_summary.pdf")
    fig.savefig(outdir / "fig17_summary.png", dpi=200)
    plt.close(fig)


def summary_table(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows = []
    for policy in ["STATIC", "REACTIVE", "PREDICTIVE"]:
        sub = df[df.policy == policy]
        rows.append({
            "policy": policy,
            "mean_cost":      sub.cost_usd.mean(),
            "mean_vm_hours":  sub.active_vm_hours.mean(),
            "mean_p99_ttft":  sub.p99_ttft_s.mean(),
            "mean_slo_pct":   100 * sub.slo_attainment.mean(),
            "cold_start_violations_total": sub.cold_start_violations.sum(),
        })
    summary = pd.DataFrame(rows).set_index("policy")
    summary.to_csv(outdir / "table_case_study_4.csv")
    tex = summary.to_latex(float_format="%.2f")
    import re
    tex = re.sub(r"(?<!\\)_", r"\\_", tex)
    (outdir / "table_case_study_4.tex").write_text(tex)
    return summary


METRIC_COLS = ["n_finished", "mean_ttft_s", "p99_ttft_s", "mean_tpot_s",
               "p99_tpot_s", "mean_e2e_s", "slo_attainment",
               "active_vm_hours", "idle_vm_hours", "cost_usd",
               "total_energy_kwh", "wall_ms", "cold_start_violations"]


def aggregate_seeds(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metrics = [c for c in METRIC_COLS if c in df.columns]
    g = df.groupby(group_cols, dropna=False)
    agg = g[metrics].agg(["mean", "std"]).reset_index()
    new_cols = []
    for col in agg.columns:
        metric, stat = col if isinstance(col, tuple) else (col, "")
        new_cols.append(metric if stat in ("", "mean") else f"{metric}_{stat}")
    agg.columns = new_cols
    agg["n_seeds"] = g.size().reset_index(drop=True)
    return agg


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
    df_raw = pd.read_csv(args.results)
    df = aggregate_seeds(df_raw, ["policy", "burst", "startup_sec"])
    print(f"[load] {len(df)} cells (aggregated from {len(df_raw)} raw rows × {df.n_seeds.iloc[0]} seeds)")

    fig15_cost_slo_pareto(df, outdir);   print(f"[fig15] {outdir/'fig15_cost_slo_pareto.pdf'}")
    fig16_savings_vs_startup(df, outdir); print(f"[fig16] {outdir/'fig16_savings_vs_startup.pdf'}")
    fig17_summary(df, outdir);            print(f"[fig17] {outdir/'fig17_summary.pdf'}")

    summary = summary_table(df, outdir)
    print(f"\n[summary]\n{summary.to_string(float_format='%.3f')}")


if __name__ == "__main__":
    main()
