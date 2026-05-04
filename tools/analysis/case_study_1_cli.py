"""
Headless companion to case_study_1.ipynb.

Produces Fig 6/7/8 + summary table + LaTeX from the sweep CSV.

Usage:
    python case_study_1_cli.py \
        --results ../experiments/results/splitwise_sweep.csv \
        --outdir figures/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WORKLOADS = ["short", "medium", "long"]
COLORS    = {"short": "#1f77b4", "medium": "#2ca02c", "long": "#d62728"}
MARKERS   = {"short": "o", "medium": "s", "long": "^"}


METRIC_COLS = ["n_finished", "mean_ttft_s", "p99_ttft_s", "mean_tpot_s",
               "p99_tpot_s", "mean_e2e_s", "slo_attainment",
               "total_energy_kwh", "total_carbon_kg", "wall_ms"]


def aggregate_seeds(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Group by configuration columns and report mean+std across seeds."""
    metrics = [c for c in METRIC_COLS if c in df.columns]
    g = df.groupby(group_cols, dropna=False)
    agg = g[metrics].agg(["mean", "std"]).reset_index()
    # Flatten MultiIndex columns: ('p99_ttft_s', 'mean')->'p99_ttft_s';
    # ('p99_ttft_s', 'std')->'p99_ttft_s_std'; group cols are ('name','').
    new_cols = []
    for col in agg.columns:
        metric, stat = col if isinstance(col, tuple) else (col, "")
        if stat == "" or stat == "mean":
            new_cols.append(metric)
        else:
            new_cols.append(f"{metric}_{stat}")
    agg.columns = new_cols
    agg["n_seeds"] = g.size().reset_index(drop=True)
    return agg


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["pd_ratio"] = df.apply(lambda r: f"{int(r.prefill_gpus)}:{int(r.decode_gpus)}", axis=1)
    df["pd_share_prefill"] = df.prefill_gpus / (df.prefill_gpus + df.decode_gpus)
    # Aggregate across seeds: each (mode, workload, prefill, decode, kv_bw) becomes one row.
    group = ["mode", "workload", "prefill_gpus", "decode_gpus", "kv_bw_gbs",
             "pd_ratio", "pd_share_prefill"]
    return aggregate_seeds(df, group)


def fig6_ttft_vs_pd(df: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    view = df[(df["mode"] == "splitwise") & (df.kv_bw_gbs == 200)]
    for w in WORKLOADS:
        sub = view[view.workload == w].sort_values("pd_share_prefill")
        if sub.empty: continue
        ax.errorbar(sub.pd_share_prefill, sub.p99_ttft_s,
                    yerr=sub.p99_ttft_s_std, marker=MARKERS[w], color=COLORS[w],
                    label=f"splitwise — {w}", linewidth=1.6, capsize=2.5)
        base = df[(df["mode"] == "colocated") & (df.workload == w)]
        if not base.empty:
            ax.axhline(base.p99_ttft_s.mean(), color=COLORS[w], linestyle=":",
                       alpha=0.6, label=f"colocated — {w}")
    ax.set_xlabel("prefill GPU share  =  prefill / (prefill + decode)")
    ax.set_ylabel("P99 TTFT (s)  [mean $\\pm$ std, 5 seeds]")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right", ncols=2, frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "fig6_ttft_vs_pd_ratio.pdf")
    fig.savefig(outdir / "fig6_ttft_vs_pd_ratio.png", dpi=200)
    plt.close(fig)


def fig7_pareto(df: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    last_sc = None
    for ax, w in zip(axes, WORKLOADS):
        view = df[(df.workload == w) & (df.kv_bw_gbs == 200)]
        sw = view[view["mode"] == "splitwise"].sort_values("pd_share_prefill")
        co = view[view["mode"] == "colocated"]
        if sw.empty: continue
        sc = ax.scatter(sw.mean_e2e_s, 100*sw.slo_attainment, c=sw.prefill_gpus,
                        cmap="viridis", s=85, edgecolor="k", linewidth=0.5)
        last_sc = sc
        if not co.empty:
            ax.scatter(co.mean_e2e_s, 100*co.slo_attainment, c="red",
                       marker="*", s=180, label="colocated", edgecolor="k", zorder=5)
        for _, r in sw.iterrows():
            ax.annotate(f"{int(r.prefill_gpus)}p{int(r.decode_gpus)}d",
                        (r.mean_e2e_s, 100*r.slo_attainment),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("Mean E2E latency (s)")
        ax.set_ylabel("SLO attainment (%)")
        ax.grid(alpha=0.3)
        if w == WORKLOADS[-1] and not co.empty:
            ax.legend(loc="lower right", frameon=False)
    if last_sc is not None:
        plt.colorbar(last_sc, ax=axes[-1], fraction=0.045, label="prefill GPUs")
    fig.tight_layout()
    fig.savefig(outdir / "fig7_pareto.pdf")
    fig.savefig(outdir / "fig7_pareto.png", dpi=200)
    plt.close(fig)


def fig8_kvbw_sensitivity(df: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sw = df[df["mode"] == "splitwise"]
    bws = sorted(sw.kv_bw_gbs.unique())
    if len(bws) < 2:
        print("[fig8] skipping — need ≥ 2 KV BW values in sweep")
        return
    x = np.arange(len(bws))
    width = 0.25
    for i, w in enumerate(WORKLOADS):
        ref = sw[(sw.workload == w) & (sw.kv_bw_gbs == 200)]
        if ref.empty: continue
        best_pd = ref.loc[ref.p99_ttft_s.idxmin(), "pd_ratio"]
        sub = sw[(sw.workload == w) & (sw.pd_ratio == best_pd)].sort_values("kv_bw_gbs")
        ax.bar(x + i*width - width, sub.p99_ttft_s.values, width=width,
               yerr=sub.p99_ttft_s_std.values, capsize=2.5,
               color=COLORS[w], label=f"{w} (best={best_pd})",
               edgecolor="k", linewidth=0.4, error_kw=dict(ecolor='black', lw=0.6))
    ax.set_xticks(x); ax.set_xticklabels([f"{int(v)} GB/s" for v in bws])
    ax.set_ylabel("P99 TTFT (s)  [mean $\\pm$ std, 5 seeds]")
    ax.set_xlabel("Inter-host KV transfer bandwidth")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "fig8_kvbw_sensitivity.pdf")
    fig.savefig(outdir / "fig8_kvbw_sensitivity.png", dpi=200)
    plt.close(fig)


def summary_table(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows = []
    for w in WORKLOADS:
        co = df[(df.workload == w) & (df["mode"] == "colocated")]
        sw = df[(df.workload == w) & (df["mode"] == "splitwise") & (df.kv_bw_gbs == 200)]
        if co.empty or sw.empty: continue
        best = sw.loc[sw.p99_ttft_s.idxmin()]
        co_p99 = co.p99_ttft_s.mean()
        co_p99_std = co.p99_ttft_s.std() if len(co) > 1 else float("nan")
        rows.append({
            "workload": w, "best_pd": best.pd_ratio,
            "colo_p99_ttft": f"{co_p99:.2f}±{co_p99_std:.2f}" if not pd.isna(co_p99_std) else f"{co_p99:.2f}",
            "sw_p99_ttft":   f"{best.p99_ttft_s:.2f}±{best.p99_ttft_s_std:.2f}",
            "p99_ttft_speedup": f"{co_p99/best.p99_ttft_s:.2f}",
            "colo_slo":      f"{100*co.slo_attainment.mean():.1f}",
            "sw_slo":        f"{100*best.slo_attainment:.1f}±{100*best.slo_attainment_std:.1f}",
            "colo_energy_kwh": f"{co.total_energy_kwh.mean():.3f}",
            "sw_energy_kwh":   f"{best.total_energy_kwh:.3f}±{best.total_energy_kwh_std:.3f}",
        })
    summary = pd.DataFrame(rows).set_index("workload")
    summary.to_csv(outdir / "table_case_study_1.csv")
    tex = summary.to_latex()
    import re
    tex = re.sub(r"(?<!\\)_", r"\\_", tex)
    (outdir / "table_case_study_1.tex").write_text(tex)
    return summary


def sanity_checks(df: pd.DataFrame) -> list[str]:
    issues = []
    if (df.mean_ttft_s < 0).any():
        issues.append("negative mean TTFT detected — check arrival gating")
    if not (df.p99_ttft_s >= df.mean_ttft_s - 1e-6).all():
        issues.append("P99 TTFT < mean TTFT in some aggregated cells — check percentile computation")
    if "n_seeds" in df.columns and (df.n_seeds < 5).any():
        issues.append(f"some cells have <5 seeds (min={df.n_seeds.min()})")
    return issues


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

    fig6_ttft_vs_pd(df, outdir);          print(f"[fig6] {outdir/'fig6_ttft_vs_pd_ratio.pdf'}")
    fig7_pareto(df, outdir);              print(f"[fig7] {outdir/'fig7_pareto.pdf'}")
    fig8_kvbw_sensitivity(df, outdir);    print(f"[fig8] {outdir/'fig8_kvbw_sensitivity.pdf'}")

    summary = summary_table(df, outdir)
    print(f"\n[summary]\n{summary.to_string(float_format='%.3f')}\n")

    issues = sanity_checks(df)
    if issues:
        print("⚠  data quality concerns:")
        for i in issues: print("  -", i)
    else:
        print("✅ all sanity checks pass")


if __name__ == "__main__":
    main()
