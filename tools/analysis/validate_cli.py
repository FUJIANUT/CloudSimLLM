"""
Headless companion to validation.ipynb. Same outputs (table + Q-Q + heatmap +
pass/fail), but runnable in CI / on a server with no Jupyter.

Usage:
    python validate_cli.py \
        --real measurements_a100_llama3_8b.json \
        --sim  sim_a100_llama3_8b.csv \
        --outdir figures/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TARGET_REL_ERR_PCT = {
    "ttft_p50": 8, "ttft_p99": 8,
    "tpot_p50": 6, "tpot_p99": 6,
    "throughput_tok_per_s": 5,
}
KEY = ["input_len", "output_len", "concurrency"]


def load(real_path: str, sim_path: str) -> pd.DataFrame:
    real = pd.DataFrame(json.load(open(real_path)))
    sim = pd.read_csv(sim_path)
    df = real.merge(sim, on=KEY, suffixes=("_real", "_sim"))
    if df.empty:
        raise SystemExit(f"no overlapping cells between {real_path} and {sim_path}")
    return df


def error_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in TARGET_REL_ERR_PCT:
        rv = df[f"{m}_real"].values
        sv = df[f"{m}_sim"].values
        abs_err = np.abs(sv - rv)
        rel_err = abs_err / np.maximum(1e-9, rv)
        rows.append({
            "metric": m,
            "mean_abs_err": abs_err.mean(),
            "p99_abs_err":  np.quantile(abs_err, 0.99),
            "mean_rel_err_%": 100 * rel_err.mean(),
            "p99_rel_err_%":  100 * np.quantile(rel_err, 0.99),
        })
    return pd.DataFrame(rows).set_index("metric")


def qq_plot(df: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    panels = [
        ("ttft_p50", "TTFT (median per cell)", "s", 1.0),
        ("tpot_p50", "TPOT (median per cell)", "ms", 1000.0),
        ("throughput_tok_per_s", "Throughput", "tok/s", 1.0),
    ]
    for ax, (m, title, unit, scale) in zip(axes, panels):
        rv = df[f"{m}_real"].values * scale
        sv = df[f"{m}_sim"].values * scale
        qs = np.linspace(0.01, 0.99, 99)
        qr = np.quantile(rv, qs); qs_ = np.quantile(sv, qs)
        lo, hi = min(qr.min(), qs_.min()), max(qr.max(), qs_.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6)
        ax.plot(qr, qs_, "o", ms=3.5, alpha=0.8)
        ax.set_xlabel(f"real ({unit})"); ax.set_ylabel(f"simulated ({unit})")
        ax.set_title(title); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "fig_qq_validation.pdf")
    fig.savefig(outdir / "fig_qq_validation.png", dpi=200)
    plt.close(fig)


def residual_heatmap(df: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, m in zip(axes, ("ttft_p50", "tpot_p50")):
        rel = 100 * (df[f"{m}_sim"] - df[f"{m}_real"]) / df[f"{m}_real"]
        pivot = df.assign(rel=rel).pivot_table(
            index="input_len", columns="concurrency", values="rel", aggfunc="mean"
        )
        im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-15, vmax=15, aspect="auto")
        ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)));   ax.set_yticklabels(pivot.index)
        ax.set_xlabel("concurrency"); ax.set_ylabel("input tokens")
        ax.set_title(f"{m}: relative error (%)")
        plt.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(outdir / "fig_residual_heatmap.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = load(args.real, args.sim)

    errtab = error_table(df)
    errtab.to_csv(outdir / "table_validation_errors.csv")
    with open(outdir / "table_validation_errors.tex", "w") as f:
        f.write(errtab.to_latex(float_format="%.3f"))

    qq_plot(df, outdir)
    residual_heatmap(df, outdir)

    # Pass/fail summary
    print("\n===== Validation summary =====")
    passed = True
    for m, target in TARGET_REL_ERR_PCT.items():
        actual = errtab.loc[m, "mean_rel_err_%"]
        ok = actual <= target
        passed &= ok
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}]  {m:30s}  mean_rel_err = {actual:5.2f}%   target <= {target}%")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
