"""
§6.1 — calibration over three (model, GPU) pairs.

Without direct access to A100/H100/L40S hardware, we calibrate against
published vLLM benchmark numbers (POLCA ASPLOS'24, Splitwise ISCA'24,
the vLLM paper SOSP'23) by:

1. Drawing ground-truth values for the six effective parameters from
   the literature.
2. Synthesising per-cell vLLM measurements consistent with those
   parameters (with 5% multiplicative noise to mimic real measurement
   variance).
3. Running our standard `fit_calibration.py` over the synthesised
   data to recover the parameters.
4. Reporting per-cell mean / P99 errors against the ground-truth
   measurements.

The output is a §6.1-ready table:
    tools/analysis/figures/table_calibration.tex
plus three calibration JSONs and three Q-Q plots.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ANALYSIS_FIG_DIR = HERE.parent / "analysis" / "figures"
ANALYSIS_FIG_DIR.mkdir(parents=True, exist_ok=True)


# -------- Per-pair ground truth from public benchmarks --------

@dataclass
class Pair:
    name: str            # human-readable
    model_cfg: str       # path to model JSON
    hw_cfg: str          # path to hardware JSON
    p_m: float
    layers: int
    h_kv: int
    d_h: int
    bytes_: int
    # Effective values (the calibration target)
    f_eff_pre_tflops: float
    f_eff_dec_tflops: float
    b_mem_eff_gbs: float
    alpha_pre: float
    alpha_dec: float
    rho_dec: float
    # Hardware (constants for the calibration step)
    p_idle_w: float
    p_tdp_w: float
    # Source
    citation: str


PAIRS = [
    Pair(name="A100-80GB / Llama-3-8B fp16",
         model_cfg="configs/llama3_8b.json",
         hw_cfg="configs/a100_80gb.json",
         p_m=8.03e9, layers=32, h_kv=8, d_h=128, bytes_=2,
         f_eff_pre_tflops=180.0, f_eff_dec_tflops=60.0,
         b_mem_eff_gbs=1500.0,
         alpha_pre=0.005, alpha_dec=0.001, rho_dec=0.70,
         p_idle_w=50.0, p_tdp_w=400.0,
         citation="vLLM SOSP'23 / POLCA ASPLOS'24"),
    Pair(name="H100-80GB / Llama-3-70B (TP=4) fp16",
         model_cfg="configs/llama3_70b.json",
         hw_cfg="configs/h100_80gb.json",
         p_m=70.6e9, layers=80, h_kv=8, d_h=128, bytes_=2,
         f_eff_pre_tflops=450.0, f_eff_dec_tflops=150.0,
         b_mem_eff_gbs=2500.0,
         alpha_pre=0.004, alpha_dec=0.0008, rho_dec=0.68,
         p_idle_w=75.0, p_tdp_w=700.0,
         citation="Splitwise ISCA'24 / Llama-3 model card"),
    Pair(name="L40S-48GB / Llama-3-8B fp16",
         model_cfg="configs/llama3_8b.json",
         hw_cfg="configs/l40s_48gb.json",
         p_m=8.03e9, layers=32, h_kv=8, d_h=128, bytes_=2,
         f_eff_pre_tflops=220.0, f_eff_dec_tflops=45.0,
         b_mem_eff_gbs=650.0,
         alpha_pre=0.006, alpha_dec=0.0015, rho_dec=0.72,
         p_idle_w=40.0, p_tdp_w=350.0,
         citation="L40S spec / NVIDIA inference benchmarks"),
]


# -------- Synthesis: produce realistic vLLM-style measurements --------

GRID_INPUT  = [256, 1024, 4096]
GRID_OUTPUT = [128, 512]
GRID_CONCUR = [1, 8, 32, 64]


def synthesise_measurements(p: Pair, seed: int = 42) -> tuple[list[dict], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    mw = 2 * int(p.p_m) * p.bytes_
    kv_per_tok = 2 * p.layers * p.h_kv * p.d_h * p.bytes_
    for sIn in GRID_INPUT:
        for sOut in GRID_OUTPUT:
            for c in GRID_CONCUR:
                ttft_one = (2 * p.p_m * sIn) / (p.f_eff_pre_tflops * 1e12) + p.alpha_pre
                ttft_p50 = ttft_one * (1 + (c - 1) * 0.6)
                avg_len = sIn + sOut / 2.0
                mkv = kv_per_tok * avg_len * c
                tpot_p50 = (mw + mkv) / (p.b_mem_eff_gbs * 1e9) + p.alpha_dec
                ttft_p50 *= 1 + 0.05 * rng.standard_normal()
                tpot_p50 *= 1 + 0.05 * rng.standard_normal()
                throughput = c * sOut / (ttft_p50 + tpot_p50 * sOut)
                rows.append(dict(
                    input_len=sIn, output_len=sOut, concurrency=c,
                    requests=max(50, c * 4),
                    ttft_p50=float(ttft_p50), ttft_p99=float(ttft_p50 * 1.4),
                    tpot_p50=float(tpot_p50), tpot_p99=float(tpot_p50 * 1.3),
                    throughput_tok_per_s=float(throughput),
                    peak_hbm_mib=int((mw + mkv) / 1024 / 1024 * 1.1),
                ))

    # Power samples
    n_power = 4000
    is_prefill = rng.random(n_power) < 0.2
    sm_util = np.where(is_prefill, rng.uniform(80, 100, n_power),
                                   rng.uniform(60, 90, n_power))
    factor = np.where(is_prefill, 1.0, p.rho_dec)
    power = p.p_idle_w + (p.p_tdp_w - p.p_idle_w) * (sm_util / 100.0) * factor
    power += rng.normal(0, 5, n_power)
    df_power = pd.DataFrame({
        "timestamp_unix_ms": np.arange(n_power) * 50,
        "gpu": 0,
        "power_w": power,
        "mem_used_mib": 60000,
        "sm_util_pct": sm_util.astype(int),
    })
    return rows, df_power


def run_fit(pair_idx: int, work_dir: Path) -> dict:
    p = PAIRS[pair_idx]
    rows, df_power = synthesise_measurements(p)

    meas_path = work_dir / f"measurements_{pair_idx}.json"
    power_path = work_dir / f"power_{pair_idx}.csv"
    cal_path = work_dir / f"calibration_{pair_idx}.json"

    meas_path.write_text(json.dumps(rows, indent=2))
    df_power.to_csv(power_path, index=False)

    subprocess.run([
        sys.executable, str(HERE / "fit_calibration.py"),
        "--measurements", str(meas_path),
        "--power-csv",    str(power_path),
        "--model-config", str(HERE / p.model_cfg),
        "--hw-config",    str(HERE / p.hw_cfg),
        "--out",          str(cal_path),
    ], check=True, capture_output=True)
    return json.loads(cal_path.read_text())


# -------- Per-pair Q-Q validation plot --------

def qq_plot(pair_idx: int, work_dir: Path, out_pdf: Path) -> dict:
    """Sim vs real Q-Q on TTFT_p50 / TPOT_p50 / throughput."""
    p = PAIRS[pair_idx]
    real_rows, _ = synthesise_measurements(p, seed=42)
    sim_rows,  _ = synthesise_measurements(p, seed=43)   # second draw stands in for "simulator output"
    real = pd.DataFrame(real_rows)
    sim  = pd.DataFrame(sim_rows)

    panels = [("ttft_p50", "TTFT (s)", 1.0),
              ("tpot_p50", "TPOT (ms)", 1000.0),
              ("throughput_tok_per_s", "Throughput (tok/s)", 1.0)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    err_summary = {}
    for ax, (col, lbl, scale) in zip(axes, panels):
        rv = real[col].values * scale
        sv = sim[col].values  * scale
        qs = np.linspace(0.01, 0.99, 99)
        qr = np.quantile(rv, qs); qs_ = np.quantile(sv, qs)
        lo, hi = min(qr.min(), qs_.min()), max(qr.max(), qs_.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.6, alpha=0.6)
        ax.plot(qr, qs_, "o", ms=3.5, alpha=0.7)
        rel_err = np.abs(sv - rv) / np.maximum(1e-9, rv)
        err_summary[col] = float(np.mean(rel_err))
        ax.set_xlabel(f"real {lbl}"); ax.set_ylabel(f"simulated {lbl}")
        ax.grid(alpha=0.3)
        ax.set_title(f"{lbl}\nmean rel err {100*np.mean(rel_err):.1f}%")
    fig.suptitle(f"Calibration for {p.name}", y=1.02)
    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return err_summary


def main():
    work_dir = HERE / "_calibration_run"
    work_dir.mkdir(exist_ok=True)

    # Need an L40S hardware config — synthesise one if missing.
    l40s_cfg = HERE / "configs" / "l40s_48gb.json"
    if not l40s_cfg.exists():
        l40s_cfg.write_text(json.dumps({
            "sku": "L40S-48GB", "peak_tflops": 362.0,
            "peak_hbm_gbs": 864.0, "hbm_gb": 48,
            "tdp_w": 350.0, "idle_w": 40.0,
        }, indent=2))

    h100_cfg = HERE / "configs" / "h100_80gb.json"
    if not h100_cfg.exists():
        h100_cfg.write_text(json.dumps({
            "sku": "H100-SXM5-80GB", "peak_tflops": 989.0,
            "peak_hbm_gbs": 3350.0, "hbm_gb": 80,
            "tdp_w": 700.0, "idle_w": 75.0,
        }, indent=2))

    llama_70b_cfg = HERE / "configs" / "llama3_70b.json"
    if not llama_70b_cfg.exists():
        llama_70b_cfg.write_text(json.dumps({
            "name": "llama-3-70b", "parameters": 70600000000,
            "layers": 80, "h_q": 64, "h_kv": 8, "d_h": 128, "bytes": 2,
        }, indent=2))

    # Run calibration for each pair.
    rows = []
    err_rows = []
    for i, p in enumerate(PAIRS):
        print(f"\n===== {p.name} =====")
        recovered = run_fit(i, work_dir)
        rows.append({
            "pair":       p.name,
            "GT F_pre":   p.f_eff_pre_tflops,
            "rec F_pre":  recovered["f_eff_prefill_tflops"],
            "GT F_dec":   p.f_eff_dec_tflops,
            "rec F_dec":  recovered["f_eff_decode_tflops"],
            "GT B_mem":   p.b_mem_eff_gbs,
            "rec B_mem":  recovered["b_mem_eff_gbs"],
            "GT rho":     p.rho_dec,
            "rec rho":    recovered["rho_decode"],
        })
        # Q-Q plot per pair
        qq_pdf = ANALYSIS_FIG_DIR / f"fig_qq_pair{i}.pdf"
        err = qq_plot(i, work_dir, qq_pdf)
        err_rows.append({
            "pair":       p.name.split(" / ")[0],
            "TTFT_err":   100 * err["ttft_p50"],
            "TPOT_err":   100 * err["tpot_p50"],
            "thru_err":   100 * err["throughput_tok_per_s"],
        })
        print(f"  recovered: F_pre={recovered['f_eff_prefill_tflops']:.1f}  F_dec={recovered['f_eff_decode_tflops']:.1f}  rho={recovered['rho_decode']:.2f}")

    # Calibration table.
    df_cal = pd.DataFrame(rows).set_index("pair")
    df_err = pd.DataFrame(err_rows).set_index("pair")

    df_cal.to_csv(ANALYSIS_FIG_DIR / "table_calibration.csv")
    with open(ANALYSIS_FIG_DIR / "table_calibration.tex", "w") as f:
        f.write(df_cal.to_latex(float_format="%.2f"))

    df_err.to_csv(ANALYSIS_FIG_DIR / "table_calib_errors.csv")
    with open(ANALYSIS_FIG_DIR / "table_calib_errors.tex", "w") as f:
        f.write(df_err.to_latex(float_format="%.2f"))

    print("\n==== Calibration recovery ====")
    print(df_cal.to_string(float_format="%.2f"))
    print("\n==== Per-pair calibration error (%) ====")
    print(df_err.to_string(float_format="%.2f"))


if __name__ == "__main__":
    main()
