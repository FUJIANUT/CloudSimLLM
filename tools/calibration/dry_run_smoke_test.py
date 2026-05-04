"""
Synthetic-data smoke test of the calibration pipeline.

Generates plausible vLLM-shaped measurements + power samples for an A100 +
Llama-3-8B workload (using the analytic models that fit_calibration.py is
trying to invert), then runs fit_calibration.py and asserts the recovered
parameters match the ground-truth values within tolerance.

Useful when there is no GPU available: validates the entire pipeline so a real
calibration run on hardware later only differs in the data source.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# Ground-truth A100 calibration we want fit_calibration.py to recover.
GT = dict(
    p_m=8.03e9,                   # Llama-3-8B parameters
    layers=32, h_kv=8, d_h=128, bytes_=2,
    f_eff_pre_tflops=180.0,
    f_eff_dec_tflops=58.0,
    b_mem_eff_gbs=1500.0,
    alpha_pre=0.005,
    alpha_dec=0.001,
    rho_dec=0.70,
    p_idle=50.0,
    p_tdp=400.0,
)


def synthesize_measurements(out_json: Path) -> None:
    rng = np.random.default_rng(42)
    inputs  = [256, 1024, 4096]
    outputs = [128, 512]
    concurs = [1, 8, 32, 64]
    rows = []
    p_m = GT["p_m"]
    mw = 2 * int(p_m) * GT["bytes_"]
    kv_per_tok = 2 * GT["layers"] * GT["h_kv"] * GT["d_h"] * GT["bytes_"]
    for sIn in inputs:
        for sOut in outputs:
            for c in concurs:
                # Eq. (1) for prefill — concurrency=1 is single-request batch
                ttft_one = (2 * p_m * sIn) / (GT["f_eff_pre_tflops"] * 1e12) + GT["alpha_pre"]
                # When concur > 1, batch sums tokens; serialize as queue + last-batch prefill.
                ttft_p50 = ttft_one * (1 + (c - 1) * 0.6)   # heavy queueing
                # Eq. (2) decode — memory bound dominates
                avg_len = sIn + sOut / 2.0
                mkv = kv_per_tok * avg_len * c
                tpot_p50 = (mw + mkv) / (GT["b_mem_eff_gbs"] * 1e9) + GT["alpha_dec"]
                # Add 5% noise
                ttft_p50 *= 1 + 0.05 * rng.standard_normal()
                tpot_p50 *= 1 + 0.05 * rng.standard_normal()
                throughput = c * sOut / (ttft_p50 + tpot_p50 * sOut)
                rows.append(dict(
                    input_len=sIn, output_len=sOut, concurrency=c, requests=max(50, c * 4),
                    ttft_p50=float(ttft_p50), ttft_p99=float(ttft_p50 * 1.4),
                    tpot_p50=float(tpot_p50), tpot_p99=float(tpot_p50 * 1.3),
                    throughput_tok_per_s=float(throughput),
                    peak_hbm_mib=int((mw + mkv) / 1024 / 1024 * 1.1),
                ))
    out_json.write_text(json.dumps(rows, indent=2))


def synthesize_power(out_csv: Path) -> None:
    """Mostly-decode samples at rho_dec*TDP plus some prefill at full TDP."""
    rng = np.random.default_rng(7)
    n = 4000
    is_prefill = rng.random(n) < 0.2
    sm_util = np.where(is_prefill, rng.uniform(80, 100, n), rng.uniform(60, 90, n))
    factor = np.where(is_prefill, 1.0, GT["rho_dec"])
    power = GT["p_idle"] + (GT["p_tdp"] - GT["p_idle"]) * (sm_util / 100.0) * factor
    power += rng.normal(0, 5, n)
    with open(out_csv, "w") as f:
        f.write("timestamp_unix_ms,gpu,power_w,mem_used_mib,sm_util_pct\n")
        for i in range(n):
            f.write(f"{i*50},0,{power[i]:.1f},60000,{int(sm_util[i])}\n")


def assert_close(name: str, got: float, expected: float, tol_rel: float):
    rel = abs(got - expected) / expected
    flag = "OK" if rel <= tol_rel else "FAIL"
    print(f"  [{flag}]  {name:25s}  got={got:9.3f}  expected={expected:9.3f}  rel_err={rel*100:5.2f}%   target<= {tol_rel*100:.0f}%")
    return rel <= tol_rel


def main():
    work = HERE / "_dry_run"; work.mkdir(exist_ok=True)
    meas = work / "measurements.json"
    power = work / "power.csv"
    out = work / "calibration.json"
    synthesize_measurements(meas)
    synthesize_power(power)

    # Ensure the pipeline finds the model/hw config files.
    cfg_dir = HERE / "configs"

    cmd = [
        sys.executable, str(HERE / "fit_calibration.py"),
        "--measurements", str(meas),
        "--power-csv",    str(power),
        "--model-config", str(cfg_dir / "llama3_8b.json"),
        "--hw-config",    str(cfg_dir / "a100_80gb.json"),
        "--out",          str(out),
    ]
    print("[run] " + " ".join(cmd))
    subprocess.run(cmd, check=True)

    fit = json.loads(out.read_text())
    print("\n[fit output]")
    print(json.dumps(fit, indent=2))

    print("\n[parameter recovery vs ground truth]")
    ok = []
    ok.append(assert_close("F_eff^pre (TFLOPS)", fit["f_eff_prefill_tflops"], GT["f_eff_pre_tflops"], 0.20))
    ok.append(assert_close("B_mem^eff (GB/s)",   fit["b_mem_eff_gbs"],         GT["b_mem_eff_gbs"], 0.10))
    ok.append(assert_close("alpha_pre (sec)",    fit["alpha_prefill_sec"],     GT["alpha_pre"], 0.50))
    ok.append(assert_close("alpha_dec (sec)",    fit["alpha_decode_sec"],      GT["alpha_dec"], 1.00))
    ok.append(assert_close("rho_dec",            fit["rho_decode"],            GT["rho_dec"], 0.20))
    print()
    if all(ok):
        print("[ok] calibration pipeline recovers ground truth within tolerances")
        sys.exit(0)
    else:
        print("[fail] some parameters drifted beyond tolerance")
        sys.exit(1)


if __name__ == "__main__":
    main()
