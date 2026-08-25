"""
Calibration step 3 — fit the six effective parameters needed by `GpuPe`:
    F_eff^pre   (TFLOPS effective during prefill)
    F_eff^dec   (TFLOPS effective during decode)
    B_mem^eff   (HBM bandwidth, GB/s)
    alpha_pre, alpha_dec   (kernel launch overheads, sec)
    rho_dec     (decode power discount factor, dimensionless)

Inputs:
    --measurements   measurements_*.json from run_vllm_benchmark.py
    --power-csv      power_*.csv from sample_power.py
    --model-config   model_config.json — { name, parameters, layers, h_kv, d_h, bytes }
    --hw-config      hw_config.json    — { peak_tflops, peak_hbm_gbs, hbm_gb, tdp_w, idle_w }

Output:
    calibration.json — to be loaded by Java side via GpuPe.set*() calls.
"""
from __future__ import annotations

import argparse
import json
import statistics

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# ----- Eq. (1) prefill model: T_pre(B) = (2 * P_m * sum_s) / F_eff^pre + alpha_pre
def prefill_fn(sum_s, f_eff_tflops, alpha_pre, p_m):
    return (2.0 * p_m * sum_s) / (f_eff_tflops * 1e12) + alpha_pre


# ----- Eq. (2) decode bandwidth-bound: T_dec ≈ (M_w + M_kv(B)) / B_mem^eff + alpha_dec
def decode_fn(state, b_mem_gbs, alpha_dec):
    mw, mkv = state
    return (mw + mkv) / (b_mem_gbs * 1e9) + alpha_dec


def kv_bytes_per_token(model: dict) -> int:
    return 2 * model["layers"] * model["h_kv"] * model["d_h"] * model["bytes"]


def fit_prefill(meas: pd.DataFrame, p_m: float):
    # Use concurrency=1 cells to avoid batch interleaving artifacts.
    df = meas[meas.concurrency == 1]
    sum_s = df.input_len.values.astype(float)
    t = df.ttft_p50.values
    if len(t) < 3:
        raise RuntimeError("Need ≥ 3 prefill measurement cells")
    # Constrain alpha ≥ 0 (kernel launch can't be negative).
    p, _ = curve_fit(
        lambda x, f, a: prefill_fn(x, f, a, p_m),
        sum_s, t, p0=[150.0, 0.005],
        bounds=([1.0, 0.0], [10000.0, 1.0]),
    )
    return float(p[0]), float(p[1])


def fit_decode(meas: pd.DataFrame, p_m: float, mw: int, kv_per_tok: int):
    rows = []
    for _, r in meas.iterrows():
        # Approximate average active KV across decode steps as B*(s + o/2)
        avg_len = r.input_len + r.output_len / 2.0
        mkv = kv_per_tok * avg_len * r.concurrency
        rows.append((mw, mkv, r.tpot_p50))
    if len(rows) < 3:
        raise RuntimeError("Need ≥ 3 decode measurement cells")
    arr = np.array(rows)
    state = (arr[:, 0], arr[:, 1])
    # Constrain alpha ≥ 0 to avoid noise being absorbed into a fictitious negative offset.
    p, _ = curve_fit(
        lambda x, b, a: decode_fn(x, b, a),
        state, arr[:, 2], p0=[1500.0, 0.001],
        bounds=([100.0, 0.0], [10000.0, 1.0]),
    )
    return float(p[0]), float(p[1])


def derive_decode_tflops(meas: pd.DataFrame, p_m: float):
    """Compute-bound check: 2*P_m*B / T_dec. Take robust max over cells."""
    cands = []
    for _, r in meas.iterrows():
        if r.tpot_p50 <= 0: continue
        cands.append((2.0 * p_m * r.concurrency) / (r.tpot_p50 * 1e12))
    return float(statistics.median(cands)) if cands else float("nan")


def fit_rho_dec(power: pd.DataFrame, hw: dict):
    """
    Estimate ρ_dec from the power distribution.

    For each sample with non-trivial utilization, compute the implied phase factor:
        factor = (P − P_idle) / ((P_tdp − P_idle) * sm_util/100)
    Prefill samples → factor ≈ 1.0; decode samples → factor ≈ ρ_dec.
    Take the lower mode of the resulting distribution.
    """
    p = power[power.sm_util_pct >= 50].copy()
    if p.empty:
        return 0.7

    util_frac = p.sm_util_pct.values / 100.0
    denom = (hw["tdp_w"] - hw["idle_w"]) * util_frac
    factor = ((p.power_w.values - hw["idle_w"]) / np.where(denom > 1e-6, denom, 1e-6))
    factor = np.clip(factor, 0.0, 1.5)

    if len(factor) < 50:
        return float(np.clip(np.median(factor), 0.4, 0.95))

    # Histogram in 0.05 bins over [0.3, 1.2]; pick the lower mode in [0.4, 0.85].
    bins = np.arange(0.30, 1.25, 0.05)
    hist, edges = np.histogram(factor, bins=bins)
    decode_mask = (edges[:-1] >= 0.40) & (edges[:-1] <= 0.85)
    if not decode_mask.any() or hist[decode_mask].sum() == 0:
        return float(np.clip(np.median(factor), 0.4, 0.95))
    masked = hist * decode_mask
    best = int(np.argmax(masked))
    centre = 0.5 * (edges[best] + edges[best + 1])
    return float(np.clip(centre, 0.4, 0.95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurements", required=True)
    ap.add_argument("--power-csv", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--hw-config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    meas = pd.DataFrame(json.load(open(args.measurements)))
    power = pd.read_csv(args.power_csv)
    model = json.load(open(args.model_config))
    hw = json.load(open(args.hw_config))

    p_m = float(model["parameters"])
    mw = int(p_m) * int(model["bytes"])
    kv_per_tok = kv_bytes_per_token(model)

    f_eff_pre, alpha_pre = fit_prefill(meas, p_m)
    b_mem_eff, alpha_dec = fit_decode(meas, p_m, mw, kv_per_tok)
    f_eff_dec = derive_decode_tflops(meas, p_m)
    rho_dec = fit_rho_dec(power, hw)

    out = {
        "model": model["name"],
        "hardware": hw.get("sku", "unknown"),
        "f_eff_prefill_tflops": round(f_eff_pre, 2),
        "f_eff_decode_tflops": round(f_eff_dec, 2),
        "b_mem_eff_gbs": round(b_mem_eff, 1),
        "alpha_prefill_sec": round(alpha_pre, 6),
        "alpha_decode_sec": round(alpha_dec, 6),
        "rho_decode": round(rho_dec, 3),
        "weight_bytes": mw,
        "kv_bytes_per_token": kv_per_tok,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
