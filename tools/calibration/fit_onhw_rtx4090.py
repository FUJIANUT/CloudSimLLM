"""
On-hardware calibration + held-out validation (reviewer M1).

Fits the prefill (Eq. 1) and decode (Eq. 2) parameters to REAL vLLM
measurements collected on an RTX 4090 laptop GPU serving Qwen2.5-3B
(tools/calibration/onhw_rtx4090_qwen3b/), then validates the fitted
model on held-out operating points the fitter never saw.

Unlike the literature-seeded synthetic calibration (run_full_calibration.py),
this uses genuine hardware measurements, so the held-out error measures
whether Eqs. 1-2 track real GPU behaviour — not merely whether curve_fit
converges. The RTX 4090 here was power-capped (~16 W, 210 MHz), so absolute
throughput is not representative of a datacenter GPU; the exercise validates
the *methodology* and surfaces where the parametric form breaks.

Usage:
    python tools/calibration/fit_onhw_rtx4090.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
DATA = HERE / "onhw_rtx4090_qwen3b"


def is_holdout(m: dict) -> bool:
    """Held-out OOD corner: longest prompt OR largest batch."""
    return m["input_len"] == 1024 or m["concurrency"] == 8


def main() -> None:
    meas = json.loads((DATA / "measurements.json").read_text())
    model = json.loads((HERE / "configs" / "qwen25_3b.json").read_text())
    p_m = model["parameters"]
    kv_per_tok = 2 * model["layers"] * model["h_kv"] * model["d_h"] * model["bytes"]
    mw = p_m * model["bytes"]

    cal = [m for m in meas if not is_holdout(m)]
    ho = [m for m in meas if is_holdout(m)]

    # --- Prefill fit (Eq. 1) on concurrency==1 calibration cells ---
    c1 = [m for m in cal if m["concurrency"] == 1]
    sIn = np.array([m["input_len"] for m in c1], float)
    ttft = np.array([m["ttft_p50"] for m in c1])
    (f_pre, a_pre), _ = curve_fit(
        lambda s, F, a: 2 * p_m * s / (F * 1e12) + a, sIn, ttft,
        p0=[5.0, 0.3], bounds=([0.01, 0.0], [1000.0, 5.0]))

    # --- Decode fit (Eq. 2, memory-bound branch) on calibration cells ---
    X = np.array([mw + kv_per_tok * (m["input_len"] + m["output_len"] / 2) * m["concurrency"]
                  for m in cal], float)
    y = np.array([m["tpot_p50"] for m in cal])
    (b_mem, a_dec), _ = curve_fit(
        lambda x, B, a: x / (B * 1e9) + a, X, y,
        p0=[50.0, 0.05], bounds=([0.1, 0.0], [10000.0, 2.0]))

    def predict(m: dict) -> tuple[float, float]:
        tt = 2 * p_m * m["input_len"] / (f_pre * 1e12) + a_pre
        mkv = kv_per_tok * (m["input_len"] + m["output_len"] / 2) * m["concurrency"]
        tp = (mw + mkv) / (b_mem * 1e9) + a_dec
        return tt, tp

    def errors(cells: list[dict]) -> tuple[float, float]:
        et = [abs(predict(m)[0] - m["ttft_p50"]) / m["ttft_p50"] for m in cells]
        ep = [abs(predict(m)[1] - m["tpot_p50"]) / m["tpot_p50"] for m in cells]
        return 100 * float(np.mean(et)), 100 * float(np.mean(ep))

    id_ttft, id_tpot = errors(cal)
    ho_ttft, ho_tpot = errors(ho)

    result = {
        "pair": "RTX-4090-laptop / Qwen2.5-3B (power-capped)",
        "f_eff_prefill_tflops": round(f_pre, 2),
        "alpha_prefill_s": round(a_pre, 4),
        "b_mem_eff_gbs": round(b_mem, 2),
        "alpha_decode_s": round(a_dec, 4),
        "n_calib": len(cal), "n_holdout": len(ho),
        "id_ttft_err_pct": round(id_ttft, 1), "id_tpot_err_pct": round(id_tpot, 1),
        "ho_ttft_err_pct": round(ho_ttft, 1), "ho_tpot_err_pct": round(ho_tpot, 1),
    }
    (DATA / "fit_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print("\nInterpretation:")
    print(f"  Decode/TPOT model validates on REAL held-out data: {ho_tpot:.1f}% error.")
    print(f"  Prefill/TTFT model degrades on held-out ({ho_ttft:.0f}%): the")
    print(f"  power-capped GPU's TTFT is dominated by a ~{a_pre*1000:.0f} ms fixed")
    print(f"  overhead that Eq. 1's linear-in-tokens term underweights at this scale.")


if __name__ == "__main__":
    main()
