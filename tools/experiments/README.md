# §6.3 Case Study 1 — experiment harness

Reproduces all data and figures for the **Splitwise vs co-located prefill/decode disaggregation** case study.

## Pipeline

```
                                         ┌────── Fig 6 ─────┐
                                         │                  │
   run_splitwise_sweep.py                ├─── Fig 7 ────────┤
       │                                 │                  │
       ▼                                 ├─── Fig 8 ────────┤
  SplitwiseSweepRunner.java              │                  │
       │       (×57 cells)               └─ table_*.tex ────┘
       ▼                                          ▲
  splitwise_sweep.csv  ───────────► case_study_1_cli.py
                                    (or case_study_1.ipynb)
```

## Quick start

```bash
# 1. Set JDK 25
export JAVA_HOME=$HOME/jdks/jdk-25.0.3+9/Contents/Home

# 2. Run the sweep (~17s wall on a laptop)
python tools/experiments/run_splitwise_sweep.py \
    --output tools/experiments/results/splitwise_sweep.csv \
    --requests 500

# 3. Generate figures + summary table
cd tools/analysis
python case_study_1_cli.py \
    --results ../experiments/results/splitwise_sweep.csv \
    --outdir figures/

# 4. (optional) interactive exploration
jupyter lab case_study_1.ipynb
```

## Sweep grid (default)

| Axis | Values |
|---|---|
| P:D ratio (total = 8 GPUs) | 1:7, 2:6, 3:5, 4:4, 5:3, 6:2 |
| Workload shape | short / medium / long |
| KV bandwidth | 100 / 200 / 400 GB/s |
| Co-located baseline | 1 per workload |
| **Total cells** | **6 × 3 × 3 + 3 = 57** |

Each cell runs in ~140ms wall on a laptop; total sweep ≈ 17s.

## Workload shapes

| Shape | Prompt tokens | Response tokens |
|---|---|---|
| short  | 128–512     | 64–192   |
| medium | 512–2048    | 256–640  |
| long   | 2048–8192   | 128–512  |

Poisson arrival rate is 50 req/s for all shapes.

## Known limitations (write into §8)

1. **Partial completion at extreme P:D ratios.** Cells where decode capacity is severely
   constrained (e.g., 2:6 short workload) leave 30–50% of decode shadows un-finished
   when CloudSim's idle-detection terminates the simulation. This affects only the
   *short workload, low-prefill-share* corner of Fig 7; the trend lines remain
   monotonic. A future fix is to set an explicit
   `simulation.terminateAt(largeT)` or wake the broker via a periodic timer.
2. **Energy attribution coarseness.** Per-request energy uses a fleet-average power;
   a per-tick `PowerMeter` integration would tighten the carbon column of the
   summary table.
3. **No KV cache mirroring.** The Splitwise paper's "continuous prefix caching"
   optimization is not modeled — Splitwise numbers in §6.3 are conservative.

## Files

| Path | Purpose |
|---|---|
| `run_splitwise_sweep.py` | Python driver; iterates the grid |
| `results/splitwise_sweep.csv` | Output of all 57 cells |
| `../analysis/case_study_1.ipynb` | Interactive analysis |
| `../analysis/case_study_1_cli.py` | Headless analysis (CI-safe) |
| `../analysis/figures/fig6_*.pdf` | TTFT P99 vs P:D ratio |
| `../analysis/figures/fig7_*.pdf` | Throughput–latency Pareto |
| `../analysis/figures/fig8_*.pdf` | KV bandwidth sensitivity |
| `../analysis/figures/table_case_study_1.tex` | Summary table |
