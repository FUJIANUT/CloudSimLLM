# Validation analysis (§6.1)

Pipeline:

```
real measurements          simulator records
(measurements_*.json)      (sim_records_*.csv  +  cell_map_*.json)
        |                              |
        |                              v
        |                  aggregate_sim_to_cells.py
        |                              |
        |                              v
        |                       sim_*.csv  (per-cell)
        v                              v
        +------------ validate_cli.py -+
                       |
                       v
              figures/  (PDF + PNG + LaTeX table)
              + pass/fail exit code
```

## End-to-end run

```bash
# 0. produce real measurements (vLLM)
python ../calibration/run_vllm_benchmark.py \
    --base-url http://localhost:8000/v1 \
    --model meta-llama/Llama-3-8B-Instruct \
    --out measurements_a100_llama3_8b.json

# 1. produce simulator output (Java driver mirrors the same grid)
./mvnw exec:java \
    -Dexec.mainClass=org.cloudsimplus.llm.example.LlmCalibrationDriver \
    -Dexec.args="--calibration calibration_a100_llama3_8b.json \
                 --grid-from measurements_a100_llama3_8b.json \
                 --records-out sim_records_a100_llama3_8b.csv \
                 --cell-map-out cell_map_a100_llama3_8b.json"

# 2. aggregate per-record to per-cell
python aggregate_sim_to_cells.py \
    --records sim_records_a100_llama3_8b.csv \
    --cell-map cell_map_a100_llama3_8b.json \
    --out sim_a100_llama3_8b.csv

# 3. validate against real measurements; fail CI if errors > targets
python validate_cli.py \
    --real measurements_a100_llama3_8b.json \
    --sim  sim_a100_llama3_8b.csv \
    --outdir figures/
```

## Output files

| File | Used in paper |
|---|---|
| `figures/table_validation_errors.tex` | §6.1 main error table |
| `figures/fig_qq_validation.pdf`       | §6.1 Fig. — Q-Q distribution agreement |
| `figures/fig_residual_heatmap.pdf`    | §6.1 Fig. — residual heatmap |

## Notes

- `LlmCalibrationDriver` is **not yet written** — it's a small driver loop that
  iterates the (input_len, output_len, concurrency) grid and emits per-cell
  request id mappings. Add it next; it shares structure with `LlmExample.java`.
- `validate_cli.py` exits non-zero if any metric exceeds its target relative
  error, so it can gate CI.
