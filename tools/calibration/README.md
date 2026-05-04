# CloudSimLLM Calibration (§6.1)

Three steps to fill the **calibration table** in §A.3 of the paper.

## 0. Prerequisites

```bash
# On the GPU box:
pip install vllm openai pandas numpy scipy
sudo apt install -y nvidia-utils-535      # for nvidia-smi
```

## 1. Start the vLLM server (one model at a time)

```bash
vllm serve meta-llama/Llama-3-8B-Instruct \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.92
```

## 2. Collect measurements

Run these in **two separate terminals**:

```bash
# Terminal A — power sampler (background)
python sample_power.py --interval-ms 50 --out power_a100_llama3_8b.csv

# Terminal B — workload sweep
python run_vllm_benchmark.py \
    --base-url http://localhost:8000/v1 \
    --model meta-llama/Llama-3-8B-Instruct \
    --out measurements_a100_llama3_8b.json
```

When the workload sweep finishes, Ctrl-C the sampler.

## 3. Fit and emit calibration JSON

```bash
python fit_calibration.py \
    --measurements measurements_a100_llama3_8b.json \
    --power-csv    power_a100_llama3_8b.csv \
    --model-config configs/llama3_8b.json \
    --hw-config    configs/a100_80gb.json \
    --out          calibration_a100_llama3_8b.json
```

Sample output:

```json
{
  "model": "llama-3-8b",
  "hardware": "A100-SXM4-80GB",
  "f_eff_prefill_tflops": 182.34,
  "f_eff_decode_tflops": 58.11,
  "b_mem_eff_gbs": 1492.6,
  "alpha_prefill_sec": 0.0048,
  "alpha_decode_sec": 0.0011,
  "rho_decode": 0.69
}
```

## 4. Plug into Java (one-liner)

```java
GpuPe gpu = GpuPe.a100_80gb()
    .setEffFp16TflopsPrefill(182.34)
    .setEffFp16TflopsDecode(58.11)
    .setEffHbmBwGbs(1492.6)
    .setAlphaPrefillSec(0.0048)
    .setAlphaDecodeSec(0.0011);
LlmPowerModel pm = new LlmPowerModel().setDecodeDiscount(0.69);
```

## 5. Repeat for every (model, hardware) pair

The §A.3 table needs at least 3 pairs:
- A100-80GB / Llama-3-8B fp16
- H100-80GB / Llama-3-70B (TP=4) fp16
- L40S      / Llama-3-8B fp8
