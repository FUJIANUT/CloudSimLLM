"""
Calibration step 1 — sweep (input_len, output_len, batch_size) on a real vLLM
server and record TTFT / TPOT / throughput / peak HBM. Output: JSON consumed
by `fit_calibration.py`.

Prereq:
    pip install vllm openai pandas
    vllm serve meta-llama/Llama-3-8B-Instruct \
        --tensor-parallel-size 1 --gpu-memory-utilization 0.92

Usage:
    python run_vllm_benchmark.py \
        --base-url http://localhost:8000/v1 \
        --model meta-llama/Llama-3-8B-Instruct \
        --out measurements_a100_llama3_8b.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass

import openai


@dataclass
class Measurement:
    input_len: int
    output_len: int
    concurrency: int
    requests: int
    ttft_p50: float
    ttft_p99: float
    tpot_p50: float
    tpot_p99: float
    throughput_tok_per_s: float
    peak_hbm_mib: int


GRID_INPUT  = [256, 1024, 4096]
GRID_OUTPUT = [128, 512]
GRID_CONCUR = [1, 8, 32, 64]


def nvidia_smi_peak_hbm() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    ).decode().splitlines()
    return max(int(x) for x in out if x.strip())


async def one_request(client, model, prompt_tokens, max_tokens):
    prompt = "Hello world. " * (prompt_tokens // 3)
    t0 = time.perf_counter()
    first = None
    n_tok = 0
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=True,
        # Force exactly max_tokens output tokens so the measured cell matches
        # the nominal (input, output) grid point; without this the model stops
        # at EOS after a handful of tokens on synthetic prompts and TPOT /
        # throughput no longer correspond to the nominal operating point.
        extra_body={"ignore_eos": True},
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            n_tok += 1
            if first is None:
                first = time.perf_counter()
    end = time.perf_counter()
    if first is None or n_tok < 2:
        return None
    ttft = first - t0
    tpot = (end - first) / max(1, n_tok - 1)
    return ttft, tpot, n_tok


async def sweep_one(client, model, sIn, sOut, concur, requests):
    sem = asyncio.Semaphore(concur)
    ttfts, tpots, toks = [], [], 0
    t_start = time.perf_counter()

    async def _worker():
        async with sem:
            r = await one_request(client, model, sIn, sOut)
            if r:
                ttfts.append(r[0]); tpots.append(r[1])
                return r[2]
            return 0

    counts = await asyncio.gather(*[_worker() for _ in range(requests)])
    elapsed = time.perf_counter() - t_start
    if not ttfts:
        return None
    return Measurement(
        input_len=sIn,
        output_len=sOut,
        concurrency=concur,
        requests=requests,
        ttft_p50=statistics.median(ttfts),
        ttft_p99=statistics.quantiles(ttfts, n=100)[98] if len(ttfts) >= 100 else max(ttfts),
        tpot_p50=statistics.median(tpots),
        tpot_p99=statistics.quantiles(tpots, n=100)[98] if len(tpots) >= 100 else max(tpots),
        throughput_tok_per_s=sum(counts) / elapsed,
        peak_hbm_mib=nvidia_smi_peak_hbm(),
    )


async def main_async(args):
    client = openai.AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")
    out = []
    for sIn in GRID_INPUT:
        for sOut in GRID_OUTPUT:
            for concur in GRID_CONCUR:
                reqs = max(50, concur * 4)
                print(f"[run] in={sIn} out={sOut} concur={concur} reqs={reqs}", flush=True)
                m = await sweep_one(client, args.model, sIn, sOut, concur, reqs)
                if m:
                    out.append(asdict(m))
                    print(f"      ttft50={m.ttft_p50:.3f}s tpot50={m.tpot_p50*1000:.1f}ms tput={m.throughput_tok_per_s:.0f}t/s")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] wrote {args.out} ({len(out)} cells)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    asyncio.run(main_async(p.parse_args()))
