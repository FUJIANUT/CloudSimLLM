"""
Calibration step 2 — background nvidia-smi power sampler. Run alongside
`run_vllm_benchmark.py` and stop when benchmarks complete.

Output: CSV `timestamp_unix_ms, gpu_index, power_w, mem_used_mib, sm_util_pct`.

Usage:
    python sample_power.py --interval-ms 50 --out power_a100.csv
    # then in another shell:
    python run_vllm_benchmark.py ... --out measurements.json
    # finally Ctrl-C this sampler
"""
from __future__ import annotations

import argparse
import csv
import signal
import subprocess
import sys
import time

QUERIES = "timestamp,index,power.draw,memory.used,utilization.gpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval-ms", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    f = open(args.out, "w", newline="")
    w = csv.writer(f)
    w.writerow(["timestamp_unix_ms", "gpu", "power_w", "mem_used_mib", "sm_util_pct"])

    def _stop(*_):
        f.flush(); f.close()
        sys.exit(0)
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    cmd = [
        "nvidia-smi",
        f"--query-gpu={QUERIES}",
        "--format=csv,noheader,nounits",
        f"-lms={args.interval_ms}",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1)
    print(f"[sampler] writing to {args.out} every {args.interval_ms}ms; Ctrl-C to stop")
    try:
        for line in proc.stdout:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5: continue
            ts = int(time.time() * 1000)
            try:
                gpu = int(parts[1])
                power = float(parts[2])
                mem = int(float(parts[3]))
                util = int(float(parts[4]))
            except ValueError:
                continue
            w.writerow([ts, gpu, power, mem, util])
    finally:
        _stop()


if __name__ == "__main__":
    main()
