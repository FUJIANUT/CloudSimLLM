"""
§6.3 Case Study 1 — Splitwise vs co-located parametric sweep.

Drives `SplitwiseSweepRunner` (Java) over the (P:D ratio, workload shape,
KV bandwidth) grid. Each cell appends one row to the output CSV.

Usage:
    cd <repo root>
    python tools/experiments/run_splitwise_sweep.py \
        --output tools/experiments/results/splitwise_sweep.csv

Override:
    --total-gpus 8        total GPU budget held constant
    --requests   500
    --seeds      "42,43,44"  multiple seeds for confidence intervals
    --jobs       4        parallel cells
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


# (prefill_gpus, decode_gpus) — keep total fixed at TOTAL_GPUS in main()
PD_RATIOS_DEFAULT = [(1, 7), (2, 6), (3, 5), (4, 4), (5, 3), (6, 2)]
WORKLOADS_DEFAULT = ["short", "medium", "long"]
KV_BWS_DEFAULT    = [100.0, 200.0, 400.0]


def java_cmd(java_home: str, classpath: str, args: list[str]) -> list[str]:
    java_bin = Path(java_home) / "bin" / "java"
    return [
        str(java_bin), "-cp", classpath,
        "org.cloudsimplus.llm.example.SplitwiseSweepRunner",
        *args,
    ]


def get_classpath() -> tuple[str, str]:
    """Resolve JAVA_HOME and classpath via mvnw."""
    java_home = os.environ.get("JAVA_HOME")
    if not java_home:
        sys.exit("Set JAVA_HOME (e.g. export JAVA_HOME=$HOME/jdks/jdk-25.0.3+9/Contents/Home)")

    cp_file = REPO_ROOT / "target" / "classpath.txt"
    if not cp_file.exists():
        print("[setup] resolving classpath via mvnw…", flush=True)
        subprocess.run(
            ["./mvnw", "-q", "dependency:build-classpath",
             f"-Dmdep.outputFile={cp_file}"],
            cwd=REPO_ROOT, check=True,
            env={**os.environ, "JAVA_HOME": java_home},
        )
    classes = REPO_ROOT / "target" / "classes"
    if not classes.exists():
        print("[setup] running mvnw compile…", flush=True)
        subprocess.run(["./mvnw", "-q", "compile", "-DskipTests"],
                       cwd=REPO_ROOT, check=True,
                       env={**os.environ, "JAVA_HOME": java_home})
    classpath = f"{classes}:{cp_file.read_text().strip()}"
    return java_home, classpath


def run_one(java_home: str, classpath: str, args: list[str]) -> tuple[int, str]:
    cmd = java_cmd(java_home, classpath, args)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def build_grid(pd_ratios, workloads, kv_bws, seeds, total_gpus, requests):
    cells = []
    for workload in workloads:
        # baseline once per workload (KV bw and P:D irrelevant)
        for seed in seeds:
            cells.append({
                "mode": "colocated",
                "prefill_gpus": total_gpus,
                "decode_gpus": 0,
                "kv_bw_gbs": 200.0,
                "workload": workload,
                "requests": requests,
                "seed": seed,
                "label": f"baseline-{workload}-s{seed}",
            })
        # splitwise grid
        for kv_bw in kv_bws:
            for (p, d) in pd_ratios:
                if p + d != total_gpus:
                    continue
                for seed in seeds:
                    cells.append({
                        "mode": "splitwise",
                        "prefill_gpus": p,
                        "decode_gpus": d,
                        "kv_bw_gbs": kv_bw,
                        "workload": workload,
                        "requests": requests,
                        "seed": seed,
                        "label": f"sw-{workload}-{p}p{d}d-kv{int(kv_bw)}-s{seed}",
                    })
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="tools/experiments/results/splitwise_sweep.csv")
    ap.add_argument("--total-gpus", type=int, default=8)
    ap.add_argument("--requests", type=int, default=500)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--jobs", type=int, default=1, help="parallel cells")
    ap.add_argument("--rm", action="store_true", help="remove existing output before sweep")
    args = ap.parse_args()

    output = (REPO_ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.rm and output.exists():
        output.unlink()

    java_home, classpath = get_classpath()
    seeds = [int(s) for s in args.seeds.split(",")]

    cells = build_grid(PD_RATIOS_DEFAULT, WORKLOADS_DEFAULT, KV_BWS_DEFAULT,
                       seeds, args.total_gpus, args.requests)
    print(f"[sweep] {len(cells)} cells × seed × {args.jobs} jobs → {output}", flush=True)

    def to_args(cell: dict) -> list[str]:
        return [
            f"--mode={cell['mode']}",
            f"--prefill-gpus={cell['prefill_gpus']}",
            f"--decode-gpus={cell['decode_gpus']}",
            f"--requests={cell['requests']}",
            f"--workload={cell['workload']}",
            f"--kv-bw-gbs={cell['kv_bw_gbs']}",
            f"--seed={cell['seed']}",
            f"--label={cell['label']}",
            f"--output={output}",
        ]

    failed = 0
    t0 = time.perf_counter()
    if args.jobs == 1:
        for i, cell in enumerate(cells, 1):
            rc, out = run_one(java_home, classpath, to_args(cell))
            tag = "OK " if rc == 0 else "FAIL"
            line = next((l for l in out.splitlines() if l.startswith("[done]")), out.splitlines()[-1] if out else "")
            print(f"[{i:3d}/{len(cells)}] {tag} {cell['label']:40s} {line}", flush=True)
            if rc != 0:
                failed += 1
                if failed > 5:
                    sys.exit("[abort] too many failures; aborting sweep")
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futures = {ex.submit(run_one, java_home, classpath, to_args(cell)): cell for cell in cells}
            done = 0
            for fut in as_completed(futures):
                done += 1
                cell = futures[fut]
                try:
                    rc, out = fut.result()
                except Exception as e:
                    rc, out = 1, str(e)
                tag = "OK " if rc == 0 else "FAIL"
                line = next((l for l in out.splitlines() if l.startswith("[done]")), "")
                print(f"[{done:3d}/{len(cells)}] {tag} {cell['label']:40s} {line}", flush=True)
                if rc != 0:
                    failed += 1

    dt = time.perf_counter() - t0
    print(f"\n[sweep] {len(cells) - failed}/{len(cells)} cells succeeded in {dt:.1f}s")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
