"""
§6.4 Case Study 2 — Heterogeneous GPU mix sweep.

Drives `HeterogeneousMixRunner` (Java) over the (mix, policy, workload) grid.

Usage:
    python tools/experiments/run_heterogeneous_sweep.py \
        --output tools/experiments/results/heterogeneous_sweep.csv
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Per-workload arrival rates chosen so the colocated-8-A100 baseline sits at
# its SLO knee (~90-95%% attainment at the paper's 1/5/10 s TTFT thresholds):
# short 150 req/s, medium 40 req/s, long 10 req/s (see §6.2 probes).
WORKLOAD_RATES = {"short": 150.0, "medium": 40.0, "long": 10.0}



# Always 8 effective GPUs (rough TFLOPS budget): A100=180 / H100=450 / L40S=220.
# Mixes are chosen to cover the spectrum from homogeneous high-end to budget-blend.
MIXES_DEFAULT = [
    "a100x8",                       # homogeneous A100 baseline
    "h100x8",                       # homogeneous H100 baseline (premium)
    "l40sx8",                       # homogeneous L40S baseline (budget)
    "a100x4_h100x4",                # 50/50 A100+H100
    "a100x4_l40sx4",                # 50/50 A100+L40S
    "h100x2_a100x4_l40sx2",         # 3-tier blend (premium + mid + budget)
    "h100x4_l40sx4",                # 50/50 H100+L40S (skip mid tier)
]

POLICIES_DEFAULT  = ["FREE_HBM", "EST_TTFT"]
WORKLOADS_DEFAULT = ["short", "medium", "long"]


def get_classpath() -> tuple[str, str]:
    java_home = os.environ.get("JAVA_HOME")
    if not java_home:
        sys.exit("Set JAVA_HOME (e.g. export JAVA_HOME=$HOME/jdks/jdk-25.0.3+9/Contents/Home)")
    cp_file = REPO_ROOT / "target" / "classpath.txt"
    if not cp_file.exists():
        subprocess.run(
            ["./mvnw", "-q", "dependency:build-classpath", f"-Dmdep.outputFile={cp_file}"],
            cwd=REPO_ROOT, check=True,
            env={**os.environ, "JAVA_HOME": java_home})
    classes = REPO_ROOT / "target" / "classes"
    if not classes.exists():
        subprocess.run(["./mvnw", "-q", "compile", "-DskipTests"],
                       cwd=REPO_ROOT, check=True,
                       env={**os.environ, "JAVA_HOME": java_home})
    return java_home, f"{classes}:{cp_file.read_text().strip()}"


def run_one(java_home: str, classpath: str, args: list[str]) -> tuple[int, str]:
    cmd = [str(Path(java_home) / "bin" / "java"), "-cp", classpath,
           "org.cloudsimplus.llm.example.HeterogeneousMixRunner", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def build_grid(mixes, policies, workloads, seeds, requests):
    cells = []
    for workload in workloads:
        for mix in mixes:
            for policy in policies:
                for seed in seeds:
                    cells.append({
                        "mix": mix, "policy": policy, "workload": workload,
                        "requests": requests, "seed": seed,
                        "label": f"{mix}-{policy}-{workload}-s{seed}",
                    })
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="tools/experiments/results/heterogeneous_sweep.csv")
    ap.add_argument("--requests", type=int, default=500)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--rm", action="store_true")
    args = ap.parse_args()

    output = (REPO_ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.rm and output.exists():
        output.unlink()

    java_home, classpath = get_classpath()
    seeds = [int(s) for s in args.seeds.split(",")]
    cells = build_grid(MIXES_DEFAULT, POLICIES_DEFAULT, WORKLOADS_DEFAULT,
                       seeds, args.requests)
    print(f"[sweep] {len(cells)} cells × {args.jobs} jobs → {output}", flush=True)

    def to_args(cell):
        return [f"--mix={cell['mix']}", f"--policy={cell['policy']}",
                f"--requests={cell['requests']}", f"--workload={cell['workload']}",
                f"--rate={WORKLOAD_RATES[cell['workload']]}",
                f"--seed={cell['seed']}", f"--label={cell['label']}",
                f"--output={output}"]

    failed = 0
    t0 = time.perf_counter()
    if args.jobs == 1:
        for i, cell in enumerate(cells, 1):
            rc, out = run_one(java_home, classpath, to_args(cell))
            tag = "OK " if rc == 0 else "FAIL"
            line = next((l for l in out.splitlines() if l.startswith("[done]")), "")
            print(f"[{i:3d}/{len(cells)}] {tag} {cell['label']:50s} {line}", flush=True)
            if rc != 0: failed += 1
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futures = {ex.submit(run_one, java_home, classpath, to_args(c)): c for c in cells}
            done = 0
            for fut in as_completed(futures):
                done += 1
                cell = futures[fut]
                rc, out = fut.result()
                tag = "OK " if rc == 0 else "FAIL"
                line = next((l for l in out.splitlines() if l.startswith("[done]")), "")
                print(f"[{done:3d}/{len(cells)}] {tag} {cell['label']:50s} {line}", flush=True)
                if rc != 0: failed += 1

    dt = time.perf_counter() - t0
    print(f"\n[sweep] {len(cells) - failed}/{len(cells)} cells succeeded in {dt:.1f}s")
    if failed: sys.exit(1)


if __name__ == "__main__":
    main()
