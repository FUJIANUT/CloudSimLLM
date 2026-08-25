"""
§6.5 Case Study 3 — Carbon-aware geo-distributed routing sweep.

Drives `GeoDistributedRunner` (Java) over (policy × hour-of-day × workload).
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

# 12 GPUs across 3 regions; LATENCY_GREEDY splits load 3 ways onto 4-GPU
# regions, so per-GPU utilisation parity with the 8-GPU §6.3 baseline means
# 1.5x the §6.3 per-workload rates.
WORKLOAD_RATES = {"short": 90.0, "medium": 24.0, "long": 6.0}


POLICIES_DEFAULT  = ["LATENCY_GREEDY", "CARBON_AWARE", "BLENDED"]
HOURS_DEFAULT     = list(range(0, 24, 3))           # 0, 3, 6, …, 21
WORKLOADS_DEFAULT = ["short", "medium", "long"]


def get_classpath():
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


def run_one(java_home, classpath, args):
    cmd = [str(Path(java_home) / "bin" / "java"), "-cp", classpath,
           "org.cloudsimplus.llm.example.GeoDistributedRunner", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def build_grid(policies, hours, workloads, seeds, requests, lambda_):
    cells = []
    for w in workloads:
        for h in hours:
            for p in policies:
                for s in seeds:
                    cells.append({
                        "policy": p, "hour": h, "workload": w,
                        "requests": requests, "seed": s, "lambda": lambda_,
                        "label": f"{p}-h{h}-{w}-s{s}",
                    })
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="tools/experiments/results/geo_sweep.csv")
    ap.add_argument("--requests", type=int, default=500)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--lambda", dest="lambda_", default="0.005")
    ap.add_argument("--carbon-csv", default="",
                    help="Optional 24h carbon-intensity CSV "
                         "(e.g. tools/data/carbon_profiles_2026.csv). "
                         "If empty, the analytic profiles in GeoRegion are used.")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--rm", action="store_true")
    args = ap.parse_args()

    output = (REPO_ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.rm and output.exists():
        output.unlink()

    java_home, classpath = get_classpath()
    seeds = [int(s) for s in args.seeds.split(",")]
    cells = build_grid(POLICIES_DEFAULT, HOURS_DEFAULT, WORKLOADS_DEFAULT,
                       seeds, args.requests, args.lambda_)
    print(f"[sweep] {len(cells)} cells × {args.jobs} jobs → {output}", flush=True)

    def to_args(cell):
        a = [f"--policy={cell['policy']}", f"--hour={cell['hour']}",
             f"--requests={cell['requests']}", f"--workload={cell['workload']}",
             f"--rate={WORKLOAD_RATES[cell['workload']]}",
             f"--seed={cell['seed']}", f"--lambda={cell['lambda']}",
             f"--label={cell['label']}", f"--output={output}"]
        if args.carbon_csv:
            a.append(f"--carbon-csv={Path(args.carbon_csv).resolve()}")
        return a

    failed = 0; t0 = time.perf_counter()
    if args.jobs == 1:
        for i, cell in enumerate(cells, 1):
            rc, out = run_one(java_home, classpath, to_args(cell))
            tag = "OK " if rc == 0 else "FAIL"
            line = next((l for l in out.splitlines() if l.startswith("[done]")), "")
            print(f"[{i:3d}/{len(cells)}] {tag} {cell['label']:35s} {line}", flush=True)
            if rc != 0: failed += 1
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            fut2cell = {ex.submit(run_one, java_home, classpath, to_args(c)): c for c in cells}
            done = 0
            for fut in as_completed(fut2cell):
                done += 1
                c = fut2cell[fut]; rc, out = fut.result()
                tag = "OK " if rc == 0 else "FAIL"
                line = next((l for l in out.splitlines() if l.startswith("[done]")), "")
                print(f"[{done:3d}/{len(cells)}] {tag} {c['label']:35s} {line}", flush=True)
                if rc != 0: failed += 1

    dt = time.perf_counter() - t0
    print(f"\n[sweep] {len(cells) - failed}/{len(cells)} cells succeeded in {dt:.1f}s")
    if failed: sys.exit(1)


if __name__ == "__main__":
    main()
