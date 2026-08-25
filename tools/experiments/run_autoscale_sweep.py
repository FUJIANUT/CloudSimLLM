"""
§6.6 Case Study 4 — Bursty workload auto-scaling sweep.
"""
from __future__ import annotations

import argparse, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

POLICIES_DEFAULT  = ["STATIC", "REACTIVE", "PREDICTIVE"]
BURST_DEFAULT     = ["low", "med", "high"]
STARTUP_DEFAULT   = [5, 15, 30]   # seconds (HF Spaces, Lambda, AWS Lambda warmpool)


def get_classpath():
    java_home = os.environ.get("JAVA_HOME")
    if not java_home:
        sys.exit("Set JAVA_HOME")
    cp_file = REPO_ROOT / "target" / "classpath.txt"
    if not cp_file.exists():
        subprocess.run(["./mvnw", "-q", "dependency:build-classpath", f"-Dmdep.outputFile={cp_file}"],
                       cwd=REPO_ROOT, check=True, env={**os.environ, "JAVA_HOME": java_home})
    classes = REPO_ROOT / "target" / "classes"
    if not classes.exists():
        subprocess.run(["./mvnw", "-q", "compile", "-DskipTests"],
                       cwd=REPO_ROOT, check=True, env={**os.environ, "JAVA_HOME": java_home})
    return java_home, f"{classes}:{cp_file.read_text().strip()}"


def run_one(java_home, classpath, args):
    cmd = [str(Path(java_home) / "bin" / "java"), "-cp", classpath,
           "org.cloudsimplus.llm.example.AutoscalingRunner", *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="tools/experiments/results/autoscale_sweep.csv")
    ap.add_argument("--requests", type=int, default=300)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--max-pool", type=int, default=8)
    ap.add_argument("--min-pool", type=int, default=2)
    ap.add_argument("--initial-pool", type=int, default=4)
    ap.add_argument("--base-rate", type=float, default=8.0,
                    help="Base arrival rate (req/s) before burst multiplier.")
    ap.add_argument("--ttft-slo", type=float, default=5.0,
                    help="Per-cloudlet TTFT SLO threshold (s). Default 5.0 "
                         "matches the medium-workload guidance.")
    ap.add_argument("--bursts", default=",".join(BURST_DEFAULT),
                    help="Comma-separated burst levels (subset of low,med,high).")
    ap.add_argument("--startups", default=",".join(str(s) for s in STARTUP_DEFAULT),
                    help="Comma-separated cold-start delays in seconds.")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--rm", action="store_true")
    args = ap.parse_args()

    output = (REPO_ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.rm and output.exists():
        output.unlink()

    java_home, classpath = get_classpath()
    seeds = [int(s) for s in args.seeds.split(",")]

    cells = []
    bursts = [b.strip() for b in args.bursts.split(",") if b.strip()]
    startups = [int(s) for s in args.startups.split(",") if s.strip()]
    for policy in POLICIES_DEFAULT:
        for burst in bursts:
            for startup in startups:
                for seed in seeds:
                    cells.append({
                        "policy": policy, "burst": burst,
                        "startup": startup, "requests": args.requests,
                        "max_pool": args.max_pool, "min_pool": args.min_pool,
                        "initial_pool": args.initial_pool, "seed": seed,
                        "label": f"{policy}-{burst}-s{startup}-seed{seed}",
                    })
    print(f"[sweep] {len(cells)} cells × {args.jobs} jobs → {output}", flush=True)

    def to_args(cell):
        return [
            f"--policy={cell['policy']}", f"--burst={cell['burst']}",
            f"--startup-sec={cell['startup']}",
            f"--max-pool={cell['max_pool']}", f"--min-pool={cell['min_pool']}",
            f"--initial-pool={cell['initial_pool']}",
            f"--requests={cell['requests']}", f"--workload=medium",
            f"--base-rate={args.base_rate}",
            f"--ttft-slo={args.ttft_slo}",
            f"--seed={cell['seed']}", f"--label={cell['label']}",
            f"--output={output}",
        ]

    failed = 0; t0 = time.perf_counter()
    if args.jobs == 1:
        for i, cell in enumerate(cells, 1):
            try:
                proc = run_one(java_home, classpath, to_args(cell))
                rc, out = proc.returncode, (proc.stdout or "") + (proc.stderr or "")
            except subprocess.TimeoutExpired:
                rc, out = 124, "TIMEOUT"
            tag = "OK " if rc == 0 else "FAIL"
            line = next((l for l in out.splitlines() if l.startswith("[done]")), out[:100])
            print(f"[{i:3d}/{len(cells)}] {tag} {cell['label']:40s} {line}", flush=True)
            if rc != 0: failed += 1
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            fut2cell = {ex.submit(run_one, java_home, classpath, to_args(c)): c for c in cells}
            done = 0
            for fut in as_completed(fut2cell):
                done += 1
                c = fut2cell[fut]
                try: rc, out = fut.result().returncode, fut.result().stdout + fut.result().stderr
                except Exception as e: rc, out = 1, str(e)
                tag = "OK " if rc == 0 else "FAIL"
                line = next((l for l in out.splitlines() if l.startswith("[done]")), "")
                print(f"[{done:3d}/{len(cells)}] {tag} {c['label']:40s} {line}", flush=True)
                if rc != 0: failed += 1

    dt = time.perf_counter() - t0
    print(f"\n[sweep] {len(cells) - failed}/{len(cells)} cells succeeded in {dt:.1f}s")
    if failed: sys.exit(1)


if __name__ == "__main__":
    main()
