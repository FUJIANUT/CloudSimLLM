"""
§6.7 Case Study 5 — DynamoLLM-inspired instance-pool routing sweep.

Drives `PoolRoutingRunner` (Java) over (config × routing × arrival rate)
at a fixed 150 s observation horizon. Each cell appends one CSV row.

Usage:
    python tools/experiments/run_pool_sweep.py \
        --output tools/experiments/results/pool_sweep.csv \
        --seeds "42,...,61" --jobs 4
"""
from __future__ import annotations

import argparse, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# (config, routing) combos — mixed is evaluated with and without size-aware
# routing; uniform configs are routing-insensitive (round-robin).
COMBOS = [
    ("tp1x8", "ROUND_ROBIN"),
    ("tp4x2", "ROUND_ROBIN"),
    ("mixed", "SIZE_AWARE"),
    ("mixed", "ROUND_ROBIN"),
]
RATES_DEFAULT = [5, 10, 15, 20]


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
           "org.cloudsimplus.llm.example.PoolRoutingRunner", *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="tools/experiments/results/pool_sweep.csv")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--rates", default=",".join(str(r) for r in RATES_DEFAULT))
    ap.add_argument("--horizon-sec", type=float, default=150.0)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--rm", action="store_true")
    args = ap.parse_args()

    output = (REPO_ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.rm and output.exists():
        output.unlink()

    java_home, classpath = get_classpath()
    seeds = [int(s) for s in args.seeds.split(",")]
    rates = [float(r) for r in args.rates.split(",")]

    cells = []
    for config, routing in COMBOS:
        for rate in rates:
            for seed in seeds:
                cells.append({
                    "config": config, "routing": routing, "rate": rate, "seed": seed,
                    "label": f"{config}-{routing}-r{rate:g}-s{seed}",
                })
    print(f"[sweep] {len(cells)} cells × {args.jobs} jobs → {output}", flush=True)

    def to_args(cell):
        return [f"--config={cell['config']}", f"--routing={cell['routing']}",
                f"--rate={cell['rate']}", f"--horizon-sec={args.horizon_sec}",
                f"--seed={cell['seed']}", f"--label={cell['label']}",
                f"--output={output}"]

    failed = 0; t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_one, java_home, classpath, to_args(c)): c for c in cells}
        for i, fut in enumerate(as_completed(futs), 1):
            cell = futs[fut]
            try:
                r = fut.result()
                ok = r.returncode == 0
            except Exception:
                ok = False
            if not ok:
                failed += 1
                print(f"[{i}/{len(cells)}] FAIL {cell['label']}", flush=True)
            elif i % 40 == 0:
                print(f"[{i}/{len(cells)}] ... {cell['label']}", flush=True)

    print(f"[sweep] {len(cells)-failed}/{len(cells)} cells succeeded in {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
