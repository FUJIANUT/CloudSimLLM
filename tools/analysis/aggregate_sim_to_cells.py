"""
Convert per-request simulator output (CSV from LlmStatistics.writeRecordsCsv)
into the per-(input_len, output_len, concurrency) cell summary that
validate_cli.py expects.

The mapping from request → cell comes from a sidecar JSON written by the Java
calibration driver, which records which (sIn, sOut, concur) cell each
request id belongs to.

Usage:
    python aggregate_sim_to_cells.py \
        --records sim_records_a100_llama3_8b.csv \
        --cell-map cell_map_a100_llama3_8b.json \
        --out sim_a100_llama3_8b.csv
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--cell-map", required=True, help="JSON: {request_id: [in, out, concur]}")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rec = pd.read_csv(args.records)
    cmap = json.load(open(args.cell_map))
    rec["input_len"]   = rec["id"].map(lambda i: cmap[str(i)][0])
    rec["output_len"]  = rec["id"].map(lambda i: cmap[str(i)][1])
    rec["concurrency"] = rec["id"].map(lambda i: cmap[str(i)][2])

    grouped = rec.groupby(["input_len", "output_len", "concurrency"])
    agg = grouped.apply(lambda g: pd.Series({
        "ttft_p50": np.median(g.ttft_sec),
        "ttft_p99": np.quantile(g.ttft_sec, 0.99) if len(g) >= 100 else g.ttft_sec.max(),
        "tpot_p50": np.median(g.tpot_sec),
        "tpot_p99": np.quantile(g.tpot_sec, 0.99) if len(g) >= 100 else g.tpot_sec.max(),
        "throughput_tok_per_s":
            g.output_len.sum() / max(1e-6, g.e2e_sec.max()),
    })).reset_index()

    agg.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(agg)} cells)")


if __name__ == "__main__":
    main()
