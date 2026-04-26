#!/usr/bin/env python
"""Generate baseline toy logical execution records."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from failureops.data_model import BASELINE_FIELDS
from failureops.io_utils import write_csv_rows
from failureops.toy_simulator import generate_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-shots", type=int, default=1000)
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--num-operations", type=int, default=5)
    parser.add_argument("--data-error-rate", type=float, default=0.03)
    parser.add_argument("--measurement-error-rate", type=float, default=0.02)
    parser.add_argument("--idle-error-rate", type=float, default=0.01)
    parser.add_argument("--decoder-timeout-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default="p0")
    parser.add_argument("--circuit-id", default="toy_logical_workload")
    parser.add_argument("--output", default="data/results/baseline_runs.csv")
    args = parser.parse_args()

    rows = generate_runs(
        num_shots=args.num_shots,
        num_rounds=args.num_rounds,
        num_operations=args.num_operations,
        data_error_rate=args.data_error_rate,
        measurement_error_rate=args.measurement_error_rate,
        idle_error_rate=args.idle_error_rate,
        decoder_timeout_rate=args.decoder_timeout_rate,
        seed=args.seed,
        run_id=args.run_id,
        circuit_id=args.circuit_id,
    )
    write_csv_rows(args.output, rows, BASELINE_FIELDS)
    print(f"wrote {len(rows)} baseline rows to {args.output}")


if __name__ == "__main__":
    main()

