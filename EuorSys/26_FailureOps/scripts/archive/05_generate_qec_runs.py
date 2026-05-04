#!/usr/bin/env python
"""Generate P1 Stim/PyMatching-backed baseline records."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from failureops.data_model import QEC_BASELINE_FIELDS
from failureops.io_utils import write_csv_rows
from failureops.qec_backend import generate_qec_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-shots", type=int, default=1000)
    parser.add_argument("--distance", type=int, default=5)
    parser.add_argument("--num-rounds", type=int, default=8)
    parser.add_argument("--data-error-rate", type=float, default=0.03)
    parser.add_argument("--measurement-error-rate", type=float, default=0.02)
    parser.add_argument("--idle-error-rate", type=float, default=0.002)
    parser.add_argument("--decoder-timeout-base-rate", type=float, default=0.01)
    parser.add_argument("--decoder-capacity", type=float, default=4.0)
    parser.add_argument("--synchronization-slack", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default="p1")
    parser.add_argument("--circuit-id", default="stim_repetition_memory")
    parser.add_argument("--output", default="data/results/p1_baseline_runs.csv")
    args = parser.parse_args()

    rows = generate_qec_runs(
        num_shots=args.num_shots,
        distance=args.distance,
        num_rounds=args.num_rounds,
        data_error_rate=args.data_error_rate,
        measurement_error_rate=args.measurement_error_rate,
        idle_error_rate=args.idle_error_rate,
        decoder_timeout_base_rate=args.decoder_timeout_base_rate,
        seed=args.seed,
        decoder_capacity=args.decoder_capacity,
        synchronization_slack=args.synchronization_slack,
        run_id=args.run_id,
        circuit_id=args.circuit_id,
    )
    write_csv_rows(args.output, rows, QEC_BASELINE_FIELDS)
    print(f"wrote {len(rows)} QEC baseline rows to {args.output}")


if __name__ == "__main__":
    main()
