#!/usr/bin/env python
"""Run the P2 pipeline across multiple seeds and aggregate attribution metrics."""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from failureops.io_utils import fmt_float, write_csv_rows
from failureops.metrics import summarize_attribution
from failureops.qec_backend import generate_qec_runs
from failureops.qec_interventions import generate_noise_intervention_rows
from failureops.runtime_interventions import (
    P2_INTERVENTIONS,
    classify_intervention,
    generate_p2_runtime_rows,
)

REPEATED_FIELDS = [
    "intervention",
    "num_seeds",
    "mean_baseline_lfr",
    "mean_intervened_lfr",
    "mean_absolute_delta_lfr",
    "std_absolute_delta_lfr",
    "ci95_absolute_delta_lfr",
    "mean_rescue_rate",
    "std_rescue_rate",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--num-shots-per-seed", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--seed-stride", type=int, default=10000)
    parser.add_argument("--distance", type=int, default=5)
    parser.add_argument("--num-rounds", type=int, default=8)
    parser.add_argument("--data-error-rate", type=float, default=0.03)
    parser.add_argument("--measurement-error-rate", type=float, default=0.02)
    parser.add_argument("--idle-error-rate", type=float, default=0.002)
    parser.add_argument("--decoder-timeout-base-rate", type=float, default=0.02)
    parser.add_argument("--decoder-capacity", type=float, default=3.0)
    parser.add_argument("--synchronization-slack", type=float, default=0.25)
    parser.add_argument("--output", default="data/results/p2_5_repeated_summary.csv")
    args = parser.parse_args()

    per_intervention: dict[str, list[dict[str, float]]] = {name: [] for name in P2_INTERVENTIONS}
    for seed_index in range(args.num_seeds):
        seed = args.seed_start + seed_index * args.seed_stride
        summary_rows = run_one_p2_summary(args=args, seed=seed, run_id=f"p2_5_repeat_{seed_index}")
        for row in summary_rows:
            per_intervention[row["intervention"]].append(
                {
                    "baseline_lfr": float(row["baseline_logical_failure_rate"]),
                    "intervened_lfr": float(row["intervened_logical_failure_rate"]),
                    "delta_lfr": float(row["absolute_delta_lfr"]),
                    "rescue_rate": float(row["rescue_rate"]),
                }
            )

    rows = [
        aggregate_intervention(intervention, values)
        for intervention, values in per_intervention.items()
        if values
    ]
    rows.sort(key=lambda row: (float(row["mean_absolute_delta_lfr"]), row["intervention"]))
    write_csv_rows(args.output, rows, REPEATED_FIELDS)
    print(f"wrote {len(rows)} repeated-summary rows to {args.output}")


def run_one_p2_summary(argparse_namespace=None, *, args=None, seed: int, run_id: str) -> list[dict[str, object]]:
    if args is None:
        args = argparse_namespace
    baseline_rows = generate_qec_runs(
        num_shots=args.num_shots_per_seed,
        distance=args.distance,
        num_rounds=args.num_rounds,
        data_error_rate=args.data_error_rate,
        measurement_error_rate=args.measurement_error_rate,
        idle_error_rate=args.idle_error_rate,
        decoder_timeout_base_rate=args.decoder_timeout_base_rate,
        seed=seed,
        decoder_capacity=args.decoder_capacity,
        synchronization_slack=args.synchronization_slack,
        run_id=run_id,
    )
    intervention_rows = []
    for intervention in P2_INTERVENTIONS:
        if classify_intervention(intervention) == "noise":
            intervention_rows.extend(
                generate_noise_intervention_rows(
                    baseline_rows=baseline_rows,
                    intervention=intervention,
                )
            )
        else:
            intervention_rows.extend(
                generate_p2_runtime_rows(
                    baseline_rows=baseline_rows,
                    intervention=intervention,
                )
            )
    return summarize_attribution(intervention_rows)


def aggregate_intervention(intervention: str, values: list[dict[str, float]]) -> dict[str, object]:
    deltas = [row["delta_lfr"] for row in values]
    rescue_rates = [row["rescue_rate"] for row in values]
    std_delta = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    std_rescue = statistics.stdev(rescue_rates) if len(rescue_rates) > 1 else 0.0
    return {
        "intervention": intervention,
        "num_seeds": len(values),
        "mean_baseline_lfr": fmt_float(statistics.fmean(row["baseline_lfr"] for row in values)),
        "mean_intervened_lfr": fmt_float(statistics.fmean(row["intervened_lfr"] for row in values)),
        "mean_absolute_delta_lfr": fmt_float(statistics.fmean(deltas)),
        "std_absolute_delta_lfr": fmt_float(std_delta),
        "ci95_absolute_delta_lfr": fmt_float(1.96 * std_delta / math.sqrt(len(deltas))),
        "mean_rescue_rate": fmt_float(statistics.fmean(rescue_rates)),
        "std_rescue_rate": fmt_float(std_rescue),
    }


if __name__ == "__main__":
    main()

