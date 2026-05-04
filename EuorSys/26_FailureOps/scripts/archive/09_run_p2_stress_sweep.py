#!/usr/bin/env python
"""Run P2 under low/medium/high runtime stress settings."""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from failureops.io_utils import fmt_float, write_csv_rows

import importlib.util

_REPEATS_PATH = Path(__file__).resolve().with_name("08_run_p2_repeats.py")
_SPEC = importlib.util.spec_from_file_location("p2_repeats", _REPEATS_PATH)
assert _SPEC and _SPEC.loader
p2_repeats = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(p2_repeats)

STRESS_FIELDS = [
    "stress_level",
    "intervention",
    "num_seeds",
    "num_shots_per_seed",
    "decoder_timeout_base_rate",
    "decoder_capacity",
    "synchronization_slack",
    "mean_baseline_lfr",
    "mean_intervened_lfr",
    "mean_absolute_delta_lfr",
    "std_absolute_delta_lfr",
    "ci95_absolute_delta_lfr",
    "mean_rescue_rate",
]

STRESS_ORDER = ["low", "medium", "high"]

STRESS_CONFIGS = {
    "low": {
        "decoder_timeout_base_rate": 0.005,
        "decoder_capacity": 6.0,
        "synchronization_slack": 0.60,
    },
    "medium": {
        "decoder_timeout_base_rate": 0.02,
        "decoder_capacity": 3.0,
        "synchronization_slack": 0.25,
    },
    "high": {
        "decoder_timeout_base_rate": 0.04,
        "decoder_capacity": 2.0,
        "synchronization_slack": 0.10,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--num-shots-per-seed", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--seed-stride", type=int, default=10000)
    parser.add_argument("--distance", type=int, default=5)
    parser.add_argument("--num-rounds", type=int, default=8)
    parser.add_argument("--data-error-rate", type=float, default=0.03)
    parser.add_argument("--measurement-error-rate", type=float, default=0.02)
    parser.add_argument("--idle-error-rate", type=float, default=0.002)
    parser.add_argument("--output", default="data/results/p2_5_stress_sweep_summary.csv")
    args = parser.parse_args()

    rows = []
    for stress_level, stress_config in STRESS_CONFIGS.items():
        rows.extend(run_stress_level(args, stress_level, stress_config))
    stress_rank = {level: index for index, level in enumerate(STRESS_ORDER)}
    rows.sort(key=lambda row: (stress_rank[row["stress_level"]], float(row["mean_absolute_delta_lfr"]), row["intervention"]))
    write_csv_rows(args.output, rows, STRESS_FIELDS)
    print(f"wrote {len(rows)} stress-sweep rows to {args.output}")


def run_stress_level(args: argparse.Namespace, stress_level: str, stress_config: dict[str, float]) -> list[dict[str, object]]:
    per_intervention: dict[str, list[dict[str, float]]] = {}
    stress_args = argparse.Namespace(**vars(args))
    stress_args.decoder_timeout_base_rate = stress_config["decoder_timeout_base_rate"]
    stress_args.decoder_capacity = stress_config["decoder_capacity"]
    stress_args.synchronization_slack = stress_config["synchronization_slack"]

    for seed_index in range(args.num_seeds):
        seed = args.seed_start + seed_index * args.seed_stride
        summary_rows = p2_repeats.run_one_p2_summary(
            args=stress_args,
            seed=seed,
            run_id=f"p2_5_{stress_level}_{seed_index}",
        )
        for row in summary_rows:
            per_intervention.setdefault(row["intervention"], []).append(
                {
                    "baseline_lfr": float(row["baseline_logical_failure_rate"]),
                    "intervened_lfr": float(row["intervened_logical_failure_rate"]),
                    "delta_lfr": float(row["absolute_delta_lfr"]),
                    "rescue_rate": float(row["rescue_rate"]),
                }
            )

    out = []
    for intervention, values in per_intervention.items():
        deltas = [row["delta_lfr"] for row in values]
        std_delta = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
        out.append(
            {
                "stress_level": stress_level,
                "intervention": intervention,
                "num_seeds": len(values),
                "num_shots_per_seed": args.num_shots_per_seed,
                "decoder_timeout_base_rate": fmt_float(stress_config["decoder_timeout_base_rate"]),
                "decoder_capacity": fmt_float(stress_config["decoder_capacity"]),
                "synchronization_slack": fmt_float(stress_config["synchronization_slack"]),
                "mean_baseline_lfr": fmt_float(statistics.fmean(row["baseline_lfr"] for row in values)),
                "mean_intervened_lfr": fmt_float(statistics.fmean(row["intervened_lfr"] for row in values)),
                "mean_absolute_delta_lfr": fmt_float(statistics.fmean(deltas)),
                "std_absolute_delta_lfr": fmt_float(std_delta),
                "ci95_absolute_delta_lfr": fmt_float(1.96 * std_delta / math.sqrt(len(deltas))),
                "mean_rescue_rate": fmt_float(statistics.fmean(row["rescue_rate"] for row in values)),
            }
        )
    return out


if __name__ == "__main__":
    main()
