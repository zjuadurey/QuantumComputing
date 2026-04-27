#!/usr/bin/env python
"""Generate P3 baseline runs across workloads, stress levels, and seeds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from failureops.data_model import P3_BASELINE_FIELDS
from failureops.io_utils import write_csv_rows
from failureops.runtime_service import P3_STRESS_CONFIGS, generate_p3_runs
from failureops.workloads import WORKLOADS, parse_workload_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workloads", default="memory_x,idle_heavy_memory,high_detector_load_memory")
    parser.add_argument("--stress-levels", default="low,medium,high")
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--num-shots-per-seed", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--seed-stride", type=int, default=10000)
    parser.add_argument("--output", default="data/results/p3_baseline_runs.csv")
    args = parser.parse_args()

    workloads = parse_workload_ids(args.workloads)
    stress_levels = parse_stress_levels(args.stress_levels)
    rows = []
    for workload_id in workloads:
        for stress_level in stress_levels:
            for seed_index in range(args.num_seeds):
                seed = args.seed_start + seed_index * args.seed_stride
                rows.extend(
                    generate_p3_runs(
                        workload_id=workload_id,
                        stress_level=stress_level,
                        num_shots=args.num_shots_per_seed,
                        seed=seed,
                        run_id=f"p3_{workload_id}_{stress_level}_{seed_index}",
                    )
                )

    write_csv_rows(args.output, rows, P3_BASELINE_FIELDS)
    print(
        f"wrote {len(rows)} P3 baseline rows "
        f"({len(workloads)} workloads x {len(stress_levels)} stress levels x "
        f"{args.num_seeds} seeds x {args.num_shots_per_seed} shots) to {args.output}"
    )


def parse_stress_levels(value: str) -> list[str]:
    levels = [item.strip() for item in value.split(",") if item.strip()]
    for level in levels:
        if level not in P3_STRESS_CONFIGS:
            choices = ", ".join(sorted(P3_STRESS_CONFIGS))
            raise ValueError(f"unknown stress level {level!r}; choose from: {choices}")
    return levels


if __name__ == "__main__":
    main()

