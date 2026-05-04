#!/usr/bin/env python
"""Run the full P3 evaluation matrix and write all summary artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from failureops.baselines import compare_p3_baseline_methods, summarize_p3_eval_matrix
from failureops.data_model import (
    P3_BASELINE_COMPARISON_FIELDS,
    P3_BASELINE_FIELDS,
    P3_EVAL_MATRIX_FIELDS,
    P3_INTERVENTION_FIELDS,
    P3_RANK_STABILITY_FIELDS,
)
from failureops.io_utils import write_csv_rows
from failureops.rank_stability import summarize_rank_stability
from failureops.runtime_service import (
    P3_INTERVENTIONS,
    P3_STRESS_CONFIGS,
    apply_p3_intervention,
    compare_p3_records,
    generate_p3_runs,
)
from failureops.workloads import parse_workload_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workloads", default="memory_x,idle_heavy_memory,high_detector_load_memory")
    parser.add_argument("--stress-levels", default="low,medium,high")
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--num-shots-per-seed", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--seed-stride", type=int, default=10000)
    parser.add_argument("--baseline-output", default="data/results/p3_baseline_runs.csv")
    parser.add_argument("--intervention-output", default="data/results/p3_intervened_runs.csv")
    parser.add_argument("--eval-output", default="data/results/p3_eval_matrix_summary.csv")
    parser.add_argument("--comparison-output", default="data/results/p3_baseline_comparison.csv")
    parser.add_argument("--rank-output", default="data/results/p3_rank_stability.csv")
    args = parser.parse_args()

    workloads = parse_workload_ids(args.workloads)
    stress_levels = parse_stress_levels(args.stress_levels)
    baseline_rows = []
    intervention_rows = []
    for workload_id in workloads:
        for stress_level in stress_levels:
            for seed_index in range(args.num_seeds):
                seed = args.seed_start + seed_index * args.seed_stride
                run_rows = generate_p3_runs(
                    workload_id=workload_id,
                    stress_level=stress_level,
                    num_shots=args.num_shots_per_seed,
                    seed=seed,
                    run_id=f"p3_{workload_id}_{stress_level}_{seed_index}",
                )
                baseline_rows.extend(run_rows)
                for baseline in run_rows:
                    for intervention in P3_INTERVENTIONS:
                        intervened = apply_p3_intervention(baseline, intervention)
                        intervention_rows.append(compare_p3_records(baseline, intervened, intervention))

    eval_rows = summarize_p3_eval_matrix(intervention_rows)
    comparison_rows = compare_p3_baseline_methods(intervention_rows)
    rank_rows = summarize_rank_stability(intervention_rows)
    write_csv_rows(args.baseline_output, baseline_rows, P3_BASELINE_FIELDS)
    write_csv_rows(args.intervention_output, intervention_rows, P3_INTERVENTION_FIELDS)
    write_csv_rows(args.eval_output, eval_rows, P3_EVAL_MATRIX_FIELDS)
    write_csv_rows(args.comparison_output, comparison_rows, P3_BASELINE_COMPARISON_FIELDS)
    write_csv_rows(args.rank_output, rank_rows, P3_RANK_STABILITY_FIELDS)
    print(f"wrote {len(baseline_rows)} P3 baseline rows to {args.baseline_output}")
    print(f"wrote {len(intervention_rows)} P3 intervention rows to {args.intervention_output}")
    print(f"wrote {len(eval_rows)} P3 eval-matrix rows to {args.eval_output}")
    print(f"wrote {len(comparison_rows)} P3 baseline-comparison rows to {args.comparison_output}")
    print(f"wrote {len(rank_rows)} P3 rank-stability rows to {args.rank_output}")


def parse_stress_levels(value: str) -> list[str]:
    levels = [item.strip() for item in value.split(",") if item.strip()]
    for level in levels:
        if level not in P3_STRESS_CONFIGS:
            choices = ", ".join(sorted(P3_STRESS_CONFIGS))
            raise ValueError(f"unknown stress level {level!r}; choose from: {choices}")
    return levels


if __name__ == "__main__":
    main()

