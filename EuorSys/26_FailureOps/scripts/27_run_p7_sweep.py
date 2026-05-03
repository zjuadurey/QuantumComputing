#!/usr/bin/env python
"""Run a P7 decoder-pathway sweep across the Google RL QEC dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from failureops.data_model import P7_SWEEP_AGGREGATE_FIELDS, P7_SWEEP_SUMMARY_FIELDS
from failureops.google_rl_qec_adapter import (
    discover_google_rl_qec_data_dirs,
    summarize_google_rl_qec_condition,
    summarize_p7_sweep_groups,
)
from failureops.io_utils import write_csv_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default="data/raw/google_rl_qec/google_reinforcement_learning_qec",
    )
    parser.add_argument(
        "--baseline-decoder",
        default="correlated_matching_decoder_with_si1000_prior",
    )
    parser.add_argument(
        "--intervened-decoder",
        default="tesseract_decoder_with_si1000_prior",
    )
    parser.add_argument("--max-shots", type=int, default=10000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-bootstrap", type=int, default=100)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--run-id-prefix", default="p7_google_rl_qec_sweep")
    parser.add_argument(
        "--summary-output",
        default="data/results/p7_google_rl_qec_decoder_effect_matrix.csv",
    )
    parser.add_argument(
        "--aggregate-output",
        default="data/results/p7_google_rl_qec_decoder_effect_aggregate.csv",
    )
    args = parser.parse_args()

    data_dirs = discover_google_rl_qec_data_dirs(
        args.dataset_root,
        baseline_decoder_pathway=args.baseline_decoder,
        intervened_decoder_pathway=args.intervened_decoder,
    )
    if args.limit:
        data_dirs = data_dirs[: args.limit]
    if not data_dirs:
        raise ValueError(f"no Google RL QEC experiment directories found under {args.dataset_root}")

    summary_rows = []
    for index, data_dir in enumerate(data_dirs):
        run_id = f"{args.run_id_prefix}_{index:03d}"
        row = summarize_google_rl_qec_condition(
            data_dir,
            baseline_decoder_pathway=args.baseline_decoder,
            intervened_decoder_pathway=args.intervened_decoder,
            max_shots=args.max_shots,
            run_id=run_id,
            num_bootstrap=args.num_bootstrap,
            bootstrap_seed=args.bootstrap_seed,
        )
        summary_rows.append(row)
        print(
            f"[{index + 1}/{len(data_dirs)}] {row['workload_id']} "
            f"delta={row['paired_delta_lfr']} valid={row['valid_pairs']}"
        )

    aggregate_rows = summarize_p7_sweep_groups(summary_rows)
    write_csv_rows(args.summary_output, summary_rows, P7_SWEEP_SUMMARY_FIELDS)
    write_csv_rows(args.aggregate_output, aggregate_rows, P7_SWEEP_AGGREGATE_FIELDS)
    print(f"wrote {len(summary_rows)} P7 sweep rows to {args.summary_output}")
    print(f"wrote {len(aggregate_rows)} P7 aggregate rows to {args.aggregate_output}")


if __name__ == "__main__":
    main()
