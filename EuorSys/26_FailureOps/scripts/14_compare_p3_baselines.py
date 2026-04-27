#!/usr/bin/env python
"""Compare P3 paired FailureOps attribution against simple baselines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from failureops.baselines import compare_p3_baseline_methods, summarize_p3_eval_matrix
from failureops.data_model import P3_BASELINE_COMPARISON_FIELDS, P3_EVAL_MATRIX_FIELDS
from failureops.io_utils import read_csv_rows, write_csv_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/results/p3_intervened_runs.csv")
    parser.add_argument("--eval-output", default="data/results/p3_eval_matrix_summary.csv")
    parser.add_argument("--comparison-output", default="data/results/p3_baseline_comparison.csv")
    args = parser.parse_args()

    rows = read_csv_rows(args.input)
    eval_rows = summarize_p3_eval_matrix(rows)
    comparison_rows = compare_p3_baseline_methods(rows)
    write_csv_rows(args.eval_output, eval_rows, P3_EVAL_MATRIX_FIELDS)
    write_csv_rows(args.comparison_output, comparison_rows, P3_BASELINE_COMPARISON_FIELDS)
    print(f"wrote {len(eval_rows)} P3 eval-matrix rows to {args.eval_output}")
    print(f"wrote {len(comparison_rows)} P3 baseline-comparison rows to {args.comparison_output}")


if __name__ == "__main__":
    main()

