#!/usr/bin/env python
"""Analyze P3 intervention-ranking stability across seeds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from failureops.data_model import P3_RANK_STABILITY_FIELDS
from failureops.io_utils import read_csv_rows, write_csv_rows
from failureops.rank_stability import summarize_rank_stability


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/results/p3_intervened_runs.csv")
    parser.add_argument("--output", default="data/results/p3_rank_stability.csv")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    rows = summarize_rank_stability(read_csv_rows(args.input), top_k=args.top_k)
    write_csv_rows(args.output, rows, P3_RANK_STABILITY_FIELDS)
    print(f"wrote {len(rows)} P3 rank-stability rows to {args.output}")


if __name__ == "__main__":
    main()

