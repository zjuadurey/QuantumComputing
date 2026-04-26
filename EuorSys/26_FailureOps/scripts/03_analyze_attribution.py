#!/usr/bin/env python
"""Compute intervention sensitivity metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from failureops.data_model import ATTRIBUTION_FIELDS
from failureops.io_utils import read_csv_rows, write_csv_rows
from failureops.metrics import summarize_attribution


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/results/intervened_runs.csv")
    parser.add_argument("--output", default="data/results/attribution_summary.csv")
    args = parser.parse_args()

    rows = summarize_attribution(read_csv_rows(args.input))
    write_csv_rows(args.output, rows, ATTRIBUTION_FIELDS)
    print(f"wrote {len(rows)} attribution rows to {args.output}")


if __name__ == "__main__":
    main()

