#!/usr/bin/env python
"""Analyze P4 paired treatment effects with bootstrap confidence intervals."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from failureops.data_model import P4_PAIRED_EFFECT_FIELDS
from failureops.io_utils import read_csv_rows, write_csv_rows
from failureops.paired_metrics import summarize_paired_effects


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/results/p4_intervened_runs.csv")
    parser.add_argument("--output", default="data/results/p4_paired_effects.csv")
    parser.add_argument("--num-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()

    rows = summarize_paired_effects(
        read_csv_rows(args.input),
        num_bootstrap=args.num_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_csv_rows(args.output, rows, P4_PAIRED_EFFECT_FIELDS)
    print(f"wrote {len(rows)} P4 paired-effect rows to {args.output}")


if __name__ == "__main__":
    main()

