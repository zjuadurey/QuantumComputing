#!/usr/bin/env python
"""Validate P4 paired-counterfactual rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from failureops.data_model import P4_PAIRING_VALIDATION_FIELDS
from failureops.io_utils import read_csv_rows, write_csv_rows
from failureops.pairing import validate_intervention_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/results/p4_intervened_runs.csv")
    parser.add_argument("--output", default="data/results/p4_pairing_validation.csv")
    args = parser.parse_args()

    rows = validate_intervention_rows(read_csv_rows(args.input))
    write_csv_rows(args.output, rows, P4_PAIRING_VALIDATION_FIELDS)
    invalid = sum(int(row["invalid_pairs"]) for row in rows)
    print(f"wrote {len(rows)} pairing-validation rows to {args.output}; invalid pairs={invalid}")


if __name__ == "__main__":
    main()

