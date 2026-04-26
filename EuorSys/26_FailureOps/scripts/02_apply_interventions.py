#!/usr/bin/env python
"""Apply paired counterfactual interventions to baseline records."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from failureops.data_model import INTERVENTION_FIELDS
from failureops.interventions import INTERVENTIONS, compare_intervention
from failureops.io_utils import read_csv_rows, write_csv_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/results/baseline_runs.csv")
    parser.add_argument("--output", default="data/results/intervened_runs.csv")
    args = parser.parse_args()

    baseline_rows = read_csv_rows(args.input)
    rows = [
        compare_intervention(baseline, intervention)
        for baseline in baseline_rows
        for intervention in INTERVENTIONS
    ]
    write_csv_rows(args.output, rows, INTERVENTION_FIELDS)
    print(
        f"wrote {len(rows)} intervention rows "
        f"({len(baseline_rows)} shots x {len(INTERVENTIONS)} interventions) to {args.output}"
    )


if __name__ == "__main__":
    main()

