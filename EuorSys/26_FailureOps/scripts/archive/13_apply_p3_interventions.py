#!/usr/bin/env python
"""Apply P3 paired counterfactual interventions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from failureops.data_model import P3_INTERVENTION_FIELDS
from failureops.io_utils import read_csv_rows, write_csv_rows
from failureops.runtime_service import P3_INTERVENTIONS, apply_p3_intervention, compare_p3_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/results/p3_baseline_runs.csv")
    parser.add_argument("--output", default="data/results/p3_intervened_runs.csv")
    args = parser.parse_args()

    baseline_rows = read_csv_rows(args.input)
    rows = []
    for baseline in baseline_rows:
        for intervention in P3_INTERVENTIONS:
            intervened = apply_p3_intervention(baseline, intervention)
            rows.append(compare_p3_records(baseline, intervened, intervention))

    write_csv_rows(args.output, rows, P3_INTERVENTION_FIELDS)
    print(
        f"wrote {len(rows)} P3 intervention rows "
        f"({len(baseline_rows)} shots x {len(P3_INTERVENTIONS)} interventions) to {args.output}"
    )


if __name__ == "__main__":
    main()

