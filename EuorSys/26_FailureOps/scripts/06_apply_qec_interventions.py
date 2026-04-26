#!/usr/bin/env python
"""Apply P1 QEC noise and runtime interventions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from failureops.data_model import INTERVENTION_FIELDS
from failureops.io_utils import read_csv_rows, write_csv_rows
from failureops.qec_interventions import (
    NOISE_INTERVENTIONS,
    QEC_INTERVENTIONS,
    generate_noise_intervention_rows,
    generate_runtime_intervention_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/results/p1_baseline_runs.csv")
    parser.add_argument("--output", default="data/results/p1_intervened_runs.csv")
    args = parser.parse_args()

    baseline_rows = read_csv_rows(args.input)
    rows = []
    for intervention in QEC_INTERVENTIONS:
        if intervention in NOISE_INTERVENTIONS:
            rows.extend(
                generate_noise_intervention_rows(
                    baseline_rows=baseline_rows,
                    intervention=intervention,
                )
            )
        else:
            rows.extend(
                generate_runtime_intervention_rows(
                    baseline_rows=baseline_rows,
                    intervention=intervention,
                )
            )

    write_csv_rows(args.output, rows, INTERVENTION_FIELDS)
    print(
        f"wrote {len(rows)} P1 intervention rows "
        f"({len(baseline_rows)} shots x {len(QEC_INTERVENTIONS)} interventions) to {args.output}"
    )


if __name__ == "__main__":
    main()
