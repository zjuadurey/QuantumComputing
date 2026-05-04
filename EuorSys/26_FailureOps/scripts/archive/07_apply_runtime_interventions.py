#!/usr/bin/env python
"""Apply P2 noise and runtime/system interventions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from failureops.data_model import INTERVENTION_FIELDS
from failureops.io_utils import read_csv_rows, write_csv_rows
from failureops.qec_interventions import generate_noise_intervention_rows
from failureops.runtime_interventions import (
    P2_INTERVENTIONS,
    classify_intervention,
    generate_p2_runtime_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/results/p2_baseline_runs.csv")
    parser.add_argument("--output", default="data/results/p2_intervened_runs.csv")
    args = parser.parse_args()

    baseline_rows = read_csv_rows(args.input)
    rows = []
    for intervention in P2_INTERVENTIONS:
        kind = classify_intervention(intervention)
        if kind == "noise":
            rows.extend(
                generate_noise_intervention_rows(
                    baseline_rows=baseline_rows,
                    intervention=intervention,
                )
            )
        else:
            rows.extend(
                generate_p2_runtime_rows(
                    baseline_rows=baseline_rows,
                    intervention=intervention,
                )
            )

    write_csv_rows(args.output, rows, INTERVENTION_FIELDS)
    print(
        f"wrote {len(rows)} P2 intervention rows "
        f"({len(baseline_rows)} shots x {len(P2_INTERVENTIONS)} interventions) to {args.output}"
    )


if __name__ == "__main__":
    main()
