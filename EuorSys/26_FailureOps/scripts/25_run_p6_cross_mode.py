#!/usr/bin/env python
"""Run P6 cross-mode evaluation through one unified experiment interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from failureops.data_model import (
    P6_INTERVENTION_STABILITY_FIELDS,
    P6_MODE_SUMMARY_FIELDS,
    P6_RANK_STABILITY_FIELDS,
)
from failureops.io_utils import write_csv_rows
from failureops.p6_unified import load_p6_config, run_p6_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/p6_cross_mode.yaml")
    args = parser.parse_args()

    config = load_p6_config(args.config)
    summaries = run_p6_config(config, command=sys.argv)
    write_csv_rows(config["outputs"]["mode_summary"], summaries["mode_summary"], P6_MODE_SUMMARY_FIELDS)
    write_csv_rows(
        config["outputs"]["intervention_stability"],
        summaries["intervention_stability"],
        P6_INTERVENTION_STABILITY_FIELDS,
    )
    write_csv_rows(config["outputs"]["rank_stability"], summaries["rank_stability"], P6_RANK_STABILITY_FIELDS)
    print(f"wrote {len(summaries['mode_summary'])} P6 mode-summary rows to {config['outputs']['mode_summary']}")
    print(
        f"wrote {len(summaries['intervention_stability'])} P6 intervention-stability rows "
        f"to {config['outputs']['intervention_stability']}"
    )
    print(f"wrote P6 rank-stability row to {config['outputs']['rank_stability']}")


if __name__ == "__main__":
    main()

