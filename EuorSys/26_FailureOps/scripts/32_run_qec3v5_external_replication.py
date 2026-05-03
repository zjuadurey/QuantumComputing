#!/usr/bin/env python
"""Run qec3v5 external real-record decoder-pathway replication."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from failureops.data_model import (
    P10_EXTERNAL_REPLICATION_AGGREGATE_FIELDS,
    P10_EXTERNAL_REPLICATION_FIELDS,
)
from failureops.io_utils import write_csv_rows
from failureops.qec3v5_adapter import (
    discover_qec3v5_data_dirs,
    summarize_qec3v5_condition,
    summarize_qec3v5_groups,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/google_qec3v5")
    parser.add_argument("--code", default="surface_code")
    parser.add_argument("--distance", type=int, default=5)
    parser.add_argument("--center", default="5_5")
    parser.add_argument("--baseline-decoder", default="pymatching")
    parser.add_argument("--intervened-decoder", default="correlated_matching")
    parser.add_argument("--max-shots", type=int, default=10000)
    parser.add_argument("--output", default="data/results/p10_qec3v5_external_decoder_effect_matrix.csv")
    parser.add_argument("--aggregate-output", default="data/results/p10_qec3v5_external_decoder_effect_aggregate.csv")
    args = parser.parse_args()

    dirs = discover_qec3v5_data_dirs(
        args.root,
        code=args.code,
        distance=args.distance,
        center=args.center,
        baseline_decoder=args.baseline_decoder,
        intervened_decoder=args.intervened_decoder,
    )
    if not dirs:
        raise ValueError(f"no qec3v5 directories found under {args.root}")

    rows = [
        summarize_qec3v5_condition(
            data_dir,
            baseline_decoder=args.baseline_decoder,
            intervened_decoder=args.intervened_decoder,
            max_shots=args.max_shots,
        )
        for data_dir in dirs
    ]
    aggregate_rows = summarize_qec3v5_groups(rows)
    write_csv_rows(args.output, rows, P10_EXTERNAL_REPLICATION_FIELDS)
    write_csv_rows(args.aggregate_output, aggregate_rows, P10_EXTERNAL_REPLICATION_AGGREGATE_FIELDS)
    print(f"wrote {len(rows)} qec3v5 external replication rows to {args.output}")
    print(f"wrote {len(aggregate_rows)} qec3v5 external aggregate rows to {args.aggregate_output}")


if __name__ == "__main__":
    main()
