#!/usr/bin/env python
"""Export a P5 runtime trace CSV/JSON from baseline rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from failureops.data_model import P5_RUNTIME_TRACE_FIELDS
from failureops.io_utils import read_csv_rows
from failureops.runtime_trace import export_runtime_trace_rows, write_trace_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/results/p4_baseline_runs.csv")
    parser.add_argument("--output", default="data/results/p5a_runtime_trace.csv")
    parser.add_argument("--trace-source", default="failureops_proxy_export")
    args = parser.parse_args()

    rows = export_runtime_trace_rows(
        read_csv_rows(args.input),
        trace_source=args.trace_source,
    )
    write_trace_rows(args.output, rows, P5_RUNTIME_TRACE_FIELDS)
    print(f"wrote {len(rows)} runtime trace rows to {args.output}")


if __name__ == "__main__":
    main()

