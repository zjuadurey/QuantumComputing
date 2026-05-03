#!/usr/bin/env python
"""Summarize non-shot-level external QEC datasets for FailureOps claim boundaries."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from failureops.data_model import P10_EXTERNAL_SANITY_FIELDS
from failureops.io_utils import fmt_float, read_csv_rows, write_csv_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daqec-daily", default="data/raw/daqec_ibm/daily_summary.csv")
    parser.add_argument("--daqec-effects", default="data/raw/daqec_ibm/effect_sizes_by_condition.csv")
    parser.add_argument("--eth-zip", default="data/raw/eth_surface_code/Surface_Code_Data.zip")
    parser.add_argument("--output", default="data/results/p10_external_sanity_checks.csv")
    args = parser.parse_args()

    rows = []
    rows.extend(summarize_daqec_daily(args.daqec_daily))
    rows.extend(summarize_daqec_effects(args.daqec_effects))
    rows.append(summarize_eth_zip(args.eth_zip))
    write_csv_rows(args.output, rows, P10_EXTERNAL_SANITY_FIELDS)
    print(f"wrote {len(rows)} external sanity rows to {args.output}")


def summarize_daqec_daily(path: str) -> list[dict[str, object]]:
    rows = read_csv_rows(path)
    grouped = {}
    for row in rows:
        grouped.setdefault((row["day"], row["backend"]), {})[row["strategy"]] = row

    baseline_values = []
    intervened_values = []
    for strategies in grouped.values():
        baseline = strategies.get("baseline_static")
        intervened = strategies.get("drift_aware_full_stack")
        if baseline is None or intervened is None:
            continue
        baseline_values.append(float(baseline["logical_error_rate_mean"]))
        intervened_values.append(float(intervened["logical_error_rate_mean"]))

    baseline_mean = statistics.fmean(baseline_values) if baseline_values else 0.0
    intervened_mean = statistics.fmean(intervened_values) if intervened_values else 0.0
    delta = intervened_mean - baseline_mean
    return [
        {
            "source_dataset": "daqec_ibm_2025",
            "evidence_type": "independent_aggregate_ibm_hardware",
            "analysis_scope": "paired day-backend daily_summary",
            "num_units": len(baseline_values),
            "baseline_metric": fmt_float(baseline_mean),
            "intervened_metric": fmt_float(intervened_mean),
            "absolute_delta": fmt_float(delta),
            "relative_delta": fmt_float(delta / baseline_mean if baseline_mean else 0.0),
            "interpretation": "Independent IBM aggregate data shows lower logical error under the drift-aware intervention.",
            "failureops_compatibility": "aggregate paired-session sanity check; not shot-level detector-record attribution",
            "source_artifact": path,
        }
    ]


def summarize_daqec_effects(path: str) -> list[dict[str, object]]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return []
    baseline_values = [float(row["baseline_mean"]) for row in rows]
    intervened_values = [float(row["drift_aware_mean"]) for row in rows]
    reductions = [float(row["relative_reduction"]) for row in rows]
    baseline_mean = statistics.fmean(baseline_values)
    intervened_mean = statistics.fmean(intervened_values)
    delta = intervened_mean - baseline_mean
    return [
        {
            "source_dataset": "daqec_ibm_2025",
            "evidence_type": "independent_aggregate_ibm_hardware",
            "analysis_scope": "distance-backend effect_sizes_by_condition",
            "num_units": len(rows),
            "baseline_metric": fmt_float(baseline_mean),
            "intervened_metric": fmt_float(intervened_mean),
            "absolute_delta": fmt_float(delta),
            "relative_delta": fmt_float(statistics.fmean(reductions)),
            "interpretation": "Effect-size table reports aggregate logical-error reduction over IBM backend-distance conditions.",
            "failureops_compatibility": "aggregate effect-size sanity check; not shot-level paired detector records",
            "source_artifact": path,
        }
    ]


def summarize_eth_zip(path: str) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
    return {
        "source_dataset": "eth_surface_code_2020",
        "evidence_type": "independent_public_surface_code_figure_data",
        "analysis_scope": "schema inspection",
        "num_units": len(csv_names),
        "baseline_metric": "",
        "intervened_metric": "",
        "absolute_delta": "",
        "relative_delta": "",
        "interpretation": "ETH dataset is public independent surface-code data, but contains figure-level CSVs instead of shot-level detector records.",
        "failureops_compatibility": "not attribution-compatible without raw shot/syndrome records",
        "source_artifact": path,
    }


if __name__ == "__main__":
    main()
