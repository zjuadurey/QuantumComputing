#!/usr/bin/env python
"""Analyze intervention-induced failure-pattern shifts and simple sanity checks."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from failureops.io_utils import parse_bool, read_csv_rows, write_csv_rows

PATTERN_FIELDS = [
    "intervention",
    "num_rows",
    "baseline_failure_count",
    "intervened_failure_count",
    "rescued_failure_count",
    "dominant_baseline_failure_pattern",
    "dominant_intervened_failure_pattern",
    "rescued_failures_by_baseline_pattern",
    "baseline_pattern_distribution",
    "intervened_pattern_distribution",
    "timeout_correlated_delta_count",
    "idle_correlated_delta_count",
    "syndrome_burst_delta_count",
    "syndrome_decoding_mismatch_delta_count",
]

SANITY_FIELDS = [
    "check_name",
    "intervention",
    "expected_direction",
    "observed_metric",
    "passed",
    "notes",
]

KEY_PATTERNS = [
    "timeout_correlated",
    "idle_correlated",
    "syndrome_burst",
    "syndrome_decoding_mismatch",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/results/p2_intervened_runs.csv")
    parser.add_argument("--pattern-output", default="data/results/p2_5_pattern_shift_summary.csv")
    parser.add_argument("--sanity-output", default="data/results/p2_5_sanity_checks.csv")
    args = parser.parse_args()

    rows = read_csv_rows(args.input)
    pattern_rows = summarize_pattern_shifts(rows)
    sanity_rows = run_sanity_checks(pattern_rows)
    write_csv_rows(args.pattern_output, pattern_rows, PATTERN_FIELDS)
    write_csv_rows(args.sanity_output, sanity_rows, SANITY_FIELDS)
    print(f"wrote {len(pattern_rows)} pattern-shift rows to {args.pattern_output}")
    print(f"wrote {len(sanity_rows)} sanity-check rows to {args.sanity_output}")


def summarize_pattern_shifts(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["intervention"])].append(row)

    summary_rows = []
    for intervention, group in groups.items():
        baseline_patterns = Counter(str(row["baseline_failure_pattern"]) for row in group)
        intervened_patterns = Counter(str(row["intervened_failure_pattern"]) for row in group)
        rescued_patterns = Counter(
            str(row["baseline_failure_pattern"])
            for row in group
            if parse_bool(row["rescued_failure"])
        )
        baseline_failure_count = sum(parse_bool(row["baseline_logical_failure"]) for row in group)
        intervened_failure_count = sum(parse_bool(row["intervened_logical_failure"]) for row in group)
        out = {
            "intervention": intervention,
            "num_rows": len(group),
            "baseline_failure_count": baseline_failure_count,
            "intervened_failure_count": intervened_failure_count,
            "rescued_failure_count": sum(parse_bool(row["rescued_failure"]) for row in group),
            "dominant_baseline_failure_pattern": dominant_failure_pattern(baseline_patterns),
            "dominant_intervened_failure_pattern": dominant_failure_pattern(intervened_patterns),
            "rescued_failures_by_baseline_pattern": dump_counts(rescued_patterns),
            "baseline_pattern_distribution": dump_counts(baseline_patterns),
            "intervened_pattern_distribution": dump_counts(intervened_patterns),
        }
        for pattern in KEY_PATTERNS:
            out[f"{pattern}_delta_count"] = (
                intervened_patterns.get(pattern, 0) - baseline_patterns.get(pattern, 0)
            )
        summary_rows.append(out)

    summary_rows.sort(key=lambda row: (int(row["intervened_failure_count"]) - int(row["baseline_failure_count"]), row["intervention"]))
    return summary_rows


def run_sanity_checks(pattern_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_intervention = {str(row["intervention"]): row for row in pattern_rows}
    checks = [
        (
            "data_noise_reduces_failures",
            "remove_data_noise",
            "total intervened failures should be lower than baseline",
            lambda row: int(row["intervened_failure_count"]) - int(row["baseline_failure_count"]),
            lambda value: value < 0,
            "Current P2 pattern vocabulary does not separate data-dominated failures.",
        ),
        (
            "measurement_noise_reduces_failures",
            "remove_measurement_noise",
            "total intervened failures should be lower than baseline",
            lambda row: int(row["intervened_failure_count"]) - int(row["baseline_failure_count"]),
            lambda value: value < 0,
            "Current P2 pattern vocabulary does not separate measurement-dominated failures.",
        ),
        (
            "timeout_intervention_reduces_timeout_pattern",
            "remove_decoder_timeout",
            "timeout_correlated failures should decrease",
            lambda row: int(row["timeout_correlated_delta_count"]),
            lambda value: value < 0,
            "",
        ),
        (
            "backlog_intervention_reduces_runtime_patterns",
            "eliminate_decoder_backlog",
            "timeout_correlated plus syndrome_burst failures should decrease",
            lambda row: int(row["timeout_correlated_delta_count"]) + int(row["syndrome_burst_delta_count"]),
            lambda value: value < 0,
            "decoder_backlog is represented through timeout/delay/syndrome pressure patterns.",
        ),
        (
            "capacity_intervention_reduces_runtime_patterns",
            "increase_decoder_capacity_2x",
            "timeout_correlated plus syndrome_burst failures should decrease",
            lambda row: int(row["timeout_correlated_delta_count"]) + int(row["syndrome_burst_delta_count"]),
            lambda value: value < 0,
            "capacity affects decoder_backlog in the runtime proxy model.",
        ),
        (
            "idle_exposure_intervention_reduces_idle_pattern",
            "reduce_idle_exposure_50pct",
            "idle_correlated failures should decrease or total failures should decrease",
            lambda row: (
                int(row["idle_correlated_delta_count"]),
                int(row["intervened_failure_count"]) - int(row["baseline_failure_count"]),
            ),
            lambda value: value[0] < 0 or value[1] < 0,
            "",
        ),
    ]

    rows = []
    for check_name, intervention, expected, metric_fn, pass_fn, notes in checks:
        row = by_intervention.get(intervention)
        if row is None:
            rows.append(
                {
                    "check_name": check_name,
                    "intervention": intervention,
                    "expected_direction": expected,
                    "observed_metric": "missing",
                    "passed": False,
                    "notes": "intervention not present in input",
                }
            )
            continue
        observed = metric_fn(row)
        rows.append(
            {
                "check_name": check_name,
                "intervention": intervention,
                "expected_direction": expected,
                "observed_metric": str(observed),
                "passed": pass_fn(observed),
                "notes": notes,
            }
        )
    return rows


def dominant_failure_pattern(counter: Counter[str]) -> str:
    filtered = Counter({key: value for key, value in counter.items() if key != "no_failure"})
    if not filtered:
        return "no_failure"
    return filtered.most_common(1)[0][0]


def dump_counts(counter: Counter[str]) -> str:
    return json.dumps(dict(sorted(counter.items())), sort_keys=True, separators=(",", ":"))


if __name__ == "__main__":
    main()

