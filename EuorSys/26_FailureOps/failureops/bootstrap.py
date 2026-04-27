"""Bootstrap confidence intervals for paired treatment-effect metrics."""

from __future__ import annotations

import random

from failureops.io_utils import parse_bool


def paired_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    rescued = 0
    induced = 0
    unchanged_failure = 0
    unchanged_success = 0
    baseline_failure = 0
    intervened_failure = 0
    for row in rows:
        baseline = parse_bool(row["baseline_logical_failure"])
        intervened = parse_bool(row["intervened_logical_failure"])
        baseline_failure += int(baseline)
        intervened_failure += int(intervened)
        if baseline and not intervened:
            rescued += 1
        elif (not baseline) and intervened:
            induced += 1
        elif baseline and intervened:
            unchanged_failure += 1
        else:
            unchanged_success += 1
    return {
        "baseline_failure_count": baseline_failure,
        "intervened_failure_count": intervened_failure,
        "rescued_failure_count": rescued,
        "induced_failure_count": induced,
        "unchanged_failure_count": unchanged_failure,
        "unchanged_success_count": unchanged_success,
        "net_rescue_count": rescued - induced,
    }


def paired_delta_lfr(rows: list[dict[str, object]]) -> float:
    if not rows:
        return 0.0
    counts = paired_counts(rows)
    return (counts["induced_failure_count"] - counts["rescued_failure_count"]) / len(rows)


def net_rescue_rate(rows: list[dict[str, object]]) -> float:
    if not rows:
        return 0.0
    counts = paired_counts(rows)
    return counts["net_rescue_count"] / len(rows)


def bootstrap_ci(
    rows: list[dict[str, object]],
    metric_name: str,
    *,
    num_resamples: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if not rows:
        return (0.0, 0.0)
    metric_fn = {
        "paired_delta_lfr": paired_delta_lfr,
        "net_rescue_rate": net_rescue_rate,
    }[metric_name]
    rng = random.Random(seed)
    values = []
    for _ in range(num_resamples):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        values.append(metric_fn(sample))
    values.sort()
    lower_index = int((alpha / 2.0) * (len(values) - 1))
    upper_index = int((1.0 - alpha / 2.0) * (len(values) - 1))
    return values[lower_index], values[upper_index]

