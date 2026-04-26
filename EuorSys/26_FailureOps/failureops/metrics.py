"""Attribution metrics for the P0 intervention table."""

from __future__ import annotations

from collections import Counter, defaultdict

from failureops.io_utils import fmt_float, parse_bool


def summarize_attribution(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["intervention"])].append(row)

    summaries = [_summarize_one(intervention, group) for intervention, group in groups.items()]
    summaries.sort(key=lambda row: (float(row["absolute_delta_lfr"]), row["intervention"]))
    return summaries


def _summarize_one(intervention: str, rows: list[dict[str, object]]) -> dict[str, object]:
    num_shots = len(rows)
    baseline_failure_count = sum(parse_bool(row["baseline_logical_failure"]) for row in rows)
    intervened_failure_count = sum(parse_bool(row["intervened_logical_failure"]) for row in rows)
    rescued_failure_count = sum(parse_bool(row["rescued_failure"]) for row in rows)
    new_failure_count = sum(parse_bool(row["new_failure"]) for row in rows)

    baseline_lfr = _safe_rate(baseline_failure_count, num_shots)
    intervened_lfr = _safe_rate(intervened_failure_count, num_shots)
    absolute_delta = intervened_lfr - baseline_lfr
    relative_delta = absolute_delta / baseline_lfr if baseline_lfr else 0.0

    return {
        "intervention": intervention,
        "num_shots": num_shots,
        "baseline_logical_failure_rate": fmt_float(baseline_lfr),
        "intervened_logical_failure_rate": fmt_float(intervened_lfr),
        "absolute_delta_lfr": fmt_float(absolute_delta),
        "relative_delta_lfr": fmt_float(relative_delta),
        "baseline_failure_count": baseline_failure_count,
        "intervened_failure_count": intervened_failure_count,
        "rescued_failure_count": rescued_failure_count,
        "new_failure_count": new_failure_count,
        "rescue_rate": fmt_float(_safe_rate(rescued_failure_count, baseline_failure_count)),
        "new_failure_rate": fmt_float(_safe_rate(new_failure_count, num_shots)),
        "dominant_baseline_failure_pattern": _dominant_pattern(rows, "baseline_failure_pattern"),
        "dominant_intervened_failure_pattern": _dominant_pattern(rows, "intervened_failure_pattern"),
    }


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _dominant_pattern(rows: list[dict[str, object]], field: str) -> str:
    patterns = [
        str(row[field])
        for row in rows
        if str(row[field]) not in {"", "none", "no_failure"}
    ]
    if not patterns:
        return "no_failure"
    return Counter(patterns).most_common(1)[0][0]

