"""P3 attribution summaries and baseline-method comparisons."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict

from failureops.io_utils import fmt_float, parse_bool
from failureops.metrics import summarize_attribution


def summarize_p3_eval_matrix(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    per_run: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["run_id"]),
            str(row["workload_id"]),
            str(row["stress_level"]),
            str(row["intervention"]),
        )
        per_run[key].append(row)

    per_intervention: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    intervention_class = {}
    for (_run_id, workload_id, stress_level, intervention), group in per_run.items():
        summary = summarize_attribution(group)[0]
        key = (workload_id, stress_level, intervention)
        per_intervention[key].append(summary)
        intervention_class[key] = str(group[0].get("intervention_class", "unknown"))

    out = []
    for (workload_id, stress_level, intervention), summaries in per_intervention.items():
        deltas = [float(row["absolute_delta_lfr"]) for row in summaries]
        rescue_rates = [float(row["rescue_rate"]) for row in summaries]
        baseline_lfrs = [float(row["baseline_logical_failure_rate"]) for row in summaries]
        intervened_lfrs = [float(row["intervened_logical_failure_rate"]) for row in summaries]
        std_delta = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
        out.append(
            {
                "workload_id": workload_id,
                "stress_level": stress_level,
                "intervention": intervention,
                "num_groups": len(summaries),
                "num_shots": sum(int(row["num_shots"]) for row in summaries),
                "mean_baseline_lfr": fmt_float(statistics.fmean(baseline_lfrs)),
                "mean_intervened_lfr": fmt_float(statistics.fmean(intervened_lfrs)),
                "mean_absolute_delta_lfr": fmt_float(statistics.fmean(deltas)),
                "std_absolute_delta_lfr": fmt_float(std_delta),
                "ci95_absolute_delta_lfr": fmt_float(1.96 * std_delta / math.sqrt(len(deltas))),
                "mean_rescue_rate": fmt_float(statistics.fmean(rescue_rates)),
                "intervention_class": intervention_class[(workload_id, stress_level, intervention)],
            }
        )
    out.sort(key=lambda row: (row["workload_id"], row["stress_level"], float(row["mean_absolute_delta_lfr"]), row["intervention"]))
    return out


def compare_p3_baseline_methods(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["workload_id"]), str(row["stress_level"]))].append(row)

    out = []
    for (workload_id, stress_level), group in grouped.items():
        out.append(method_row("paired_counterfactual", workload_id, stress_level, group))
        out.append(method_row("noise_only_attribution", workload_id, stress_level, [
            row for row in group if row.get("intervention_class") == "noise"
        ]))
        out.append(method_row("runtime_only_attribution", workload_id, stress_level, [
            row for row in group if row.get("intervention_class") == "runtime"
        ]))
        out.append(plain_lfr_row(workload_id, stress_level, group))
        out.append(static_label_row(workload_id, stress_level, group))
    out.sort(key=lambda row: (row["workload_id"], row["stress_level"], row["method"]))
    return out


def method_row(
    method: str,
    workload_id: str,
    stress_level: str,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    if not rows:
        return empty_method_row(method, workload_id, stress_level)
    summaries = summarize_attribution(rows)
    deltas = [float(row["absolute_delta_lfr"]) for row in summaries]
    rescue_rates = [float(row["rescue_rate"]) for row in summaries]
    top = min(summaries, key=lambda row: float(row["absolute_delta_lfr"]))
    std_delta = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    return {
        "method": method,
        "workload_id": workload_id,
        "stress_level": stress_level,
        "num_groups": len(summaries),
        "num_shots": max(int(row["num_shots"]) for row in summaries),
        "mean_absolute_delta_lfr": fmt_float(statistics.fmean(deltas)),
        "std_absolute_delta_lfr": fmt_float(std_delta),
        "ci95_absolute_delta_lfr": fmt_float(1.96 * std_delta / math.sqrt(len(deltas))),
        "mean_rescue_rate": fmt_float(statistics.fmean(rescue_rates)),
        "top_intervention": top["intervention"],
        "top_intervention_delta_lfr": top["absolute_delta_lfr"],
    }


def plain_lfr_row(
    workload_id: str,
    stress_level: str,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    unique = unique_baseline_shots(rows)
    failure_rate = sum(parse_bool(row["baseline_logical_failure"]) for row in unique) / len(unique) if unique else 0.0
    return {
        "method": "plain_lfr_reporting",
        "workload_id": workload_id,
        "stress_level": stress_level,
        "num_groups": 1,
        "num_shots": len(unique),
        "mean_absolute_delta_lfr": "0.000000",
        "std_absolute_delta_lfr": "0.000000",
        "ci95_absolute_delta_lfr": "0.000000",
        "mean_rescue_rate": "0.000000",
        "top_intervention": f"baseline_lfr={fmt_float(failure_rate)}",
        "top_intervention_delta_lfr": "0.000000",
    }


def static_label_row(
    workload_id: str,
    stress_level: str,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    unique = unique_baseline_shots(rows)
    patterns = Counter(
        str(row["baseline_failure_pattern"])
        for row in unique
        if str(row["baseline_failure_pattern"]) not in {"", "none", "no_failure"}
    )
    top_pattern, top_count = patterns.most_common(1)[0] if patterns else ("no_failure", 0)
    return {
        "method": "static_label_attribution",
        "workload_id": workload_id,
        "stress_level": stress_level,
        "num_groups": len(patterns),
        "num_shots": len(unique),
        "mean_absolute_delta_lfr": "0.000000",
        "std_absolute_delta_lfr": "0.000000",
        "ci95_absolute_delta_lfr": "0.000000",
        "mean_rescue_rate": "0.000000",
        "top_intervention": f"dominant_pattern={top_pattern}:{top_count}",
        "top_intervention_delta_lfr": "0.000000",
    }


def unique_baseline_shots(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seen = set()
    unique = []
    for row in rows:
        key = (row.get("run_id"), row.get("workload_id"), row.get("stress_level"), row.get("shot_id"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def empty_method_row(method: str, workload_id: str, stress_level: str) -> dict[str, object]:
    return {
        "method": method,
        "workload_id": workload_id,
        "stress_level": stress_level,
        "num_groups": 0,
        "num_shots": 0,
        "mean_absolute_delta_lfr": "0.000000",
        "std_absolute_delta_lfr": "0.000000",
        "ci95_absolute_delta_lfr": "0.000000",
        "mean_rescue_rate": "0.000000",
        "top_intervention": "none",
        "top_intervention_delta_lfr": "0.000000",
    }

