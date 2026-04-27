"""Paired treatment-effect summaries for P4."""

from __future__ import annotations

from collections import defaultdict

from failureops.bootstrap import bootstrap_ci, net_rescue_rate, paired_counts, paired_delta_lfr
from failureops.io_utils import fmt_float, parse_bool


def summarize_paired_effects(
    rows: list[dict[str, object]],
    *,
    num_bootstrap: int = 1000,
    bootstrap_seed: int = 2026,
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["workload_id"]), str(row["stress_level"]), str(row["intervention"]))].append(row)

    out = []
    for (workload_id, stress_level, intervention), group in groups.items():
        valid = [row for row in group if parse_bool(row.get("pairing_valid", False))]
        invalid_count = len(group) - len(valid)
        counts = paired_counts(valid)
        baseline_success_count = len(valid) - counts["baseline_failure_count"]
        delta = paired_delta_lfr(valid)
        net_rate = net_rescue_rate(valid)
        delta_ci = bootstrap_ci(
            valid,
            "paired_delta_lfr",
            num_resamples=num_bootstrap,
            seed=bootstrap_seed + stable_group_offset(workload_id, stress_level, intervention),
        )
        net_ci = bootstrap_ci(
            valid,
            "net_rescue_rate",
            num_resamples=num_bootstrap,
            seed=bootstrap_seed + stable_group_offset(intervention, stress_level, workload_id),
        )
        out.append(
            {
                "workload_id": workload_id,
                "stress_level": stress_level,
                "intervention": intervention,
                "intervention_class": str(group[0].get("intervention_class", "unknown")),
                "num_pairs": len(group),
                "valid_pairs": len(valid),
                "invalid_pairs": invalid_count,
                **counts,
                "paired_delta_lfr": fmt_float(delta),
                "paired_delta_lfr_ci_lower": fmt_float(delta_ci[0]),
                "paired_delta_lfr_ci_upper": fmt_float(delta_ci[1]),
                "net_rescue_rate": fmt_float(net_rate),
                "net_rescue_rate_ci_lower": fmt_float(net_ci[0]),
                "net_rescue_rate_ci_upper": fmt_float(net_ci[1]),
                "rescue_rate_among_baseline_failures": fmt_float(
                    counts["rescued_failure_count"] / counts["baseline_failure_count"]
                    if counts["baseline_failure_count"]
                    else 0.0
                ),
                "induction_rate_among_baseline_successes": fmt_float(
                    counts["induced_failure_count"] / baseline_success_count
                    if baseline_success_count
                    else 0.0
                ),
                "num_bootstrap": num_bootstrap,
                "bootstrap_seed": bootstrap_seed,
            }
        )
    out.sort(key=lambda row: (row["workload_id"], row["stress_level"], float(row["paired_delta_lfr"]), row["intervention"]))
    return out


def stable_group_offset(*items: str) -> int:
    text = "|".join(items)
    return sum((index + 1) * ord(char) for index, char in enumerate(text))

