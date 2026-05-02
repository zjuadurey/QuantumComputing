#!/usr/bin/env python
"""Build EuroSys-main evidence tables over P7/P8 artifacts."""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from failureops.data_model import (
    P10_BASELINE_COMPARISON_FIELDS,
    P10_ROBUSTNESS_FIELDS,
    P10_RUNTIME_DEADLINE_INTERVENTION_FIELDS,
    P10_RUNTIME_DEADLINE_SUMMARY_FIELDS,
)
from failureops.io_utils import ensure_parent_dir, fmt_float, parse_bool, parse_int, read_csv_rows, write_csv_rows
from failureops.manifest import write_manifest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/failureops-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effect-matrix", default="data/results/p7_google_rl_qec_decoder_effect_matrix.csv")
    parser.add_argument("--variance", default="data/results/p7_5_paired_vs_unpaired_variance.csv")
    parser.add_argument("--features", default="data/results/p7_5_rescue_induction_features.csv")
    parser.add_argument("--prior-effects", default="data/results/p7_5_decoder_prior_interventions.csv")
    parser.add_argument("--p8-trace", default="data/results/p8_decoder_runtime_trace.csv")
    parser.add_argument("--deadlines-us", default="4,5,6,7,8")
    parser.add_argument("--baseline-output", default="data/results/p10_baseline_comparison.csv")
    parser.add_argument("--robustness-output", default="data/results/p10_realdata_robustness.csv")
    parser.add_argument(
        "--runtime-intervention-output",
        default="data/results/p10_runtime_deadline_interventions.csv",
    )
    parser.add_argument("--runtime-summary-output", default="data/results/p10_runtime_deadline_summary.csv")
    parser.add_argument("--figure-output", default="figures/p10_eurosys_main_evidence.png")
    parser.add_argument("--manifest-output", default="data/results/p10_eurosys_main_evidence_manifest.json")
    args = parser.parse_args()

    for path in (args.effect_matrix, args.variance, args.features, args.prior_effects, args.p8_trace):
        if not Path(path).exists():
            raise FileNotFoundError(f"required P10 input is missing: {path}")

    effect_rows = read_csv_rows(args.effect_matrix)
    variance_rows = read_csv_rows(args.variance)
    feature_rows = read_csv_rows(args.features)
    prior_rows = read_csv_rows(args.prior_effects)
    p8_trace_rows = read_csv_rows(args.p8_trace)
    deadlines = parse_deadlines(args.deadlines_us)

    baseline_rows = build_baseline_comparison(effect_rows, variance_rows, feature_rows, args)
    robustness_rows = build_robustness_table(effect_rows, variance_rows, prior_rows, args)
    runtime_intervention_rows, runtime_summary_rows = build_runtime_deadline_closure(
        p8_trace_rows,
        deadlines=deadlines,
        source_artifact=args.p8_trace,
    )

    write_csv_rows(args.baseline_output, baseline_rows, P10_BASELINE_COMPARISON_FIELDS)
    write_csv_rows(args.robustness_output, robustness_rows, P10_ROBUSTNESS_FIELDS)
    write_csv_rows(
        args.runtime_intervention_output,
        runtime_intervention_rows,
        P10_RUNTIME_DEADLINE_INTERVENTION_FIELDS,
    )
    write_csv_rows(args.runtime_summary_output, runtime_summary_rows, P10_RUNTIME_DEADLINE_SUMMARY_FIELDS)
    plot_p10(
        baseline_rows=baseline_rows,
        effect_rows=effect_rows,
        runtime_summary_rows=runtime_summary_rows,
        prior_rows=prior_rows,
        output=args.figure_output,
    )
    write_manifest(
        args.manifest_output,
        config={
            "experiment_id": "p10_eurosys_main_evidence",
            "inputs": {
                "effect_matrix": args.effect_matrix,
                "variance": args.variance,
                "features": args.features,
                "prior_effects": args.prior_effects,
                "p8_trace": args.p8_trace,
                "deadlines_us": deadlines,
            },
        },
        command=sys.argv,
        outputs={
            "baseline_comparison": args.baseline_output,
            "robustness": args.robustness_output,
            "runtime_interventions": args.runtime_intervention_output,
            "runtime_summary": args.runtime_summary_output,
            "figure": args.figure_output,
        },
        row_counts={
            "baseline_comparison": len(baseline_rows),
            "robustness": len(robustness_rows),
            "runtime_interventions": len(runtime_intervention_rows),
            "runtime_summary": len(runtime_summary_rows),
        },
    )
    print(f"wrote {len(baseline_rows)} P10 baseline-comparison rows to {args.baseline_output}")
    print(f"wrote {len(robustness_rows)} P10 robustness rows to {args.robustness_output}")
    print(f"wrote {len(runtime_intervention_rows)} P10 runtime-deadline intervention rows to {args.runtime_intervention_output}")
    print(f"wrote {len(runtime_summary_rows)} P10 runtime-deadline summary rows to {args.runtime_summary_output}")
    print(f"wrote P10 evidence figure to {args.figure_output}")
    print(f"wrote P10 manifest to {args.manifest_output}")


def build_baseline_comparison(
    effect_rows: list[dict[str, str]],
    variance_rows: list[dict[str, str]],
    feature_rows: list[dict[str, str]],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    failureops_scores = {
        row["workload_id"]: abs(float(row["paired_delta_lfr"]))
        for row in effect_rows
    }
    failureops_rank = rank_desc(failureops_scores)
    static_detector_scores = condition_detector_scores(feature_rows)
    baseline_lfr_scores = {row["workload_id"]: float(row["baseline_lfr"]) for row in effect_rows}
    plain_delta_scores = {
        row["workload_id"]: abs(float(row["intervened_lfr"]) - float(row["baseline_lfr"]))
        for row in effect_rows
    }
    unpaired_penalty_scores = {
        row["workload_id"]: float(row["unpaired_bootstrap_std"])
        for row in variance_rows
    }
    ratio_values = [float(row["std_ratio_unpaired_over_paired"]) for row in variance_rows]
    rows = [
        method_row(
            method="FailureOps paired intervention sensitivity",
            comparison_scope="real detector records",
            ranking_target="abs paired_delta_lfr",
            scores=failureops_scores,
            reference_rank=failureops_rank,
            mean_abs_delta=statistics.fmean(failureops_scores.values()),
            mean_uncertainty=0.0,
            interpretation="Directly ranks conditions by paired intervention sensitivity.",
            limitation="Requires matched baseline and intervened outcomes.",
            source_artifact=args.effect_matrix,
        ),
        method_row(
            method="Plain baseline LFR",
            comparison_scope="real detector records",
            ranking_target="baseline_lfr",
            scores=baseline_lfr_scores,
            reference_rank=failureops_rank,
            mean_abs_delta=statistics.fmean(failureops_scores.values()),
            mean_uncertainty=0.0,
            interpretation="Baseline LFR tracks difficult conditions but does not expose rescued or induced outcomes.",
            limitation="Does not by itself identify which intervention changes failure behavior.",
            source_artifact=args.effect_matrix,
        ),
        method_row(
            method="Static detector burden",
            comparison_scope="real detector records",
            ranking_target="mean_detector_event_count",
            scores=static_detector_scores,
            reference_rank=failureops_rank,
            mean_abs_delta=statistics.fmean(failureops_scores.values()),
            mean_uncertainty=0.0,
            interpretation="Syndrome burden is a descriptive proxy for difficult shots.",
            limitation="Static detector counts do not measure intervention effect.",
            source_artifact=args.features,
        ),
        method_row(
            method="Plain LFR delta",
            comparison_scope="real detector records",
            ranking_target="abs(intervened_lfr-baseline_lfr)",
            scores=plain_delta_scores,
            reference_rank=failureops_rank,
            mean_abs_delta=statistics.fmean(plain_delta_scores.values()),
            mean_uncertainty=0.0,
            interpretation="Aggregated LFR deltas agree numerically here because P7/P7.5 is paired.",
            limitation="Without pairing it cannot distinguish rescued and induced transitions on the same records.",
            source_artifact=args.effect_matrix,
        ),
        {
            "method": "Unpaired bootstrap estimator",
            "comparison_scope": "real detector records",
            "ranking_target": "delta_lfr uncertainty",
            "top_item": "not_applicable",
            "top_value": fmt_float(statistics.fmean(ratio_values)),
            "top3_overlap_with_failureops": "",
            "spearman_with_failureops": "",
            "mean_absolute_delta_lfr": fmt_float(statistics.fmean(failureops_scores.values())),
            "mean_uncertainty_metric": fmt_float(statistics.fmean(unpaired_penalty_scores.values())),
            "interpretation": "Unpaired resampling has higher uncertainty than paired resampling.",
            "limitation": "Unpaired uncertainty is an estimator comparison, not an attribution method.",
            "source_artifact": args.variance,
        },
    ]
    return rows


def method_row(
    *,
    method: str,
    comparison_scope: str,
    ranking_target: str,
    scores: dict[str, float],
    reference_rank: list[str],
    mean_abs_delta: float,
    mean_uncertainty: float,
    interpretation: str,
    limitation: str,
    source_artifact: str,
) -> dict[str, object]:
    ranking = rank_desc(scores)
    top = ranking[0] if ranking else ""
    return {
        "method": method,
        "comparison_scope": comparison_scope,
        "ranking_target": ranking_target,
        "top_item": top,
        "top_value": fmt_float(scores.get(top, 0.0)),
        "top3_overlap_with_failureops": fmt_float(top_k_overlap(reference_rank, ranking, top_k=3)),
        "spearman_with_failureops": fmt_float(spearman_from_rankings(reference_rank, ranking)),
        "mean_absolute_delta_lfr": fmt_float(mean_abs_delta),
        "mean_uncertainty_metric": fmt_float(mean_uncertainty),
        "interpretation": interpretation,
        "limitation": limitation,
        "source_artifact": source_artifact,
    }


def condition_detector_scores(feature_rows: list[dict[str, str]]) -> dict[str, float]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in feature_rows:
        grouped[row["workload_id"]].append(row)
    out = {}
    for workload_id, rows in grouped.items():
        total = sum(parse_int(row["num_shots"]) for row in rows)
        if total:
            out[workload_id] = sum(
                float(row["mean_detector_event_count"]) * parse_int(row["num_shots"])
                for row in rows
            ) / total
    return out


def build_robustness_table(
    effect_rows: list[dict[str, str]],
    variance_rows: list[dict[str, str]],
    prior_rows: list[dict[str, str]],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    deltas = [float(row["paired_delta_lfr"]) for row in effect_rows]
    cycle_means = cycle_mean_deltas(effect_rows)
    slope, corr = linear_slope_corr(list(cycle_means.keys()), list(cycle_means.values()))
    group_rankings = cycle_rankings_by_group(effect_rows)
    overlaps = []
    distances = []
    groups = sorted(group_rankings)
    for i, left in enumerate(groups):
        for right in groups[i + 1:]:
            overlaps.append(top_k_overlap(group_rankings[left], group_rankings[right], top_k=3))
            distances.append(pairwise_rank_distance(group_rankings[left], group_rankings[right]))
    ratios = [float(row["std_ratio_unpaired_over_paired"]) for row in variance_rows]
    prior_significant = [
        row for row in prior_rows
        if float(row["paired_delta_lfr_ci_lower"]) > 0 or float(row["paired_delta_lfr_ci_upper"]) < 0
    ]
    return [
        {
            "check_id": "R1",
            "scope": "all P7/P7.5 conditions",
            "num_units": len(effect_rows),
            "metric": "negative_delta_fraction",
            "value": fmt_float(sum(delta < 0.0 for delta in deltas) / len(deltas)),
            "supporting_value": f"min={fmt_float(min(deltas))};max={fmt_float(max(deltas))}",
            "interpretation": "Decoder-pathway intervention improves LFR in every observed real-data condition.",
            "limitation": "All conditions come from one public Google RL QEC dataset family.",
            "source_artifact": args.effect_matrix,
        },
        {
            "check_id": "R2",
            "scope": "cycle-level mean sensitivity",
            "num_units": len(cycle_means),
            "metric": "slope_per_10_cycles",
            "value": fmt_float(slope * 10.0),
            "supporting_value": f"corr={fmt_float(corr)}",
            "interpretation": "Mean sensitivity becomes stronger as memory duration increases.",
            "limitation": "This is a trend over observed cycle settings, not a mechanistic scaling law.",
            "source_artifact": args.effect_matrix,
        },
        {
            "check_id": "R3",
            "scope": "cycle rankings across control_mode and basis groups",
            "num_units": len(groups),
            "metric": "mean_top3_overlap",
            "value": fmt_float(statistics.fmean(overlaps) if overlaps else 1.0),
            "supporting_value": f"mean_rank_distance={fmt_float(statistics.fmean(distances) if distances else 0.0)}",
            "interpretation": "The most sensitive cycle regimes are broadly stable across real-data subgroups.",
            "limitation": "Ranks compare cycle regimes, not multiple independent QEC datasets.",
            "source_artifact": args.effect_matrix,
        },
        {
            "check_id": "R4",
            "scope": "paired-vs-unpaired bootstrap",
            "num_units": len(variance_rows),
            "metric": "mean_std_ratio_unpaired_over_paired",
            "value": fmt_float(statistics.fmean(ratios)),
            "supporting_value": f"min={fmt_float(min(ratios))};max={fmt_float(max(ratios))}",
            "interpretation": "Pairing reduces estimator variance in every observed condition.",
            "limitation": "Bootstrap uncertainty does not replace external replication.",
            "source_artifact": args.variance,
        },
        {
            "check_id": "R5",
            "scope": "single-condition decoder-prior sweep",
            "num_units": len(prior_rows),
            "metric": "ci_excludes_zero_fraction",
            "value": fmt_float(len(prior_significant) / len(prior_rows) if prior_rows else 0.0),
            "supporting_value": f"significant={len(prior_significant)}",
            "interpretation": "Most prior variants in the focused sweep have nonzero observed effects.",
            "limitation": "The current prior sweep covers one real-data condition.",
            "source_artifact": args.prior_effects,
        },
    ]


def build_runtime_deadline_closure(
    p8_rows: list[dict[str, str]],
    *,
    deadlines: list[float],
    source_artifact: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    intervention_rows = []
    summary_rows = []
    service_times = [float(row["decoder_service_time"]) * 1_000_000.0 for row in p8_rows]
    p95_service = percentile(service_times, 0.95)
    mean_service = statistics.fmean(service_times)
    baseline_failures = [parse_bool(row["measured_logical_failure"]) for row in p8_rows]
    for deadline in deadlines:
        miss_count = 0
        rescued = 0
        induced = 0
        unchanged_failure = 0
        unchanged_success = 0
        intervened_failure_count = 0
        for row, baseline_failure, service_us in zip(p8_rows, baseline_failures, service_times):
            deadline_miss = service_us > deadline
            miss_count += int(deadline_miss)
            baseline_prediction = parse_bool(row["measured_decoder_prediction"])
            observable = parse_bool(row["observable_flip"])
            intervened_prediction = False if deadline_miss else baseline_prediction
            intervened_failure = intervened_prediction != observable
            intervened_failure_count += int(intervened_failure)
            if baseline_failure and not intervened_failure:
                rescued += 1
            elif (not baseline_failure) and intervened_failure:
                induced += 1
            elif baseline_failure and intervened_failure:
                unchanged_failure += 1
            else:
                unchanged_success += 1
            intervention_rows.append(
                {
                    "pair_id": f"deadline_{deadline:g}|{row['shot_id']}",
                    "shot_id": row["shot_id"],
                    "deadline_us": fmt_float(deadline),
                    "decoder_service_time_us": fmt_float(service_us),
                    "deadline_miss": deadline_miss,
                    "baseline_logical_failure": baseline_failure,
                    "intervened_logical_failure": intervened_failure,
                    "rescued_failure": baseline_failure and not intervened_failure,
                    "induced_failure": (not baseline_failure) and intervened_failure,
                    "baseline_prediction": baseline_prediction,
                    "intervened_prediction": intervened_prediction,
                    "observable_flip": observable,
                }
            )
        num_pairs = len(p8_rows)
        baseline_lfr = sum(baseline_failures) / num_pairs if num_pairs else 0.0
        intervened_lfr = intervened_failure_count / num_pairs if num_pairs else 0.0
        summary_rows.append(
            {
                "deadline_us": fmt_float(deadline),
                "num_pairs": num_pairs,
                "deadline_miss_count": miss_count,
                "deadline_miss_rate": fmt_float(miss_count / num_pairs if num_pairs else 0.0),
                "baseline_lfr": fmt_float(baseline_lfr),
                "intervened_lfr": fmt_float(intervened_lfr),
                "paired_delta_lfr": fmt_float(intervened_lfr - baseline_lfr),
                "rescued_failure_count": rescued,
                "induced_failure_count": induced,
                "unchanged_failure_count": unchanged_failure,
                "unchanged_success_count": unchanged_success,
                "mean_service_time_us": fmt_float(mean_service),
                "p95_service_time_us": fmt_float(p95_service),
                "interpretation": "Deadline intervention drops late decoder predictions to a default no-flip correction.",
                "source_artifact": source_artifact,
            }
        )
    return intervention_rows, summary_rows


def plot_p10(
    *,
    baseline_rows: list[dict[str, object]],
    effect_rows: list[dict[str, str]],
    runtime_summary_rows: list[dict[str, object]],
    prior_rows: list[dict[str, str]],
    output: str,
) -> None:
    ensure_parent_dir(output)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    plot_baseline_comparison(axes[0][0], baseline_rows)
    plot_grouped_cycle_robustness(axes[0][1], effect_rows)
    plot_runtime_deadline_closure(axes[1][0], runtime_summary_rows)
    plot_prior_effects(axes[1][1], prior_rows)
    fig.suptitle("P10 EuroSys-main evidence pass", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_baseline_comparison(ax, rows: list[dict[str, object]]) -> None:
    comparable = [row for row in rows if row["spearman_with_failureops"] not in {"", None}]
    labels = [str(row["method"]).replace(" ", "\n") for row in comparable]
    values = [float(row["spearman_with_failureops"]) for row in comparable]
    ax.bar(labels, values, color="#4b78a8")
    ax.set_ylim(-1.0, 1.0)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_title("Baseline ranking agreement")
    ax.set_ylabel("Spearman vs FailureOps")
    ax.tick_params(axis="x", labelsize=7)


def plot_grouped_cycle_robustness(ax, rows: list[dict[str, str]]) -> None:
    grouped: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[(row["control_mode"], row["basis"])][parse_int(row["cycles"])].append(float(row["paired_delta_lfr"]))
    for (control, basis), by_cycle in sorted(grouped.items()):
        cycles = sorted(by_cycle)
        values = [statistics.fmean(by_cycle[cycle]) for cycle in cycles]
        ax.plot(cycles, values, marker="o", linewidth=1.4, label=f"{control} {basis}")
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_title("Real-data robustness by subgroup")
    ax.set_xlabel("QEC cycles")
    ax.set_ylabel("paired delta LFR")
    ax.legend(fontsize=6)


def plot_runtime_deadline_closure(ax, rows: list[dict[str, object]]) -> None:
    deadlines = [float(row["deadline_us"]) for row in rows]
    deltas = [float(row["paired_delta_lfr"]) for row in rows]
    miss_rates = [float(row["deadline_miss_rate"]) for row in rows]
    ax.plot(deadlines, deltas, marker="o", color="#b55335", label="delta LFR")
    ax.set_xlabel("deadline (us)")
    ax.set_ylabel("paired delta LFR")
    ax2 = ax.twinx()
    ax2.plot(deadlines, miss_rates, marker="s", color="#5f7f4f", label="miss rate")
    ax2.set_ylabel("deadline miss rate")
    ax.set_title("Measured-runtime deadline closure")


def plot_prior_effects(ax, rows: list[dict[str, str]]) -> None:
    labels = [row["intervened_prior"].replace("pymatching_", "").replace("_", "\n") for row in rows]
    values = [float(row["paired_delta_lfr"]) for row in rows]
    lower = [float(row["paired_delta_lfr"]) - float(row["paired_delta_lfr_ci_lower"]) for row in rows]
    upper = [float(row["paired_delta_lfr_ci_upper"]) - float(row["paired_delta_lfr"]) for row in rows]
    ax.bar(labels, values, color="#855c9c")
    ax.errorbar(range(len(values)), values, yerr=[lower, upper], fmt="none", color="#222222", linewidth=0.8)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_title("Prior intervention effects")
    ax.set_ylabel("paired delta LFR")
    ax.tick_params(axis="x", labelsize=6)


def cycle_mean_deltas(rows: list[dict[str, str]]) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[parse_int(row["cycles"])].append(float(row["paired_delta_lfr"]))
    return {cycle: statistics.fmean(values) for cycle, values in grouped.items()}


def cycle_rankings_by_group(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = f"{row['control_mode']}|{row['basis']}"
        grouped[key][parse_int(row["cycles"])].append(abs(float(row["paired_delta_lfr"])))
    out = {}
    for key, by_cycle in grouped.items():
        scores = {str(cycle): statistics.fmean(values) for cycle, values in by_cycle.items()}
        out[key] = rank_desc(scores)
    return out


def rank_desc(scores: dict[str, float]) -> list[str]:
    return [item for item, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


def top_k_overlap(left: list[str], right: list[str], *, top_k: int) -> float:
    if not left or not right:
        return 0.0
    return len(set(left[:top_k]) & set(right[:top_k])) / top_k


def pairwise_rank_distance(left: list[str], right: list[str]) -> float:
    items = list(dict.fromkeys([*left, *right]))
    if not items:
        return 0.0
    max_rank = len(items)
    left_rank = {item: index for index, item in enumerate(left)}
    right_rank = {item: index for index, item in enumerate(right)}
    return sum(abs(left_rank.get(item, max_rank) - right_rank.get(item, max_rank)) for item in items) / len(items)


def spearman_from_rankings(reference: list[str], candidate: list[str]) -> float:
    items = [item for item in reference if item in candidate]
    if len(items) < 2:
        return 0.0
    ref_rank = {item: index for index, item in enumerate(reference)}
    cand_rank = {item: index for index, item in enumerate(candidate)}
    n = len(items)
    squared = sum((ref_rank[item] - cand_rank[item]) ** 2 for item in items)
    return 1.0 - (6.0 * squared) / (n * (n * n - 1.0))


def linear_slope_corr(xs: list[int], ys: list[float]) -> tuple[float, float]:
    if len(xs) < 2:
        return 0.0, 0.0
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_den = sum((x - x_mean) ** 2 for x in xs)
    y_den = sum((y - y_mean) ** 2 for y in ys)
    slope = numerator / x_den if x_den else 0.0
    corr = numerator / math.sqrt(x_den * y_den) if x_den and y_den else 0.0
    return slope, corr


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(quantile * (len(ordered) - 1))))
    return ordered[index]


def parse_deadlines(value: str) -> list[float]:
    deadlines = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not deadlines:
        raise ValueError("--deadlines-us must contain at least one value")
    return deadlines


if __name__ == "__main__":
    main()
