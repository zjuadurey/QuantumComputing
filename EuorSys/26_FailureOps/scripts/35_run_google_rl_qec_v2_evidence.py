#!/usr/bin/env python
"""Run P7/P7.5-style evidence generation over the Google RL QEC v2 dataset."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from failureops.data_model import P7_5_FEATURE_FIELDS, P7_5_VARIANCE_FIELDS, P7_SWEEP_AGGREGATE_FIELDS, P7_SWEEP_SUMMARY_FIELDS
from failureops.google_rl_qec_adapter import (
    control_mode_from_experiment,
    discover_google_rl_qec_data_dirs,
    load_google_rl_qec_records,
    summarize_p7_sweep_groups,
)
from failureops.io_utils import ensure_parent_dir, fmt_float, parse_int, write_csv_rows
from failureops.manifest import write_manifest
from failureops.p7_5_analysis import (
    bootstrap_delta_values,
    feature_row,
    sample_std,
    transition_class,
    percentile,
    mean,
)
from failureops.paired_metrics import summarize_paired_effects

os.environ.setdefault("MPLCONFIGDIR", "/tmp/failureops-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="data/raw/google_rl_qec_v2")
    parser.add_argument("--baseline-decoder", default="tesseract_decoder_with_si1000_prior")
    parser.add_argument(
        "--intervened-decoder",
        default="tesseract_decoder_with_frequency_calibrated_prior",
    )
    parser.add_argument("--max-shots", type=int, default=10000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-bootstrap", type=int, default=100)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--run-id-prefix", default="p10_google_rl_qec_v2")
    parser.add_argument(
        "--summary-output",
        default="data/results/p10_google_rl_qec_v2_decoder_effect_matrix.csv",
    )
    parser.add_argument(
        "--aggregate-output",
        default="data/results/p10_google_rl_qec_v2_decoder_effect_aggregate.csv",
    )
    parser.add_argument(
        "--feature-output",
        default="data/results/p10_google_rl_qec_v2_rescue_induction_features.csv",
    )
    parser.add_argument(
        "--variance-output",
        default="data/results/p10_google_rl_qec_v2_paired_vs_unpaired_variance.csv",
    )
    parser.add_argument(
        "--manifest-output",
        default="data/results/p10_google_rl_qec_v2_manifest.json",
    )
    parser.add_argument(
        "--figure-output",
        default="figures/p10_google_rl_qec_v2_evidence.png",
    )
    args = parser.parse_args()

    data_dirs = discover_google_rl_qec_data_dirs(
        args.dataset_root,
        baseline_decoder_pathway=args.baseline_decoder,
        intervened_decoder_pathway=args.intervened_decoder,
    )
    if args.limit:
        data_dirs = data_dirs[: args.limit]
    if not data_dirs:
        raise ValueError(f"no Google RL QEC v2 experiment directories found under {args.dataset_root}")

    summary_rows = []
    feature_rows = []
    variance_rows = []
    for index, data_dir in enumerate(data_dirs):
        run_id = f"{args.run_id_prefix}_{index:03d}"
        baseline_rows, intervention_rows = load_google_rl_qec_records(
            data_dir,
            baseline_decoder_pathway=args.baseline_decoder,
            intervened_decoder_pathway=args.intervened_decoder,
            max_shots=args.max_shots,
            run_id=run_id,
        )
        summary_rows.append(
            summarize_condition_from_rows(
                baseline_rows,
                intervention_rows,
                data_dir=data_dir,
                baseline_decoder_pathway=args.baseline_decoder,
                intervened_decoder_pathway=args.intervened_decoder,
                num_bootstrap=args.num_bootstrap,
                bootstrap_seed=args.bootstrap_seed,
            )
        )
        feature_rows.extend(summarize_features_from_rows(baseline_rows, intervention_rows, data_dir=data_dir))
        variance_rows.append(
            summarize_variance_from_rows(
                baseline_rows,
                intervention_rows,
                data_dir=data_dir,
                num_bootstrap=args.num_bootstrap,
                bootstrap_seed=args.bootstrap_seed + index,
            )
        )
        row = summary_rows[-1]
        print(
            f"[{index + 1}/{len(data_dirs)}] {row['workload_id']} "
            f"delta={row['paired_delta_lfr']} valid={row['valid_pairs']}"
        )

    aggregate_rows = summarize_p7_sweep_groups(summary_rows)
    write_csv_rows(args.summary_output, summary_rows, P7_SWEEP_SUMMARY_FIELDS)
    write_csv_rows(args.aggregate_output, aggregate_rows, P7_SWEEP_AGGREGATE_FIELDS)
    write_csv_rows(args.feature_output, feature_rows, P7_5_FEATURE_FIELDS)
    write_csv_rows(args.variance_output, variance_rows, P7_5_VARIANCE_FIELDS)
    plot_google_rl_qec_v2(summary_rows, variance_rows, feature_rows, args.figure_output)
    write_manifest(
        args.manifest_output,
        config={
            "experiment_id": "p10_google_rl_qec_v2",
            "dataset_root": args.dataset_root,
            "baseline_decoder": args.baseline_decoder,
            "intervened_decoder": args.intervened_decoder,
            "max_shots": args.max_shots,
            "limit": args.limit,
            "num_bootstrap": args.num_bootstrap,
            "bootstrap_seed": args.bootstrap_seed,
        },
        command=sys.argv,
        outputs={
            "summary": args.summary_output,
            "aggregate": args.aggregate_output,
            "features": args.feature_output,
            "variance": args.variance_output,
            "figure": args.figure_output,
        },
        row_counts={
            "summary": len(summary_rows),
            "aggregate": len(aggregate_rows),
            "features": len(feature_rows),
            "variance": len(variance_rows),
        },
    )
    print(f"wrote {len(summary_rows)} Google RL QEC v2 summary rows to {args.summary_output}")
    print(f"wrote {len(aggregate_rows)} Google RL QEC v2 aggregate rows to {args.aggregate_output}")
    print(f"wrote {len(feature_rows)} Google RL QEC v2 feature rows to {args.feature_output}")
    print(f"wrote {len(variance_rows)} Google RL QEC v2 variance rows to {args.variance_output}")
    print(f"wrote Google RL QEC v2 evidence figure to {args.figure_output}")
    print(f"wrote Google RL QEC v2 manifest to {args.manifest_output}")


def summarize_condition_from_rows(
    baseline_rows: list[dict[str, object]],
    intervention_rows: list[dict[str, object]],
    *,
    data_dir: str | Path,
    baseline_decoder_pathway: str,
    intervened_decoder_pathway: str,
    num_bootstrap: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    effects = summarize_paired_effects(
        intervention_rows,
        num_bootstrap=num_bootstrap,
        bootstrap_seed=bootstrap_seed,
    )
    if len(effects) != 1:
        raise ValueError(f"expected exactly one paired-effect row for {data_dir}, got {len(effects)}")
    effect = effects[0]
    first = baseline_rows[0]
    valid_pairs = parse_int(effect["valid_pairs"])
    baseline_failures = parse_int(effect["baseline_failure_count"])
    intervened_failures = parse_int(effect["intervened_failure_count"])
    invalid_pairs = parse_int(effect["invalid_pairs"])
    return {
        "experiment_name": first["experiment_name"],
        "code_family": first["code_family"],
        "control_mode": control_mode_from_experiment(str(first["experiment_name"])),
        "basis": first["basis"],
        "cycle_dir": first["cycle_dir"],
        "cycles": first["cycles"],
        "workload_id": first["workload_id"],
        "baseline_decoder_pathway": baseline_decoder_pathway,
        "intervened_decoder_pathway": intervened_decoder_pathway,
        "num_pairs": effect["num_pairs"],
        "valid_pairs": valid_pairs,
        "invalid_pairs": invalid_pairs,
        "baseline_failure_count": baseline_failures,
        "intervened_failure_count": intervened_failures,
        "rescued_failure_count": effect["rescued_failure_count"],
        "induced_failure_count": effect["induced_failure_count"],
        "unchanged_failure_count": effect["unchanged_failure_count"],
        "unchanged_success_count": effect["unchanged_success_count"],
        "net_rescue_count": effect["net_rescue_count"],
        "baseline_lfr": fmt_float(baseline_failures / valid_pairs if valid_pairs else 0.0),
        "intervened_lfr": fmt_float(intervened_failures / valid_pairs if valid_pairs else 0.0),
        "paired_delta_lfr": effect["paired_delta_lfr"],
        "net_rescue_rate": effect["net_rescue_rate"],
        "rescue_rate_among_baseline_failures": effect["rescue_rate_among_baseline_failures"],
        "induction_rate_among_baseline_successes": effect["induction_rate_among_baseline_successes"],
        "pairing_violation_rate": fmt_float(invalid_pairs / parse_int(effect["num_pairs"]) if parse_int(effect["num_pairs"]) else 0.0),
        "source_data_dir": str(data_dir),
    }


def summarize_features_from_rows(
    baseline_rows: list[dict[str, object]],
    intervention_rows: list[dict[str, object]],
    *,
    data_dir: str | Path,
) -> list[dict[str, object]]:
    baseline_by_pair = {str(row["pair_id"]): baseline for row, baseline in zip(intervention_rows, baseline_rows)}
    groups: dict[str, list[dict[str, object]]] = {
        "rescued": [],
        "induced": [],
        "unchanged_failure": [],
        "unchanged_success": [],
    }
    for row in intervention_rows:
        groups[transition_class(row)].append(baseline_by_pair[str(row["pair_id"])])
    first = baseline_rows[0]
    return [feature_row(name, groups[name], first, str(data_dir)) for name in groups]


def summarize_variance_from_rows(
    baseline_rows: list[dict[str, object]],
    intervention_rows: list[dict[str, object]],
    *,
    data_dir: str | Path,
    num_bootstrap: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    first = baseline_rows[0]
    paired_values, unpaired_values = bootstrap_delta_values(
        intervention_rows,
        num_bootstrap=num_bootstrap,
        seed=bootstrap_seed,
    )
    paired_delta = mean(
        int(row["intervened_logical_failure"]) - int(row["baseline_logical_failure"])
        for row in intervention_rows
    )
    paired_std = sample_std(paired_values)
    unpaired_std = sample_std(unpaired_values)
    return {
        "experiment_name": first["experiment_name"],
        "code_family": first["code_family"],
        "control_mode": control_mode_from_experiment(str(first["experiment_name"])),
        "basis": first["basis"],
        "cycles": first["cycles"],
        "workload_id": first["workload_id"],
        "num_pairs": len(intervention_rows),
        "paired_delta_lfr": fmt_float(paired_delta),
        "paired_bootstrap_std": fmt_float(paired_std),
        "unpaired_bootstrap_std": fmt_float(unpaired_std),
        "std_ratio_unpaired_over_paired": fmt_float(unpaired_std / paired_std if paired_std else 0.0),
        "paired_ci_lower": fmt_float(percentile(paired_values, 0.025)),
        "paired_ci_upper": fmt_float(percentile(paired_values, 0.975)),
        "unpaired_ci_lower": fmt_float(percentile(unpaired_values, 0.025)),
        "unpaired_ci_upper": fmt_float(percentile(unpaired_values, 0.975)),
        "num_bootstrap": num_bootstrap,
        "source_data_dir": str(data_dir),
    }


def plot_google_rl_qec_v2(
    summary_rows: list[dict[str, object]],
    variance_rows: list[dict[str, object]],
    feature_rows: list[dict[str, object]],
    output: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    plot_cycle_delta_panel(axes[0][0], summary_rows)
    plot_cycle_variance_panel(axes[0][1], variance_rows)
    plot_experiment_delta_panel(axes[1][0], summary_rows)
    plot_transition_feature_panel(axes[1][1], feature_rows)
    fig.tight_layout()
    ensure_parent_dir(output)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_cycle_delta_panel(ax, rows: list[dict[str, object]]) -> None:
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[group_label(row)][int(row["cycles"])].append(float(row["paired_delta_lfr"]))
    for label, cycle_map in sorted(grouped.items()):
        cycles = sorted(cycle_map)
        ax.plot(
            cycles,
            [statistics.fmean(cycle_map[cycle]) for cycle in cycles],
            marker="o",
            linewidth=1.6,
            label=label,
        )
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_title("Paired Delta LFR By Cycle")
    ax.set_xlabel("QEC cycles")
    ax.set_ylabel("Mean paired delta LFR")
    ax.legend(fontsize=7, ncol=2)


def plot_cycle_variance_panel(ax, rows: list[dict[str, object]]) -> None:
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[group_label(row)][int(row["cycles"])].append(float(row["std_ratio_unpaired_over_paired"]))
    for label, cycle_map in sorted(grouped.items()):
        cycles = sorted(cycle_map)
        ax.plot(
            cycles,
            [statistics.fmean(cycle_map[cycle]) for cycle in cycles],
            marker="o",
            linewidth=1.6,
            label=label,
        )
    ax.axhline(1.0, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_title("Pairing Advantage By Cycle")
    ax.set_xlabel("QEC cycles")
    ax.set_ylabel("Mean std ratio (unpaired / paired)")
    ax.legend(fontsize=7, ncol=2)


def plot_experiment_delta_panel(ax, rows: list[dict[str, object]]) -> None:
    experiment_groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        experiment_groups[str(row["experiment_name"])].append(float(row["paired_delta_lfr"]))
    labels = sorted(experiment_groups)
    values = [statistics.fmean(experiment_groups[label]) for label in labels]
    positions = list(range(len(labels)))
    ax.barh(positions, values, color="#3f7fb5")
    ax.axvline(0.0, color="#333333", linewidth=0.8)
    ax.set_yticks(positions, [short_experiment_name(label) for label in labels])
    ax.set_title("Mean Effect By Experiment Family")
    ax.set_xlabel("Mean paired delta LFR")


def plot_transition_feature_panel(ax, rows: list[dict[str, object]]) -> None:
    transitions = ("rescued", "induced", "unchanged_failure", "unchanged_success")
    counts = [weighted_mean(rows, transition, "mean_detector_event_count") for transition in transitions]
    bursts = [weighted_mean(rows, transition, "burst_fraction") for transition in transitions]
    x = list(range(len(transitions)))
    ax.bar([item - 0.18 for item in x], counts, width=0.36, color="#3f7fb5", label="detector count")
    twin = ax.twinx()
    twin.bar([item + 0.18 for item in x], bursts, width=0.36, color="#b65f35", label="burst fraction")
    ax.set_xticks(x, [transition.replace("_", "\n") for transition in transitions])
    ax.set_ylabel("Weighted mean detector-event count")
    twin.set_ylabel("Weighted mean burst fraction")
    ax.set_title("Transition Profiles")
    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    ax.legend(handles, labels, fontsize=8, loc="upper right")


def weighted_mean(rows: list[dict[str, object]], transition_class: str, field: str) -> float:
    group = [row for row in rows if str(row["transition_class"]) == transition_class]
    total_weight = sum(int(row["num_shots"]) for row in group)
    if not total_weight:
        return 0.0
    return sum(float(row[field]) * int(row["num_shots"]) for row in group) / total_weight


def group_label(row: dict[str, object]) -> str:
    return " | ".join(
        (
            short_code_family(str(row["code_family"])),
            short_control_mode(str(row["control_mode"])),
            str(row["basis"]),
        )
    )


def short_code_family(code_family: str) -> str:
    if code_family == "surface_code_memory":
        return "surface"
    if code_family == "color_code_memory":
        return "color"
    return code_family


def short_control_mode(control_mode: str) -> str:
    if control_mode == "traditional_calibration":
        return "trad"
    if control_mode == "traditional_calibration_and_rl_fine_tuning":
        return "trad+rl"
    return control_mode


def short_experiment_name(name: str) -> str:
    name = name.replace("surface_code_distance_3_5_7_", "surface ")
    name = name.replace("color_code_distance_5_", "color ")
    name = name.replace("traditional_calibration_and_rl_fine_tuning", "trad+rl")
    name = name.replace("traditional_calibration", "trad")
    return name


if __name__ == "__main__":
    main()
