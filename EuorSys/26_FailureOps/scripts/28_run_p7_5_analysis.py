#!/usr/bin/env python
"""Run P7.5 EuroSys-facing analyses over real Google RL QEC records."""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from failureops.data_model import (
    P7_5_FEATURE_FIELDS,
    P7_5_PRIOR_EFFECT_FIELDS,
    P7_5_VARIANCE_FIELDS,
)
from failureops.google_rl_qec_adapter import discover_google_rl_qec_data_dirs
from failureops.io_utils import ensure_parent_dir, read_csv_rows, write_csv_rows
from failureops.p7_5_analysis import (
    run_decoder_prior_interventions,
    summarize_paired_vs_unpaired_variance,
    summarize_rescue_induction_features,
)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/failureops-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default="data/raw/google_rl_qec/google_reinforcement_learning_qec",
    )
    parser.add_argument(
        "--effect-matrix",
        default="data/results/p7_google_rl_qec_decoder_effect_matrix.csv",
    )
    parser.add_argument(
        "--baseline-decoder",
        default="correlated_matching_decoder_with_si1000_prior",
    )
    parser.add_argument(
        "--intervened-decoder",
        default="tesseract_decoder_with_si1000_prior",
    )
    parser.add_argument(
        "--prior-data-dir",
        default=(
            "data/raw/google_rl_qec/google_reinforcement_learning_qec/"
            "surface_code_traditional_calibration/Z/r010"
        ),
    )
    parser.add_argument("--max-shots", type=int, default=10000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-bootstrap", type=int, default=300)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument(
        "--feature-output",
        default="data/results/p7_5_rescue_induction_features.csv",
    )
    parser.add_argument(
        "--variance-output",
        default="data/results/p7_5_paired_vs_unpaired_variance.csv",
    )
    parser.add_argument(
        "--prior-output",
        default="data/results/p7_5_decoder_prior_interventions.csv",
    )
    parser.add_argument(
        "--effect-plot",
        default="figures/p7_5_decoder_effect_by_cycles.png",
    )
    parser.add_argument(
        "--feature-plot",
        default="figures/p7_5_rescue_induction_features.png",
    )
    parser.add_argument(
        "--variance-plot",
        default="figures/p7_5_paired_vs_unpaired_variance.png",
    )
    parser.add_argument(
        "--prior-plot",
        default="figures/p7_5_decoder_prior_interventions.png",
    )
    args = parser.parse_args()

    data_dirs = discover_google_rl_qec_data_dirs(args.dataset_root)
    if args.limit:
        data_dirs = data_dirs[: args.limit]
    if not data_dirs:
        raise ValueError(f"no paired-decoder Google RL QEC directories under {args.dataset_root}")

    feature_rows = []
    variance_rows = []
    for index, data_dir in enumerate(data_dirs):
        run_id = f"p7_5_{index:03d}"
        feature_rows.extend(
            summarize_rescue_induction_features(
                data_dir,
                baseline_decoder_pathway=args.baseline_decoder,
                intervened_decoder_pathway=args.intervened_decoder,
                max_shots=args.max_shots,
                run_id=run_id,
            )
        )
        variance_rows.append(
            summarize_paired_vs_unpaired_variance(
                data_dir,
                baseline_decoder_pathway=args.baseline_decoder,
                intervened_decoder_pathway=args.intervened_decoder,
                max_shots=args.max_shots,
                run_id=run_id,
                num_bootstrap=args.num_bootstrap,
                bootstrap_seed=args.bootstrap_seed + index,
            )
        )
        print(f"[{index + 1}/{len(data_dirs)}] analyzed {data_dir}")

    prior_rows = run_decoder_prior_interventions(
        args.prior_data_dir,
        max_shots=args.max_shots,
        run_id="p7_5_decoder_prior",
        num_bootstrap=args.num_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )

    write_csv_rows(args.feature_output, feature_rows, P7_5_FEATURE_FIELDS)
    write_csv_rows(args.variance_output, variance_rows, P7_5_VARIANCE_FIELDS)
    write_csv_rows(args.prior_output, prior_rows, P7_5_PRIOR_EFFECT_FIELDS)

    effect_rows = read_csv_rows(args.effect_matrix)
    plot_effect_by_cycles(effect_rows, args.effect_plot)
    plot_rescue_induction_features(feature_rows, args.feature_plot)
    plot_paired_vs_unpaired_variance(variance_rows, args.variance_plot)
    plot_decoder_prior_interventions(prior_rows, args.prior_plot)
    print(f"wrote {len(feature_rows)} feature rows to {args.feature_output}")
    print(f"wrote {len(variance_rows)} variance rows to {args.variance_output}")
    print(f"wrote {len(prior_rows)} decoder-prior rows to {args.prior_output}")
    print(
        "wrote P7.5 figures to "
        f"{args.effect_plot}, {args.feature_plot}, {args.variance_plot}, {args.prior_plot}"
    )


def plot_effect_by_cycles(rows: list[dict[str, str]], output: str) -> None:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["control_mode"], row["basis"])].append(row)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for (control_mode, basis), group in sorted(grouped.items()):
        group = sorted(group, key=lambda row: int(row["cycles"]))
        ax.plot(
            [int(row["cycles"]) for row in group],
            [float(row["paired_delta_lfr"]) for row in group],
            marker="o",
            linewidth=1.8,
            label=f"{control_mode} {basis}",
        )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("P7.5 decoder-pathway sensitivity grows with memory duration")
    ax.set_xlabel("QEC cycles")
    ax.set_ylabel("Paired delta LFR")
    ax.legend(fontsize=8)
    fig.tight_layout()
    ensure_parent_dir(output)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_rescue_induction_features(rows: list[dict[str, object]], output: str) -> None:
    transitions = ["rescued", "induced", "unchanged_failure"]
    means = []
    bursts = []
    for transition in transitions:
        group = [row for row in rows if row["transition_class"] == transition and int(row["num_shots"]) > 0]
        means.append(weighted_mean(group, "mean_detector_event_count"))
        bursts.append(weighted_mean(group, "burst_fraction"))
    x = list(range(len(transitions)))
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar([item - 0.18 for item in x], means, width=0.36, color="#3f7fb5", label="detector count")
    ax1.set_ylabel("Mean detector-event count")
    ax1.set_xticks(x, transitions)
    ax2 = ax1.twinx()
    ax2.bar([item + 0.18 for item in x], bursts, width=0.36, color="#b65f35", label="burst fraction")
    ax2.set_ylabel("Burst fraction")
    ax1.set_title("P7.5 rescued and induced shots have measurable syndrome profiles")
    fig.tight_layout()
    ensure_parent_dir(output)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_paired_vs_unpaired_variance(rows: list[dict[str, object]], output: str) -> None:
    by_cycles: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_cycles[int(row["cycles"])].append(row)
    cycles = sorted(by_cycles)
    paired = [average(float(row["paired_bootstrap_std"]) for row in by_cycles[cycle]) for cycle in cycles]
    unpaired = [average(float(row["unpaired_bootstrap_std"]) for row in by_cycles[cycle]) for cycle in cycles]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(cycles, paired, marker="o", linewidth=1.8, label="paired bootstrap")
    ax.plot(cycles, unpaired, marker="o", linewidth=1.8, label="unpaired bootstrap")
    ax.set_title("P7.5 paired counterfactuals reduce estimate variance")
    ax.set_xlabel("QEC cycles")
    ax.set_ylabel("Bootstrap std of delta LFR")
    ax.legend()
    fig.tight_layout()
    ensure_parent_dir(output)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_decoder_prior_interventions(rows: list[dict[str, object]], output: str) -> None:
    labels = [str(row["intervened_prior"]).replace("pymatching_", "") for row in rows]
    values = [float(row["paired_delta_lfr"]) for row in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, values, color="#5f7f4f")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("P7.5 decoder-prior interventions on real detector records")
    ax.set_ylabel("Paired delta LFR")
    ax.tick_params(axis="x", labelrotation=25)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    ensure_parent_dir(output)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def weighted_mean(rows: list[dict[str, object]], field: str) -> float:
    total_weight = sum(int(row["num_shots"]) for row in rows)
    if not total_weight:
        return 0.0
    return sum(float(row[field]) * int(row["num_shots"]) for row in rows) / total_weight


def average(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
