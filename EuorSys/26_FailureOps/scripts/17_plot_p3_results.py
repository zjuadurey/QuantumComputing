#!/usr/bin/env python
"""Plot P3 baseline comparison, rank stability, and runtime scaling results."""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from failureops.io_utils import ensure_parent_dir, read_csv_rows

os.environ.setdefault("MPLCONFIGDIR", "/tmp/failureops-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STRESS_ORDER = ["low", "medium", "high"]
RUNTIME_FOCUS = [
    "remove_decoder_timeout",
    "increase_decoder_workers_2x",
    "increase_decoder_service_rate_2x",
    "remove_decoder_queueing",
    "relax_decoder_deadline_2x",
    "reduce_idle_exposure_50pct",
    "prioritize_high_weight_syndromes",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-summary", default="data/results/p3_eval_matrix_summary.csv")
    parser.add_argument("--baseline-comparison", default="data/results/p3_baseline_comparison.csv")
    parser.add_argument("--rank-stability", default="data/results/p3_rank_stability.csv")
    parser.add_argument("--baseline-plot", default="figures/p3_failureops_vs_baselines.png")
    parser.add_argument("--rank-plot", default="figures/p3_rank_stability.png")
    parser.add_argument("--runtime-plot", default="figures/p3_runtime_intervention_scaling.png")
    parser.add_argument("--paired-plot", default="figures/p3_paired_vs_unpaired_variance.png")
    args = parser.parse_args()

    eval_rows = read_csv_rows(args.eval_summary)
    comparison_rows = read_csv_rows(args.baseline_comparison)
    rank_rows = read_csv_rows(args.rank_stability)
    plot_baseline_comparison(comparison_rows, args.baseline_plot)
    plot_rank_stability(rank_rows, args.rank_plot)
    plot_runtime_scaling(eval_rows, args.runtime_plot)
    plot_paired_proxy_variance(comparison_rows, args.paired_plot)
    print(
        "wrote P3 plots to "
        f"{args.baseline_plot}, {args.rank_plot}, {args.runtime_plot}, {args.paired_plot}"
    )


def plot_baseline_comparison(rows: list[dict[str, str]], output: str) -> None:
    methods = [
        "paired_counterfactual",
        "noise_only_attribution",
        "runtime_only_attribution",
        "plain_lfr_reporting",
        "static_label_attribution",
    ]
    values = {
        method: -average(
            abs(float(row["top_intervention_delta_lfr"]))
            for row in rows
            if row["method"] == method
        )
        for method in methods
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(methods, [values[method] for method in methods], color="#2f7f7f")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("P3 FailureOps vs baseline reporting methods")
    ax.set_ylabel("Mean strongest delta LFR")
    ax.tick_params(axis="x", labelrotation=25)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    ensure_parent_dir(output)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_rank_stability(rows: list[dict[str, str]], output: str) -> None:
    labels = [f"{row['workload_id']}\n{row['stress_level']}" for row in rows]
    overlaps = [float(row["mean_top3_overlap"]) for row in rows]
    distances = [float(row["mean_pairwise_rank_distance"]) for row in rows]
    x = list(range(len(rows)))
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.bar(x, overlaps, color="#3f7fb5", label="top-3 overlap")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Mean top-3 overlap")
    ax1.set_xticks(x, labels)
    ax1.tick_params(axis="x", labelrotation=35)
    ax2 = ax1.twinx()
    ax2.plot(x, distances, color="#b65f35", marker="o", label="rank distance")
    ax2.set_ylabel("Mean pairwise rank distance")
    ax1.set_title("P3 intervention rank stability")
    fig.tight_layout()
    ensure_parent_dir(output)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_runtime_scaling(rows: list[dict[str, str]], output: str) -> None:
    by_intervention: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["intervention"] not in RUNTIME_FOCUS:
            continue
        by_intervention[row["intervention"]][row["stress_level"]].append(float(row["mean_absolute_delta_lfr"]))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = list(range(len(STRESS_ORDER)))
    for intervention in RUNTIME_FOCUS:
        values = by_intervention.get(intervention, {})
        if not values:
            continue
        ax.plot(
            x,
            [average(values[level]) for level in STRESS_ORDER],
            marker="o",
            linewidth=1.8,
            label=intervention,
        )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x, STRESS_ORDER)
    ax.set_title("P3 runtime intervention scaling under stress")
    ax.set_ylabel("Mean absolute delta LFR")
    ax.set_xlabel("Runtime stress level")
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    ensure_parent_dir(output)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_paired_proxy_variance(rows: list[dict[str, str]], output: str) -> None:
    selected = [
        row for row in rows
        if row["method"] in {"paired_counterfactual", "noise_only_attribution", "runtime_only_attribution"}
    ]
    labels = [f"{row['method']}\n{row['workload_id']}:{row['stress_level']}" for row in selected]
    values = [float(row["std_absolute_delta_lfr"]) for row in selected]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(labels, values, color="#6d6a75")
    ax.set_title("P3 attribution spread by method")
    ax.set_ylabel("Std of intervention delta LFR")
    ax.tick_params(axis="x", labelrotation=55)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    ensure_parent_dir(output)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def average(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()

