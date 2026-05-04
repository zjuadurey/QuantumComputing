#!/usr/bin/env python
"""Plot P2.5 repeat, stress, and pattern-shift summaries."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from failureops.io_utils import ensure_parent_dir, read_csv_rows

os.environ.setdefault("MPLCONFIGDIR", "/tmp/failureops-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STRESS_ORDER = ["low", "medium", "high"]
RUNTIME_FOCUS = [
    "remove_decoder_timeout",
    "relax_timeout_policy",
    "reduce_decoder_delay_50pct",
    "reduce_idle_exposure_50pct",
    "increase_decoder_capacity_2x",
    "eliminate_decoder_backlog",
    "increase_synchronization_slack",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeated-summary", default="data/results/p2_5_repeated_summary.csv")
    parser.add_argument("--stress-summary", default="data/results/p2_5_stress_sweep_summary.csv")
    parser.add_argument("--pattern-summary", default="data/results/p2_5_pattern_shift_summary.csv")
    parser.add_argument("--ci-plot", default="figures/p2_5_intervention_delta_lfr_with_ci.png")
    parser.add_argument("--stress-plot", default="figures/p2_5_runtime_stress_sweep.png")
    parser.add_argument("--pattern-plot", default="figures/p2_5_failure_pattern_shift.png")
    args = parser.parse_args()

    plot_repeated_ci(read_csv_rows(args.repeated_summary), args.ci_plot)
    plot_stress_sweep(read_csv_rows(args.stress_summary), args.stress_plot)
    plot_pattern_shift(read_csv_rows(args.pattern_summary), args.pattern_plot)
    print(f"wrote P2.5 plots to {args.ci_plot}, {args.stress_plot}, {args.pattern_plot}")


def plot_repeated_ci(rows: list[dict[str, str]], output: str) -> None:
    rows = sorted(rows, key=lambda row: float(row["mean_absolute_delta_lfr"]))
    labels = [row["intervention"] for row in rows]
    deltas = [float(row["mean_absolute_delta_lfr"]) for row in rows]
    cis = [float(row["ci95_absolute_delta_lfr"]) for row in rows]
    colors = ["#2f7f7f" if value <= 0 else "#b65f35" for value in deltas]

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(labels, deltas, yerr=cis, capsize=4, color=colors)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_title("P2.5 intervention delta LFR across seeds")
    ax.set_ylabel("Mean absolute delta LFR")
    ax.set_xlabel("Intervention")
    ax.tick_params(axis="x", labelrotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    ensure_parent_dir(output)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_stress_sweep(rows: list[dict[str, str]], output: str) -> None:
    by_intervention = {
        intervention: {
            row["stress_level"]: float(row["mean_absolute_delta_lfr"])
            for row in rows
            if row["intervention"] == intervention
        }
        for intervention in RUNTIME_FOCUS
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    x = list(range(len(STRESS_ORDER)))
    for intervention, values in by_intervention.items():
        if not values:
            continue
        ax.plot(
            x,
            [values.get(level, 0.0) for level in STRESS_ORDER],
            marker="o",
            linewidth=1.8,
            label=intervention,
        )
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xticks(x, STRESS_ORDER)
    ax.set_title("P2.5 runtime intervention sensitivity under stress")
    ax.set_ylabel("Mean absolute delta LFR")
    ax.set_xlabel("Runtime stress level")
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    ensure_parent_dir(output)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_pattern_shift(rows: list[dict[str, str]], output: str) -> None:
    rows = sorted(
        rows,
        key=lambda row: int(row["intervened_failure_count"]) - int(row["baseline_failure_count"]),
    )
    labels = [row["intervention"] for row in rows]
    timeout_deltas = [int(row["timeout_correlated_delta_count"]) for row in rows]
    idle_deltas = [int(row["idle_correlated_delta_count"]) for row in rows]
    syndrome_deltas = [int(row["syndrome_burst_delta_count"]) for row in rows]

    y = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh([value - 0.22 for value in y], timeout_deltas, height=0.22, label="timeout_correlated")
    ax.barh(y, idle_deltas, height=0.22, label="idle_correlated")
    ax.barh([value + 0.22 for value in y], syndrome_deltas, height=0.22, label="syndrome_burst")
    ax.axvline(0.0, color="#333333", linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.set_title("P2.5 failure-pattern count shifts")
    ax.set_xlabel("Intervened count - baseline count")
    ax.legend(fontsize=8)
    fig.tight_layout()
    ensure_parent_dir(output)
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
