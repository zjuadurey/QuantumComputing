from __future__ import annotations

import math
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = PROJECT_ROOT / ".cache"
MPLCONFIGDIR = CACHE_ROOT / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DECODER_ORDER = ("BP-OSD", "BP-LSD")


def load_summary_df(results_dir: Path) -> pd.DataFrame:
    repeated_summary_path = results_dir / "results_repeated_summary.csv"
    results_path = results_dir / "results.csv"
    input_path = repeated_summary_path if repeated_summary_path.exists() else results_path
    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError(f"No rows found in {input_path}")
    return df


def make_grid(num_panels: int, *, scale: float = 1.0):
    ncols = min(2, num_panels)
    nrows = math.ceil(num_panels / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.5 * ncols * scale, 4.3 * nrows * scale),
        squeeze=False,
    )
    return fig, axes.ravel()


def plot_ler_with_ci(summary_df: pd.DataFrame, output_path: Path) -> None:
    code_order = list(dict.fromkeys(summary_df["code_name"]))
    fig, axes = make_grid(len(code_order))

    for ax, code_name in zip(axes, code_order):
        code_df = summary_df[summary_df["code_name"] == code_name]
        n = int(code_df["n"].iloc[0])
        k = int(code_df["k"].iloc[0])
        d_value = code_df["d"].iloc[0]
        d_text = "?" if pd.isna(d_value) else str(int(d_value))

        for decoder_name in DECODER_ORDER:
            subdf = code_df[code_df["decoder_name"] == decoder_name].sort_values("p")
            if subdf.empty:
                continue

            p_values = subdf["p"].to_numpy(dtype=float)
            num_shots = subdf["num_shots"].to_numpy(dtype=float)
            y = subdf["ler_mean"].to_numpy(dtype=float) if "ler_mean" in subdf else subdf["logical_error_rate"].to_numpy(dtype=float)
            y_plot = np.maximum(y, 0.5 / num_shots)
            ci_low = subdf["ler_ci_low"].to_numpy(dtype=float)
            ci_high = subdf["ler_ci_high"].to_numpy(dtype=float)
            lower_for_plot = np.minimum(y_plot, np.maximum(ci_low, 0.0))
            yerr = np.vstack(
                [
                    np.maximum(y_plot - lower_for_plot, 0.0),
                    np.maximum(ci_high - y_plot, 0.0),
                ]
            )

            ax.errorbar(
                p_values,
                y_plot,
                yerr=yerr,
                marker="o",
                linewidth=2,
                capsize=3,
                label=decoder_name,
            )

        ax.set_title(f"{code_name} (n={n}, k={k}, d={d_text})")
        ax.set_xlabel("Physical Z error rate p")
        ax.set_ylabel("Logical error rate")
        ax.set_yscale("log")
        ax.legend()

    for ax in axes[len(code_order) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_latency_by_code_with_errorbars(
    summary_df: pd.DataFrame,
    *,
    metric_mean_col: str,
    metric_std_col: str,
    ylabel: str,
    title_prefix: str,
    output_path: Path,
) -> None:
    code_order = list(dict.fromkeys(summary_df["code_name"]))
    fig, axes = make_grid(len(code_order))

    for ax, code_name in zip(axes, code_order):
        code_df = summary_df[summary_df["code_name"] == code_name]
        n = int(code_df["n"].iloc[0])
        k = int(code_df["k"].iloc[0])

        for decoder_name in DECODER_ORDER:
            subdf = code_df[code_df["decoder_name"] == decoder_name].sort_values("p")
            if subdf.empty:
                continue

            ax.errorbar(
                subdf["p"],
                subdf[metric_mean_col],
                yerr=subdf[metric_std_col],
                marker="o",
                linewidth=2,
                capsize=3,
                label=decoder_name,
            )

        ax.set_title(f"{title_prefix}: {code_name} (n={n}, k={k})")
        ax.set_xlabel("Physical Z error rate p")
        ax.set_ylabel(ylabel)
        ax.legend()

    for ax in axes[len(code_order) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_compare_codes_ler_by_decoder(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(DECODER_ORDER), figsize=(13, 4.8), squeeze=False)
    axes = axes.ravel()
    code_order = list(dict.fromkeys(summary_df["code_name"]))

    for ax, decoder_name in zip(axes, DECODER_ORDER, strict=True):
        decoder_df = summary_df[summary_df["decoder_name"] == decoder_name]
        for code_name in code_order:
            subdf = decoder_df[decoder_df["code_name"] == code_name].sort_values("p")
            if subdf.empty:
                continue

            p_values = subdf["p"].to_numpy(dtype=float)
            num_shots = subdf["num_shots"].to_numpy(dtype=float)
            y = subdf["ler_mean"].to_numpy(dtype=float)
            y_plot = np.maximum(y, 0.5 / num_shots)
            ci_low = subdf["ler_ci_low"].to_numpy(dtype=float)
            ci_high = subdf["ler_ci_high"].to_numpy(dtype=float)
            lower_for_plot = np.minimum(y_plot, np.maximum(ci_low, 0.0))
            yerr = np.vstack(
                [
                    np.maximum(y_plot - lower_for_plot, 0.0),
                    np.maximum(ci_high - y_plot, 0.0),
                ]
            )

            ax.errorbar(
                p_values,
                y_plot,
                yerr=yerr,
                marker="o",
                linewidth=2,
                capsize=3,
                label=code_name,
            )

        ax.set_title(decoder_name)
        ax.set_xlabel("Physical Z error rate p")
        ax.set_ylabel("Logical error rate")
        ax.set_yscale("log")
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_compare_codes_p95_latency_with_errorbars(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(DECODER_ORDER), figsize=(13, 4.8), squeeze=False)
    axes = axes.ravel()
    code_order = list(dict.fromkeys(summary_df["code_name"]))

    for ax, decoder_name in zip(axes, DECODER_ORDER, strict=True):
        decoder_df = summary_df[summary_df["decoder_name"] == decoder_name]
        for code_name in code_order:
            subdf = decoder_df[decoder_df["code_name"] == code_name].sort_values("p")
            if subdf.empty:
                continue

            ax.errorbar(
                subdf["p"],
                subdf["p95_latency_mean"],
                yerr=subdf["p95_latency_std"],
                marker="o",
                linewidth=2,
                capsize=3,
                label=code_name,
            )

        ax.set_title(decoder_name)
        ax.set_xlabel("Physical Z error rate p")
        ax.set_ylabel("p95 latency per shot (ms)")
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_bb_n144_focus(summary_df: pd.DataFrame, output_path: Path) -> None:
    focus_df = summary_df[summary_df["code_name"] == "bb_n144_k12"].copy()
    if focus_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), squeeze=False)
    axes = axes.ravel()
    plot_specs = [
        ("avg_latency_mean", "avg_latency_std", "Average latency per shot (ms)", "Average latency"),
        ("p95_latency_mean", "p95_latency_std", "p95 latency per shot (ms)", "p95 latency"),
    ]

    for ax, (mean_col, std_col, ylabel, title) in zip(axes, plot_specs, strict=True):
        for decoder_name in DECODER_ORDER:
            subdf = focus_df[focus_df["decoder_name"] == decoder_name].sort_values("p")
            if subdf.empty:
                continue

            ax.errorbar(
                subdf["p"],
                subdf[mean_col],
                yerr=subdf[std_col],
                marker="o",
                linewidth=2,
                capsize=3,
                label=decoder_name,
            )

        ax.set_title(f"bb_n144_k12: {title}")
        ax.set_xlabel("Physical Z error rate p")
        ax.set_ylabel(ylabel)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    results_dir = PROJECT_ROOT / "data" / "results"
    summary_df = load_summary_df(results_dir)

    plt.style.use("seaborn-v0_8-whitegrid")

    plot_ler_with_ci(summary_df, results_dir / "ler_vs_p_with_ci.png")
    plot_ler_with_ci(summary_df, results_dir / "ler_vs_p.png")
    plot_latency_by_code_with_errorbars(
        summary_df,
        metric_mean_col="avg_latency_mean",
        metric_std_col="avg_latency_std",
        ylabel="Average latency per shot (ms)",
        title_prefix="Average latency",
        output_path=results_dir / "latency_avg_vs_p.png",
    )
    plot_latency_by_code_with_errorbars(
        summary_df,
        metric_mean_col="avg_latency_mean",
        metric_std_col="avg_latency_std",
        ylabel="Average latency per shot (ms)",
        title_prefix="Average latency",
        output_path=results_dir / "latency_vs_p.png",
    )
    plot_latency_by_code_with_errorbars(
        summary_df,
        metric_mean_col="p95_latency_mean",
        metric_std_col="p95_latency_std",
        ylabel="p95 latency per shot (ms)",
        title_prefix="p95 latency",
        output_path=results_dir / "latency_p95_vs_p.png",
    )
    plot_compare_codes_ler_by_decoder(summary_df, results_dir / "compare_codes_ler_by_decoder.png")
    plot_compare_codes_p95_latency_with_errorbars(
        summary_df,
        results_dir / "compare_codes_p95_latency_with_errorbars.png",
    )
    plot_compare_codes_p95_latency_with_errorbars(
        summary_df,
        results_dir / "compare_codes_p95_latency_by_decoder.png",
    )
    plot_bb_n144_focus(summary_df, results_dir / "bb_n144_latency_focus.png")

    print(f"Saved {results_dir / 'ler_vs_p_with_ci.png'}")
    print(f"Saved {results_dir / 'latency_avg_vs_p.png'}")
    print(f"Saved {results_dir / 'latency_p95_vs_p.png'}")
    print(f"Saved {results_dir / 'compare_codes_p95_latency_with_errorbars.png'}")
    print(f"Saved {results_dir / 'bb_n144_latency_focus.png'}")


if __name__ == "__main__":
    main()
