from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = PROJECT_ROOT / ".cache"
MPLCONFIGDIR = CACHE_ROOT / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt
import pandas as pd


def safe_std(series: pd.Series) -> float:
    if len(series) <= 1:
        return 0.0
    return float(series.std(ddof=1))


def build_delta_repeated_df(repeated_df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["code_name", "p", "repeat_id", "seed", "num_shots"]
    osd_df = repeated_df[repeated_df["decoder_name"] == "BP-OSD"].copy()
    lsd_df = repeated_df[repeated_df["decoder_name"] == "BP-LSD"].copy()

    merged = osd_df.merge(lsd_df, on=key_cols, suffixes=("_osd", "_lsd"))
    delta_df = pd.DataFrame(
        {
            "code_name": merged["code_name"],
            "p": merged["p"],
            "repeat_id": merged["repeat_id"],
            "seed": merged["seed"],
            "num_shots": merged["num_shots"],
            "delta_ler": merged["logical_error_rate_lsd"] - merged["logical_error_rate_osd"],
            "delta_avg_latency": merged["avg_latency_ms_per_shot_lsd"] - merged["avg_latency_ms_per_shot_osd"],
            "delta_p95_latency": merged["p95_latency_ms_per_shot_lsd"] - merged["p95_latency_ms_per_shot_osd"],
            "delta_p99_latency": merged["p99_latency_ms_per_shot_lsd"] - merged["p99_latency_ms_per_shot_osd"],
            "delta_avg_bp_iterations": merged["avg_bp_iterations_per_shot_lsd"] - merged["avg_bp_iterations_per_shot_osd"],
            "delta_bp_converged_fraction": merged["bp_converged_fraction_lsd"] - merged["bp_converged_fraction_osd"],
            "n": merged["n_osd"],
            "k": merged["k_osd"],
            "d": merged["d_osd"],
            "mx_rows": merged["mx_rows_osd"],
            "mx_cols": merged["mx_cols_osd"],
            "mz_rows": merged["mz_rows_osd"],
            "mz_cols": merged["mz_cols_osd"],
            "mx_density": merged["mx_density_osd"],
            "mz_density": merged["mz_density_osd"],
            "syndrome_avg_weight": merged["syndrome_avg_weight_osd"],
            "syndrome_p95_weight": merged["syndrome_p95_weight_osd"],
        }
    )
    return delta_df


def build_delta_summary_df(delta_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for _, subdf in delta_df.groupby(["code_name", "p"], sort=False):
        first = subdf.iloc[0]
        summary_rows.append(
            {
                "code_name": first["code_name"],
                "p": float(first["p"]),
                "num_repeats": int(len(subdf)),
                "delta_ler_mean": float(subdf["delta_ler"].mean()),
                "delta_ler_std": safe_std(subdf["delta_ler"]),
                "delta_avg_latency_mean": float(subdf["delta_avg_latency"].mean()),
                "delta_avg_latency_std": safe_std(subdf["delta_avg_latency"]),
                "delta_p95_latency_mean": float(subdf["delta_p95_latency"].mean()),
                "delta_p95_latency_std": safe_std(subdf["delta_p95_latency"]),
                "delta_p99_latency_mean": float(subdf["delta_p99_latency"].mean()),
                "delta_p99_latency_std": safe_std(subdf["delta_p99_latency"]),
                "delta_avg_bp_iterations_mean": float(subdf["delta_avg_bp_iterations"].mean()),
                "delta_avg_bp_iterations_std": safe_std(subdf["delta_avg_bp_iterations"]),
                "delta_bp_converged_fraction_mean": float(subdf["delta_bp_converged_fraction"].mean()),
                "delta_bp_converged_fraction_std": safe_std(subdf["delta_bp_converged_fraction"]),
                "delta_avg_latency_positive_fraction": float((subdf["delta_avg_latency"] > 0).mean()),
                "delta_p95_latency_positive_fraction": float((subdf["delta_p95_latency"] > 0).mean()),
                "delta_ler_positive_fraction": float((subdf["delta_ler"] > 0).mean()),
                "n": int(first["n"]),
                "k": int(first["k"]),
                "d": first["d"],
                "mx_rows": int(first["mx_rows"]),
                "mx_cols": int(first["mx_cols"]),
                "mz_rows": int(first["mz_rows"]),
                "mz_cols": int(first["mz_cols"]),
                "mx_density": float(first["mx_density"]),
                "mz_density": float(first["mz_density"]),
                "syndrome_avg_weight_mean": float(subdf["syndrome_avg_weight"].mean()),
                "syndrome_p95_weight_mean": float(subdf["syndrome_p95_weight"].mean()),
            }
        )
    return pd.DataFrame(summary_rows)


def build_code_level_summary(summary_df: pd.DataFrame, delta_summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code_name in dict.fromkeys(summary_df["code_name"]):
        code_df = summary_df[summary_df["code_name"] == code_name]
        osd_df = code_df[code_df["decoder_name"] == "BP-OSD"]
        lsd_df = code_df[code_df["decoder_name"] == "BP-LSD"]
        delta_df = delta_summary_df[delta_summary_df["code_name"] == code_name]
        first = code_df.iloc[0]

        rows.append(
            {
                "code_name": code_name,
                "n": int(first["n"]),
                "k": int(first["k"]),
                "d": first["d"],
                "mx_rows": int(first["mx_rows"]),
                "mx_cols": int(first["mx_cols"]),
                "mz_rows": int(first["mz_rows"]),
                "mz_cols": int(first["mz_cols"]),
                "mx_density": float(first["mx_density"]),
                "mz_density": float(first["mz_density"]),
                "bp_osd_mean_ler_over_p": float(osd_df["ler_mean"].mean()),
                "bp_lsd_mean_ler_over_p": float(lsd_df["ler_mean"].mean()),
                "bp_osd_mean_p95_latency_over_p": float(osd_df["p95_latency_mean"].mean()),
                "bp_lsd_mean_p95_latency_over_p": float(lsd_df["p95_latency_mean"].mean()),
                "mean_delta_ler_over_p": float(delta_df["delta_ler_mean"].mean()),
                "mean_delta_p95_latency_over_p": float(delta_df["delta_p95_latency_mean"].mean()),
                "mean_delta_avg_latency_over_p": float(delta_df["delta_avg_latency_mean"].mean()),
                "mean_syndrome_avg_weight_over_p": float(code_df["syndrome_avg_weight_mean"].mean()),
                "mean_avg_bp_iterations_over_p_bp_osd": float(osd_df["avg_bp_iterations_mean"].mean()),
                "mean_avg_bp_iterations_over_p_bp_lsd": float(lsd_df["avg_bp_iterations_mean"].mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_delta_p95_latency(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    code_order = list(dict.fromkeys(summary_df["code_name"]))

    for code_name in code_order:
        subdf = summary_df[summary_df["code_name"] == code_name].sort_values("p")
        ax.errorbar(
            subdf["p"],
            subdf["delta_p95_latency_mean"],
            yerr=subdf["delta_p95_latency_std"],
            marker="o",
            linewidth=2,
            capsize=3,
            label=code_name,
        )

    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Physical Z error rate p")
    ax.set_ylabel("delta p95 latency (BP-LSD - BP-OSD) [ms]")
    ax.set_title("Paired Decoder p95 Latency Delta")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    results_dir = PROJECT_ROOT / "data" / "results"
    repeated_path = results_dir / "results_repeated.csv"
    repeated_df = pd.read_csv(repeated_path)
    if repeated_df.empty:
        raise ValueError(f"No rows found in {repeated_path}")

    delta_repeated_df = build_delta_repeated_df(repeated_df)
    delta_summary_df = build_delta_summary_df(delta_repeated_df)
    repeated_summary_df = pd.read_csv(results_dir / "results_repeated_summary.csv")
    code_level_summary_df = build_code_level_summary(repeated_summary_df, delta_summary_df)

    delta_repeated_path = results_dir / "decoder_delta_repeated.csv"
    delta_summary_path = results_dir / "decoder_delta_summary.csv"
    code_level_summary_path = results_dir / "code_level_summary.csv"
    delta_plot_path = results_dir / "delta_p95_latency_by_code.png"

    delta_repeated_df.to_csv(delta_repeated_path, index=False)
    delta_summary_df.to_csv(delta_summary_path, index=False)
    code_level_summary_df.to_csv(code_level_summary_path, index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    plot_delta_p95_latency(delta_summary_df, delta_plot_path)

    print(f"Saved {delta_repeated_path}")
    print(f"Saved {delta_summary_path}")
    print(f"Saved {code_level_summary_path}")
    print(f"Saved {delta_plot_path}")


if __name__ == "__main__":
    main()
