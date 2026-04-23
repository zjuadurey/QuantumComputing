"""Run a tiny downstream-learning probe for the ShadowFluid rebuttal.

This script loads the cached tiny dataset, fits the same PCA + Ridge learner to
two feature families, and writes a rebuttal-friendly summary:

- coherence-aware Shadow features
- task-level low-pass density inputs

Prepared for the response to Reviewer oB6r.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


DEFAULT_REVIEWER = "oB6r"
DEFAULT_TRAIN_SIZES = [16, 32, 64]
DEFAULT_NUM_SPLITS = 3


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def float_str(x: float, digits: int = 6) -> str:
    xf = float(x)
    if math.isnan(xf):
        return "nan"
    if abs(xf) < 1e-12:
        return "0"
    if abs(xf) >= 1e-2 and abs(xf) < 1e3:
        return f"{xf:.{digits}f}".rstrip("0").rstrip(".")
    return f"{xf:.{digits}e}"


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fit_predict_mse(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_dim_cap: int,
    ridge_alpha: float,
) -> tuple[float, int]:
    pca_dim = min(int(feature_dim_cap), X_train.shape[0], X_train.shape[1])
    pca = PCA(n_components=pca_dim)
    X_train_red = pca.fit_transform(X_train)
    X_test_red = pca.transform(X_test)

    model = Ridge(alpha=float(ridge_alpha))
    model.fit(X_train_red, y_train)
    pred = model.predict(X_test_red)
    mse = float(mean_squared_error(y_test, pred))
    return mse, int(pca_dim)


def aggregate_main_rows(
    split_rows: list[dict[str, object]],
    train_sizes: list[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for train_size in train_sizes:
        group = [row for row in split_rows if int(row["train_size"]) == int(train_size)]
        if not group:
            continue
        baseline_rows = [row for row in group if row["feature_type"] == "Task-level low-pass density"]
        shadow_rows = [row for row in group if row["feature_type"] == "Shadow coherence Z"]

        baseline_mean = float(np.mean([float(row["test_mse"]) for row in baseline_rows]))
        shadow_mean = float(np.mean([float(row["test_mse"]) for row in shadow_rows]))

        for feature_type in ["Task-level low-pass density", "Shadow coherence Z"]:
            feature_rows = [row for row in group if row["feature_type"] == feature_type]
            mses = [float(row["test_mse"]) for row in feature_rows]
            mse_mean = float(np.mean(mses))
            mse_std = float(np.std(mses, ddof=0)) if len(mses) > 1 else float("nan")
            feature_dim = int(feature_rows[0]["feature_dim"])

            if feature_type == "Shadow coherence Z":
                rel_gain = float((baseline_mean - shadow_mean) / baseline_mean) if baseline_mean != 0.0 else float("nan")
            else:
                rel_gain = 0.0

            rows.append(
                {
                    "train_size": int(train_size),
                    "feature_type": feature_type,
                    "learner": "PCA + Ridge",
                    "feature_dim": int(feature_dim),
                    "test_mse": mse_mean,
                    "std": mse_std,
                    "relative_gain_vs_lowpass": rel_gain,
                }
            )
    return rows


def to_markdown_table(rows: list[dict[str, object]]) -> str:
    columns = [
        "train_size",
        "feature_type",
        "learner",
        "feature_dim",
        "test_mse",
        "std",
        "relative_gain_vs_lowpass",
    ]
    labels = {
        "train_size": "Train size",
        "feature_type": "Feature type",
        "learner": "Learner",
        "feature_dim": "Feature dim",
        "test_mse": "Test MSE",
        "std": "Std",
        "relative_gain_vs_lowpass": "Relative gain vs low-pass",
    }
    header = "| " + " | ".join(labels[c] for c in columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        vals = []
        for col in columns:
            value = row[col]
            if isinstance(value, str):
                vals.append(value)
            elif col == "train_size" or col == "feature_dim":
                vals.append(str(int(value)))
            else:
                vals.append(float_str(float(value)))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *body])


def plot_results(rows: list[dict[str, object]], path: Path) -> None:
    feature_order = ["Task-level low-pass density", "Shadow coherence Z"]
    colors = {
        "Task-level low-pass density": "#d55e00",
        "Shadow coherence Z": "#0072b2",
    }

    plt.figure(figsize=(6.0, 4.0))
    for feature_type in feature_order:
        feature_rows = [row for row in rows if row["feature_type"] == feature_type]
        feature_rows.sort(key=lambda row: int(row["train_size"]))
        x = [int(row["train_size"]) for row in feature_rows]
        y = [float(row["test_mse"]) for row in feature_rows]
        s = [
            0.0 if math.isnan(float(row["std"])) else float(row["std"])
            for row in feature_rows
        ]

        plt.plot(x, y, marker="o", linewidth=2, color=colors[feature_type], label=feature_type)
        if any(val > 0.0 for val in s):
            y_lo = [max(0.0, yi - si) for yi, si in zip(y, s)]
            y_hi = [yi + si for yi, si in zip(y, s)]
            plt.fill_between(x, y_lo, y_hi, color=colors[feature_type], alpha=0.15)

    plt.xlabel("Train size")
    plt.ylabel("Test MSE")
    plt.title("Tiny downstream-learning probe")
    plt.legend(frameon=False)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def generate_summary(
    *,
    reviewer: str,
    out_path: Path,
    dataset_path: Path,
    manifest_path: Path,
    main_rows: list[dict[str, object]],
    train_sizes: list[int],
    num_splits: int,
    ridge_alpha: float,
    feature_dim_cap: int,
    n_samples: int,
    N: int,
    K0: float,
    dt: float,
    qx: int,
    qy: int,
    alphas: list[float],
    times: list[float],
) -> str:
    shadow_rows = [row for row in main_rows if row["feature_type"] == "Shadow coherence Z"]
    lowpass_rows = [row for row in main_rows if row["feature_type"] == "Task-level low-pass density"]

    wins = 0
    rel_gains = []
    for train_size in train_sizes:
        shadow_row = next((row for row in shadow_rows if int(row["train_size"]) == int(train_size)), None)
        lowpass_row = next((row for row in lowpass_rows if int(row["train_size"]) == int(train_size)), None)
        if shadow_row is None or lowpass_row is None:
            continue
        if float(shadow_row["test_mse"]) < float(lowpass_row["test_mse"]):
            wins += 1
        rel_gains.append(float(shadow_row["relative_gain_vs_lowpass"]))

    max_gain = max(rel_gains) if rel_gains else float("nan")
    min_gain = min(rel_gains) if rel_gains else float("nan")
    all_better = wins == len(train_sizes)

    if all_better:
        result_line = (
            f"in the tested regime, the coherence-aware Shadow features achieve lower test MSE than the task-level low-pass density inputs at all reported train sizes."
        )
    else:
        result_line = (
            f"in the tested regime, the coherence-aware Shadow features improve over the task-level low-pass density inputs in {wins} out of {len(train_sizes)} reported train sizes."
        )

    rebuttal_paragraph = (
        f"In a small downstream-learning probe prepared for Reviewer {reviewer}, we compared two matched-dimensional input representations under the same PCA + Ridge learner: "
        f"(i) coherence-aware Shadow features obtained from the reduced `Z(t)` representation and (ii) task-level low-pass density inputs at the same cutoff `K0 = {float_str(K0)}`. "
        f"The downstream target was the next-step unresolved high-frequency energy `E_HF(t+dt)`, which is not directly available from the task truncation. "
        f"On the single-cosine benchmark with `N = {N}`, `q = ({qx},{qy})`, `alpha in {{{', '.join(float_str(a) for a in alphas)}}}`, and `t in {{{', '.join(float_str(t) for t in times)}}}`, "
        f"{result_line} Across train sizes `{train_sizes}`, the relative gain of the Shadow features ranges from {float_str(min_gain)} to {float_str(max_gain)}. "
        f"This small probe suggests that ShadowFluid can act as a compact coherence-aware frontend for downstream learning, while remaining far from a full AI benchmark."
    )

    summary = "\n".join(
        [
            "# Tiny Shadow Learning Probe",
            "",
            f"Prepared for Reviewer {reviewer}.",
            "",
            "## Benchmark",
            "",
            f"- Single-cosine potential with `q = ({qx}, {qy})`.",
            f"- Grid: `N = {N}`.",
            f"- Cutoff: `K0 = {float_str(K0)}`.",
            f"- Couplings: `alpha in {{{', '.join(float_str(a) for a in alphas)}}}`.",
            f"- Time points: `t in {{{', '.join(float_str(t) for t in times)}}}`, with `dt = {float_str(dt)}`.",
            f"- Initial-condition family: Gaussian vortex only (multiple deterministic seeds inside the same family).",
            f"- Total supervised samples: `{n_samples}`.",
            "",
            "## Features and Target",
            "",
            "- Shadow feature type: flattened real/imaginary parts of the reduced coherence matrices `Z_1(t)` and `Z_2(t)` built from the existing ShadowFluid reduced evolution.",
            "- Low-pass baseline feature type: the task-level low-pass density field at the same cutoff `K0`.",
            "- Downstream target: next-step unresolved high-frequency energy `E_HF(t+dt)`.",
            "",
            "## Learner",
            "",
            f"- Same learner for both feature families: `PCA(feature_dim <= {feature_dim_cap}) + Ridge(alpha={float_str(ridge_alpha)})`.",
            f"- Train-size sweep: `{train_sizes}`.",
            f"- Splits: `{num_splits}` deterministic random splits with shared train/test partitions across feature families.",
            f"- Cached dataset: `{dataset_path.name}` with sample manifest `{manifest_path.name}`.",
            "",
            "## Main Result Table",
            "",
            to_markdown_table(main_rows),
            "",
            "## Short Interpretation",
            "",
            rebuttal_paragraph,
            "",
        ]
    )
    out_path.write_text(summary)
    return rebuttal_paragraph


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the tiny ShadowFluid downstream-learning probe")
    p.add_argument("--reviewer", default=DEFAULT_REVIEWER)
    p.add_argument("--data", default="results/tiny_shadow_learning/dataset.npz")
    p.add_argument("--manifest", default="results/tiny_shadow_learning/dataset_manifest.csv")
    p.add_argument("--out-dir", default="results/tiny_shadow_learning")
    p.add_argument("--train-sizes", default=",".join(map(str, DEFAULT_TRAIN_SIZES)))
    p.add_argument("--num-splits", type=int, default=DEFAULT_NUM_SPLITS)
    p.add_argument("--feature-dim-cap", type=int, default=16)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data)
    manifest_path = Path(args.manifest)
    data = np.load(data_path)

    X_shadow = np.asarray(data["shadow_features"], dtype=np.float64)
    X_lowpass = np.asarray(data["lowpass_density_features"], dtype=np.float64)
    y = np.asarray(data["targets"], dtype=np.float64)

    train_sizes = [size for size in parse_int_list(args.train_sizes) if size < len(y)]
    num_splits = int(args.num_splits)
    feature_dim_cap = int(args.feature_dim_cap)
    ridge_alpha = float(args.ridge_alpha)

    split_rows: list[dict[str, object]] = []
    for split_id in range(num_splits):
        rng = np.random.default_rng(20260418 + split_id)
        perm = rng.permutation(len(y))

        for train_size in train_sizes:
            train_idx = perm[:train_size]
            test_idx = perm[train_size:]

            for feature_type, X in [
                ("Task-level low-pass density", X_lowpass),
                ("Shadow coherence Z", X_shadow),
            ]:
                mse, feature_dim = fit_predict_mse(
                    X[train_idx],
                    y[train_idx],
                    X[test_idx],
                    y[test_idx],
                    feature_dim_cap=feature_dim_cap,
                    ridge_alpha=ridge_alpha,
                )
                split_rows.append(
                    {
                        "reviewer": args.reviewer,
                        "split_id": int(split_id),
                        "train_size": int(train_size),
                        "test_size": int(len(test_idx)),
                        "feature_type": feature_type,
                        "learner": "PCA + Ridge",
                        "feature_dim": int(feature_dim),
                        "test_mse": float(mse),
                    }
                )

    main_rows = aggregate_main_rows(split_rows, train_sizes)
    feature_rank = {
        "Task-level low-pass density": 0,
        "Shadow coherence Z": 1,
    }
    main_rows.sort(key=lambda row: (int(row["train_size"]), feature_rank[str(row["feature_type"])]))

    split_csv = out_dir / "split_results.csv"
    main_csv = out_dir / "main_results.csv"
    plot_path = out_dir / "train_size_sweep.png"
    summary_path = out_dir / "summary.md"

    write_csv(
        split_csv,
        split_rows,
        [
            "reviewer",
            "split_id",
            "train_size",
            "test_size",
            "feature_type",
            "learner",
            "feature_dim",
            "test_mse",
        ],
    )
    write_csv(
        main_csv,
        main_rows,
        [
            "train_size",
            "feature_type",
            "learner",
            "feature_dim",
            "test_mse",
            "std",
            "relative_gain_vs_lowpass",
        ],
    )
    plot_results(main_rows, plot_path)

    rebuttal_paragraph = generate_summary(
        reviewer=str(args.reviewer),
        out_path=summary_path,
        dataset_path=data_path,
        manifest_path=manifest_path,
        main_rows=main_rows,
        train_sizes=train_sizes,
        num_splits=num_splits,
        ridge_alpha=ridge_alpha,
        feature_dim_cap=feature_dim_cap,
        n_samples=int(len(y)),
        N=int(data["N"][0]),
        K0=float(data["K0"][0]),
        dt=float(data["dt"][0]),
        qx=int(data["qx"][0]),
        qy=int(data["qy"][0]),
        alphas=sorted({float(x) for x in data["alphas"]}),
        times=sorted({float(x) for x in data["times"]}),
    )

    table_md = to_markdown_table(main_rows)

    print(f"Wrote split results: {split_csv}")
    print(f"Wrote main results: {main_csv}")
    print(f"Wrote plot: {plot_path}")
    print(f"Wrote summary: {summary_path}")
    print("Final main table:")
    print(table_md)
    print("\nRebuttal-ready paragraph:")
    print(rebuttal_paragraph)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
