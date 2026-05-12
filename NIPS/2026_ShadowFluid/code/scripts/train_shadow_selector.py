"""Train and benchmark a lightweight learned ShadowFluid reference selector."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shiftflow import core_v1, selector_learning as sl  # noqa: E402


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_model(model_name: str, seed: int, hidden_sizes: list[int], max_iter: int):
    """Construct a simple candidate classifier."""
    if model_name == "logistic":
        clf = LogisticRegression(
            max_iter=int(max_iter),
            class_weight="balanced",
            random_state=int(seed),
        )
    elif model_name == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=tuple(int(x) for x in hidden_sizes),
            random_state=int(seed),
            max_iter=int(max_iter),
            activation="relu",
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", clf),
        ]
    )


def aggregate_strategy_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate per-sample evaluation rows by strategy."""
    out: list[dict[str, object]] = []
    strategies = sorted({str(row["strategy"]) for row in rows})
    for strategy in strategies:
        group = [row for row in rows if str(row["strategy"]) == strategy]
        out.append(
            {
                "strategy": strategy,
                "num_samples": int(len(group)),
                "mean_objective": float(np.mean([float(row["objective"]) for row in group])),
                "mean_oracle_gap": float(np.mean([float(row["oracle_gap"]) for row in group])),
                "mean_err_rho_vs_full": float(np.mean([float(row["mean_err_rho_vs_full"]) for row in group])),
                "mean_leakage": float(np.mean([float(row["mean_leakage"]) for row in group])),
                "exact_match_rate": float(np.mean([float(row["exact_match_vs_oracle"]) for row in group])),
            }
        )
    return out


def summarize_results(
    *,
    out_path: Path,
    model_name: str,
    hidden_sizes: list[int],
    train_sample_count: int,
    test_sample_count: int,
    class_metrics: dict[str, float],
    aggregate_rows: list[dict[str, object]],
) -> None:
    """Write a short Markdown summary for quick iteration."""
    header = [
        "# Shadow Selector Benchmark",
        "",
        f"- Model: `{model_name}`",
        f"- Hidden sizes: `{hidden_sizes}`",
        f"- Train samples: `{train_sample_count}`",
        f"- Test samples: `{test_sample_count}`",
        "",
        "## Candidate Classification",
        "",
        f"- Accuracy: `{class_metrics['accuracy']:.6f}`",
        f"- Precision: `{class_metrics['precision']:.6f}`",
        f"- Recall: `{class_metrics['recall']:.6f}`",
        f"- F1: `{class_metrics['f1']:.6f}`",
        "",
        "## Strategy Summary",
        "",
        "| Strategy | Samples | Mean objective | Mean oracle gap | Mean density error | Mean leakage | Exact match rate |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in aggregate_rows:
        header.append(
            "| "
            + " | ".join(
                [
                    str(row["strategy"]),
                    str(int(row["num_samples"])),
                    f"{float(row['mean_objective']):.6f}",
                    f"{float(row['mean_oracle_gap']):.6f}",
                    f"{float(row['mean_err_rho_vs_full']):.6f}",
                    f"{float(row['mean_leakage']):.6f}",
                    f"{float(row['exact_match_rate']):.6f}",
                ]
            )
            + " |"
        )
    out_path.write_text("\n".join(header) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train a lightweight learned ShadowFluid selector")
    p.add_argument("--samples", default=str(ROOT / "results" / "shadow_selector" / "samples.csv"))
    p.add_argument("--candidates", default=str(ROOT / "results" / "shadow_selector" / "candidates.csv"))
    p.add_argument("--out-dir", default=str(ROOT / "results" / "shadow_selector" / "benchmark"))
    p.add_argument("--test-fraction", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=20260505)
    p.add_argument("--model", choices=["logistic", "mlp"], default="mlp")
    p.add_argument("--hidden-sizes", default="32,16")
    p.add_argument("--max-iter", type=int, default=500)
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = pd.read_csv(args.samples)
    candidates = pd.read_csv(args.candidates)
    feature_cols = [col for col in candidates.columns if col.startswith("feat_")]

    sample_ids = samples["sample_id"].to_numpy(dtype=int)
    if len(sample_ids) < 2:
        raise ValueError("Need at least two samples to create train/test splits")
    rng = np.random.default_rng(int(args.seed))
    perm = rng.permutation(sample_ids)
    raw_test = int(round(len(perm) * float(args.test_fraction)))
    test_count = min(max(1, raw_test), len(perm) - 1)
    test_sample_ids = set(int(x) for x in perm[:test_count])
    train_sample_ids = set(int(x) for x in perm[test_count:])

    non_anchor = candidates[candidates["is_anchor"] == 0].copy()
    train_rows = non_anchor[non_anchor["sample_id"].isin(train_sample_ids)]
    test_rows = non_anchor[non_anchor["sample_id"].isin(test_sample_ids)]

    X_train = train_rows[feature_cols].to_numpy(dtype=float)
    y_train = train_rows["label_selected"].to_numpy(dtype=int)
    X_test = test_rows[feature_cols].to_numpy(dtype=float)
    y_test = test_rows["label_selected"].to_numpy(dtype=int)

    model = build_model(
        model_name=str(args.model),
        seed=int(args.seed),
        hidden_sizes=parse_int_list(args.hidden_sizes),
        max_iter=int(args.max_iter),
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    class_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    eval_rows: list[dict[str, object]] = []
    for sample_id in sorted(test_sample_ids):
        sample_row = samples[samples["sample_id"] == sample_id].iloc[0]
        cand_rows = candidates[candidates["sample_id"] == sample_id].sort_values("candidate_rank")

        components = sl.components_from_json(str(sample_row["components_json"]))
        pool_flat = sl.reference_set_from_text(str(sample_row["pool_text"]))
        oracle_ref = sl.reference_set_from_text(str(sample_row["oracle_reference_text"]))
        coupling_ref = sl.reference_set_from_text(str(sample_row["coupling_reference_text"]))
        low_energy_ref = sl.reference_set_from_text(str(sample_row["low_energy_reference_text"]))
        eval_seeds = parse_int_list(str(sample_row["eval_seeds"]))
        eval_times = [float(x) for x in str(sample_row["eval_times"]).split(",") if str(x).strip()]
        budget = int(sample_row["budget"])
        N = int(sample_row["N"])
        anchor_flat = sl.anchor_flat_from_r0((0, 0), N)
        H_dense = core_v1.build_H_dense(N, components)
        eig = core_v1.eigendecompose(H_dense)

        other_rows = cand_rows[cand_rows["is_anchor"] == 0]
        X_sample = other_rows[feature_cols].to_numpy(dtype=float)
        probs = model.predict_proba(X_sample)[:, 1]
        learned_ref = sl.make_budgeted_reference_set(
            pool_flat=pool_flat,
            budget=budget,
            anchor_flat=anchor_flat,
            scores=probs,
        )
        random_ref = sl.heuristic_reference_set(
            pool_flat=pool_flat,
            budget=budget,
            components=components,
            N=N,
            anchor_flat=anchor_flat,
            strategy="random",
            rng_seed=int(args.seed) + int(sample_id),
        )

        strategy_refs = {
            "oracle": oracle_ref,
            "learned": learned_ref,
            "coupling_greedy": coupling_ref,
            "low_energy": low_energy_ref,
            "random": random_ref,
        }

        oracle_objective = None
        for strategy, ref in strategy_refs.items():
            result = sl.evaluate_reference_set(
                nx=int(sample_row["nx"]),
                K0=float(sample_row["K0"]),
                components=components,
                R_flat=ref,
                seeds=eval_seeds,
                times=eval_times,
                H_dense=H_dense,
                eig=eig,
            )
            if strategy == "oracle":
                oracle_objective = float(result.objective)
            eval_rows.append(
                {
                    "sample_id": int(sample_id),
                    "strategy": strategy,
                    "reference_text": result.reference_text,
                    "objective": float(result.objective),
                    "mean_err_rho_vs_full": float(result.mean_err_rho_vs_full),
                    "mean_leakage": float(result.mean_leakage),
                    "exact_match_vs_oracle": int(result.reference_text == sl.reference_set_to_text(oracle_ref)),
                    "oracle_gap": 0.0,  # filled below once oracle is known
                }
            )

        if oracle_objective is None:
            raise RuntimeError("Oracle objective was not computed")
        for row in eval_rows[-len(strategy_refs) :]:
            row["oracle_gap"] = float(row["objective"]) - float(oracle_objective)

    aggregate_rows = aggregate_strategy_rows(eval_rows)

    eval_path = out_dir / "per_sample_results.csv"
    summary_path = out_dir / "strategy_summary.csv"
    md_path = out_dir / "summary.md"

    eval_fields = [
        "sample_id",
        "strategy",
        "reference_text",
        "objective",
        "mean_err_rho_vs_full",
        "mean_leakage",
        "exact_match_vs_oracle",
        "oracle_gap",
    ]
    summary_fields = [
        "strategy",
        "num_samples",
        "mean_objective",
        "mean_oracle_gap",
        "mean_err_rho_vs_full",
        "mean_leakage",
        "exact_match_rate",
    ]
    write_csv(eval_path, eval_rows, eval_fields)
    write_csv(summary_path, aggregate_rows, summary_fields)
    summarize_results(
        out_path=md_path,
        model_name=str(args.model),
        hidden_sizes=parse_int_list(args.hidden_sizes),
        train_sample_count=int(len(train_sample_ids)),
        test_sample_count=int(len(test_sample_ids)),
        class_metrics=class_metrics,
        aggregate_rows=aggregate_rows,
    )

    print(f"Wrote per-sample results: {eval_path}")
    print(f"Wrote summary CSV: {summary_path}")
    print(f"Wrote summary Markdown: {md_path}")
    print(
        "Candidate classification:",
        ", ".join(f"{k}={v:.6f}" for k, v in class_metrics.items()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
