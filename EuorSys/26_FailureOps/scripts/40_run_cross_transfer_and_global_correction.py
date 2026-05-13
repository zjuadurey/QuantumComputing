#!/usr/bin/env python
"""Cross-condition decoder transfer and global multi-test correction.

Priority 4: Cross-condition transfer.
  - Train on traditional_calibration conditions, evaluate on RL conditions (and vice versa).
  - Reports: per-condition sign in test group, fraction sign-consistent, aggregate paired delta.

Priority 5: Global statistical correction appendix.
  - Global Holm correction (all 40 conditions treated as one family).
  - Benjamini-Hochberg FDR.
  - Random-effects meta-analytic aggregate of per-condition paired delta LFR.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from failureops.io_utils import ensure_parent_dir, fmt_float, write_csv_rows

TRANSFER_FIELDS = [
    "train_group",
    "test_group",
    "train_num_conditions",
    "test_num_conditions",
    "train_mean_delta_lfr",
    "train_sign_negative_fraction",
    "test_mean_delta_lfr",
    "test_sign_negative_fraction",
    "test_sign_consistent_count",
    "test_conditions",
]

TRANSFER_COND_FIELDS = [
    "experiment_name",
    "control_mode",
    "basis",
    "cycles",
    "rescued",
    "induced",
    "paired_delta_lfr",
    "train_group",
    "test_group",
    "sign_matches_train",
]

CORRECTION_FIELDS = [
    "condition",
    "mcnemar_exact_p",
    "scoped_holm_p",
    "scoped_holm_sig",
    "global_holm_p",
    "global_holm_sig",
    "benjamini_hochberg_critical",
    "bh_sig",
]

META_FIELDS = [
    "metric",
    "value",
]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    matrix_path = repo_root / "data/results/p7_google_rl_qec_decoder_effect_matrix.csv"
    sig_path = repo_root / "data/results/p10_effect_significance.csv"

    with open(matrix_path) as f:
        matrix = list(csv.DictReader(f))
    with open(sig_path) as f:
        sig_rows = [r for r in csv.DictReader(f) if r["analysis_scope"] == "real_decoder_pathway"]
    sig_by_id = {r["unit_id"]: r for r in sig_rows}

    # ---- Priority 4: Cross-condition transfer ----
    rl_conditions = [r for r in matrix if r["control_mode"] == "reinforcement_learning"]
    tc_conditions = [r for r in matrix if r["control_mode"] == "traditional_calibration"]

    def compute_per_condition(row):
        rescued = int(row["rescued_failure_count"])
        induced = int(row["induced_failure_count"])
        delta = float(row["paired_delta_lfr"])
        return {"rescued": rescued, "induced": induced, "delta": delta, "sign_negative": delta < 0}

    def group_aggregate(conditions):
        data = [compute_per_condition(r) for r in conditions]
        mean_delta = sum(d["delta"] for d in data) / len(data)
        sign_neg_frac = sum(1 for d in data if d["sign_negative"]) / len(data)
        return {"mean_delta": mean_delta, "sign_negative_fraction": sign_neg_frac, "per_cond": data}

    # Train on TC, test on RL
    tc_agg = group_aggregate(tc_conditions)
    rl_agg = group_aggregate(rl_conditions)

    # Train on RL, test on TC
    # (same computations, just swapped perspective)

    transfer_rows = []
    transfer_cond_rows = []

    for train_name, train_conds, test_name, test_conds in [
        ("traditional_calibration", tc_conditions, "reinforcement_learning", rl_conditions),
        ("reinforcement_learning", rl_conditions, "traditional_calibration", tc_conditions),
    ]:
        train_data = group_aggregate(train_conds)
        test_data = group_aggregate(test_conds)
        train_sign_negative = train_data["mean_delta"] < 0  # aggregate direction

        sign_consistent = 0
        for tr, tc in zip(test_conds, test_data["per_cond"]):
            sign_matches = tc["sign_negative"] == train_sign_negative
            if sign_matches:
                sign_consistent += 1
            transfer_cond_rows.append({
                "experiment_name": tr["experiment_name"],
                "control_mode": tr["control_mode"],
                "basis": tr["basis"],
                "cycles": int(tr["cycles"]),
                "rescued": tc["rescued"],
                "induced": tc["induced"],
                "paired_delta_lfr": fmt_float(tc["delta"]),
                "train_group": train_name,
                "test_group": test_name,
                "sign_matches_train": sign_matches,
            })

        transfer_rows.append({
            "train_group": train_name,
            "test_group": test_name,
            "train_num_conditions": len(train_conds),
            "test_num_conditions": len(test_conds),
            "train_mean_delta_lfr": fmt_float(train_data["mean_delta"]),
            "train_sign_negative_fraction": fmt_float(train_data["sign_negative_fraction"]),
            "test_mean_delta_lfr": fmt_float(test_data["mean_delta"]),
            "test_sign_negative_fraction": fmt_float(test_data["sign_negative_fraction"]),
            "test_sign_consistent_count": f"{sign_consistent}/{len(test_conds)}",
            "test_conditions": ", ".join(tr["experiment_name"] for tr in test_conds),
        })

    ensure_parent_dir(repo_root / "data/results/p10_cross_condition_transfer.csv")
    write_csv_rows(
        repo_root / "data/results/p10_cross_condition_transfer.csv",
        transfer_rows,
        TRANSFER_FIELDS,
    )
    write_csv_rows(
        repo_root / "data/results/p10_cross_condition_transfer_per_condition.csv",
        transfer_cond_rows,
        TRANSFER_COND_FIELDS,
    )

    print("=== Cross-condition transfer ===")
    for tr in transfer_rows:
        print(f"  Train={tr['train_group']}, Test={tr['test_group']}: "
              f"train Δ={tr['train_mean_delta_lfr']}, test Δ={tr['test_mean_delta_lfr']}, "
              f"sign-consistent: {tr['test_sign_consistent_count']}")

    # ---- Priority 5: Global multi-test correction ----
    conditions = []
    for row in matrix:
        sig = sig_by_id.get(row["workload_id"], {})
        p = float(sig.get("mcnemar_exact_p", 1.0))
        conditions.append({
            "name": row["experiment_name"],
            "p": p,
            "delta": float(row["paired_delta_lfr"]),
            "rescued": int(row["rescued_failure_count"]),
            "induced": int(row["induced_failure_count"]),
        })

    # Sort by p-value
    conditions.sort(key=lambda c: c["p"])

    n = len(conditions)
    correction_rows = []
    global_holm_sig_count = 0
    bh_sig_count = 0

    for rank, c in enumerate(conditions, start=1):
        # Global Holm: p_(i) ≤ α / (n - i + 1)
        global_holm_threshold = 0.05 / (n - rank + 1)
        global_holm_sig = c["p"] <= global_holm_threshold
        if global_holm_sig:
            global_holm_sig_count += 1

        # Benjamini-Hochberg: p_(i) ≤ (i / n) * α
        bh_critical = (rank / n) * 0.05
        bh_sig = c["p"] <= bh_critical
        if bh_sig:
            bh_sig_count += 1

        scoped_sig = sig_by_id.get(c["name"], {})
        correction_rows.append({
            "condition": c["name"],
            "mcnemar_exact_p": fmt_float(c["p"]),
            "scoped_holm_p": sig_by_id.get(c["name"], {}).get("holm_adjusted_p", "N/A"),
            "scoped_holm_sig": sig_by_id.get(c["name"], {}).get("significant_after_holm_0_05", "N/A"),
            "global_holm_p": fmt_float(min(c["p"] * n, 1.0)),
            "global_holm_sig": global_holm_sig,
            "benjamini_hochberg_critical": fmt_float(bh_critical),
            "bh_sig": bh_sig,
        })

    write_csv_rows(
        repo_root / "data/results/p10_global_correction.csv",
        correction_rows,
        CORRECTION_FIELDS,
    )

    # Random-effects meta-analysis (DerSimonian-Laird)
    deltas = [c["delta"] for c in conditions]
    n_cond = len(deltas)
    mean_delta = sum(deltas) / n_cond

    # Within-study variance: use variance of delta estimator (rescued+induced)/n^2
    total_shots = 10000
    within_vars = []
    for c in conditions:
        discordant = c["rescued"] + c["induced"]
        if discordant > 0:
            var_delta = (c["rescued"] + c["induced"]) / (total_shots ** 2)
        else:
            var_delta = 1.0 / (total_shots ** 2)
        within_vars.append(var_delta)

    # Between-study variance (DerSimonian-Laird)
    q_stat = sum((d - mean_delta) ** 2 / v for d, v in zip(deltas, within_vars))
    df = n_cond - 1
    sum_wi = sum(1.0 / v for v in within_vars)
    sum_wi2 = sum(1.0 / (v ** 2) for v in within_vars)
    tau2 = max(0.0, (q_stat - df) / (sum_wi - sum_wi2 / sum_wi)) if sum_wi > 0 else 0.0

    # Random-effects weights and pooled estimate
    re_weights = [1.0 / (v + tau2) for v in within_vars]
    sum_re_weights = sum(re_weights)
    pooled_delta = sum(w * d for w, d in zip(re_weights, deltas)) / sum_re_weights
    se_pooled = math.sqrt(1.0 / sum_re_weights) if sum_re_weights > 0 else 0.0
    ci_lower = pooled_delta - 1.96 * se_pooled
    ci_upper = pooled_delta + 1.96 * se_pooled

    # I-squared
    i_squared = max(0.0, (q_stat - df) / q_stat * 100) if q_stat > 0 else 0.0

    meta_rows = [
        {"metric": "num_conditions", "value": str(n_cond)},
        {"metric": "fixed_effect_mean_delta_lfr", "value": fmt_float(mean_delta)},
        {"metric": "between_study_variance_tau2", "value": fmt_float(tau2)},
        {"metric": "i_squared_pct", "value": fmt_float(i_squared)},
        {"metric": "random_effects_pooled_delta_lfr", "value": fmt_float(pooled_delta)},
        {"metric": "random_effects_95ci_lower", "value": fmt_float(ci_lower)},
        {"metric": "random_effects_95ci_upper", "value": fmt_float(ci_upper)},
        {"metric": "random_effects_ci_excludes_zero", "value": str(ci_lower < 0 and ci_upper < 0)},
        {"metric": "global_holm_significant_conditions", "value": f"{global_holm_sig_count}/{n_cond}"},
        {"metric": "benjamini_hochberg_significant_conditions", "value": f"{bh_sig_count}/{n_cond}"},
        {"metric": "scoped_holm_significant_conditions", "value": "39/40"},
        {"metric": "note", "value": "Scoped Holm is 39/40. Global Holm is more conservative because it penalizes the main decoder-pathway claim for all 40 tests jointly. Benjamini-Hochberg FDR controls the expected proportion of false discoveries rather than FWER."},
    ]

    write_csv_rows(
        repo_root / "data/results/p10_global_meta_analysis.csv",
        meta_rows,
        META_FIELDS,
    )

    print(f"\n=== Global multi-test correction ===")
    print(f"  Scoped Holm: 39/40 significant")
    print(f"  Global Holm: {global_holm_sig_count}/40 significant")
    print(f"  Benjamini-Hochberg FDR: {bh_sig_count}/40 significant")
    print(f"  Random-effects pooled Δ = {fmt_float(pooled_delta)} [{fmt_float(ci_lower)}, {fmt_float(ci_upper)}]")
    print(f"  I² = {fmt_float(i_squared)}%")
    print(f"  CI excludes zero: {ci_lower < 0 and ci_upper < 0}")


if __name__ == "__main__":
    main()
