"""Build a small supervised dataset for learned ShadowFluid dictionary selection.

Each sample corresponds to one Hamiltonian family plus a fixed task cutoff and
reference budget. We:

1. construct a clipped candidate reference pool
2. exhaustively search the pool for the oracle budgeted reference set
3. emit per-candidate feature rows and per-sample oracle metadata

This is the smallest useful stepping stone toward a learned dictionary selector.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shiftflow import core_v1, selector_learning as sl  # noqa: E402


DEFAULT_J_CHOICES = [2, 3, 4]
DEFAULT_EVAL_SEEDS = [0, 1]
DEFAULT_EVAL_TIMES = [0.3, 0.6]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def list_text_int(vals: list[int]) -> str:
    return ",".join(str(int(v)) for v in vals)


def list_text_float(vals: list[float]) -> str:
    return ",".join(f"{float(v):.6g}" for v in vals)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build a dataset for learned ShadowFluid reference selection")
    p.add_argument("--out-dir", default=str(ROOT / "results" / "shadow_selector"))
    p.add_argument("--num-samples", type=int, default=24)
    p.add_argument("--nx", type=int, default=4)
    p.add_argument("--K0", type=float, default=4.0)
    p.add_argument("--budget", type=int, default=4)
    p.add_argument("--pool-hops", type=int, default=1)
    p.add_argument("--pool-max-candidates", type=int, default=7)
    p.add_argument("--J-choices", default=",".join(map(str, DEFAULT_J_CHOICES)))
    p.add_argument("--alpha-scale", type=float, default=0.5)
    p.add_argument("--q-max", type=int, default=3)
    p.add_argument("--base-seed", type=int, default=20260505)
    p.add_argument("--eval-seeds", default=",".join(map(str, DEFAULT_EVAL_SEEDS)))
    p.add_argument("--eval-times", default=",".join(map(str, DEFAULT_EVAL_TIMES)))
    p.add_argument("--density-weight", type=float, default=1.0)
    p.add_argument("--leakage-weight", type=float, default=0.1)
    p.add_argument("--task-weight", type=float, default=0.0)
    p.add_argument("--z-weight", type=float, default=0.0)
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nx = int(args.nx)
    N = 2 ** nx
    K0 = float(args.K0)
    budget = int(args.budget)
    pool_hops = int(args.pool_hops)
    pool_max_candidates = int(args.pool_max_candidates)
    J_choices = parse_int_list(args.J_choices)
    eval_seeds = parse_int_list(args.eval_seeds)
    eval_times = parse_float_list(args.eval_times)

    sample_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []

    collected = 0
    attempt = 0
    max_attempts = max(int(args.num_samples) * 20, 50)
    while collected < int(args.num_samples):
        if attempt >= max_attempts:
            raise RuntimeError(
                "Could not collect enough valid selector samples. "
                "Try lowering --budget or raising --pool-max-candidates."
            )

        J = int(J_choices[attempt % len(J_choices)])
        components = core_v1.potential_multi_random(
            J=J,
            alpha_scale=float(args.alpha_scale),
            q_max=int(args.q_max),
            seed=int(args.base_seed) + attempt,
        )
        pool_flat = sl.build_candidate_reference_pool(
            N=N,
            components=components,
            r0=(0, 0),
            max_hops=pool_hops,
            max_candidates=pool_max_candidates,
        )
        if len(pool_flat) < budget:
            attempt += 1
            continue

        H_dense = core_v1.build_H_dense(N, components)
        eig = core_v1.eigendecompose(H_dense)

        oracle_ref, oracle_eval, _all_evals = sl.find_oracle_reference_set(
            nx=nx,
            K0=K0,
            components=components,
            pool_flat=pool_flat,
            budget=budget,
            seeds=eval_seeds,
            times=eval_times,
            density_weight=float(args.density_weight),
            leakage_weight=float(args.leakage_weight),
            task_weight=float(args.task_weight),
            z_weight=float(args.z_weight),
            r0=(0, 0),
            H_dense=H_dense,
            eig=eig,
        )

        anchor_flat = sl.anchor_flat_from_r0((0, 0), N)
        coupling_ref = sl.heuristic_reference_set(
            pool_flat=pool_flat,
            budget=budget,
            components=components,
            N=N,
            anchor_flat=anchor_flat,
            strategy="coupling_greedy",
        )
        coupling_eval = sl.evaluate_reference_set(
            nx=nx,
            K0=K0,
            components=components,
            R_flat=coupling_ref,
            seeds=eval_seeds,
            times=eval_times,
            density_weight=float(args.density_weight),
            leakage_weight=float(args.leakage_weight),
            task_weight=float(args.task_weight),
            z_weight=float(args.z_weight),
            H_dense=H_dense,
            eig=eig,
        )
        low_energy_ref = sl.heuristic_reference_set(
            pool_flat=pool_flat,
            budget=budget,
            components=components,
            N=N,
            anchor_flat=anchor_flat,
            strategy="low_energy",
        )
        low_energy_eval = sl.evaluate_reference_set(
            nx=nx,
            K0=K0,
            components=components,
            R_flat=low_energy_ref,
            seeds=eval_seeds,
            times=eval_times,
            density_weight=float(args.density_weight),
            leakage_weight=float(args.leakage_weight),
            task_weight=float(args.task_weight),
            z_weight=float(args.z_weight),
            H_dense=H_dense,
            eig=eig,
        )

        sample_id = collected
        sample_rows.append(
            {
                "sample_id": int(sample_id),
                "attempt_id": int(attempt),
                "nx": nx,
                "N": N,
                "K0": K0,
                "budget": budget,
                "pool_hops": pool_hops,
                "pool_size": int(len(pool_flat)),
                "pool_text": sl.reference_set_to_text(pool_flat),
                "components_json": sl.components_to_json(components),
                "components_label": core_v1.potential_label(components),
                "eval_seeds": list_text_int(eval_seeds),
                "eval_times": list_text_float(eval_times),
                "oracle_reference_text": sl.reference_set_to_text(oracle_ref),
                "oracle_objective": float(oracle_eval.objective),
                "oracle_err_rho_vs_full": float(oracle_eval.mean_err_rho_vs_full),
                "oracle_leakage": float(oracle_eval.mean_leakage),
                "coupling_reference_text": sl.reference_set_to_text(coupling_ref),
                "coupling_objective": float(coupling_eval.objective),
                "low_energy_reference_text": sl.reference_set_to_text(low_energy_ref),
                "low_energy_objective": float(low_energy_eval.objective),
            }
        )
        candidate_rows.extend(
            sl.build_candidate_feature_rows(
                sample_id=sample_id,
                nx=nx,
                K0=K0,
                budget=budget,
                components=components,
                pool_flat=pool_flat,
                selected_flat=oracle_ref,
                r0=(0, 0),
            )
        )

        collected += 1
        attempt += 1

    sample_path = out_dir / "samples.csv"
    candidate_path = out_dir / "candidates.csv"
    note_path = out_dir / "README.md"

    sample_fields = [
        "sample_id",
        "attempt_id",
        "nx",
        "N",
        "K0",
        "budget",
        "pool_hops",
        "pool_size",
        "pool_text",
        "components_json",
        "components_label",
        "eval_seeds",
        "eval_times",
        "oracle_reference_text",
        "oracle_objective",
        "oracle_err_rho_vs_full",
        "oracle_leakage",
        "coupling_reference_text",
        "coupling_objective",
        "low_energy_reference_text",
        "low_energy_objective",
    ]
    candidate_fields = [
        "sample_id",
        "candidate_rank",
        "flat_idx",
        "is_anchor",
        "label_selected",
        "kx",
        "ky",
        "feat_abs_kx",
        "feat_abs_ky",
        "feat_radius2",
        "feat_is_task_mode",
        "feat_direct_coupling",
        "feat_pool_coupling",
        "feat_outside_pool_coupling",
        "feat_task_coupling_mean",
        "feat_single_ref_leakage",
        "feat_alpha_total",
        "feat_alpha_l2",
        "feat_alpha_max",
        "feat_mean_abs_q",
        "feat_num_components",
        "feat_budget",
        "feat_pool_size",
        "feat_task_size",
    ]

    write_csv(sample_path, sample_rows, sample_fields)
    write_csv(candidate_path, candidate_rows, candidate_fields)

    note_path.write_text(
        "\n".join(
            [
                "# Shadow Selector Dataset",
                "",
                "This dataset is built from the existing ShadowFluid rollout engine.",
                "",
                "Each sample stores:",
                "- a random multi-component potential",
                "- a clipped candidate reference pool",
                "- an oracle budgeted reference set found by exhaustive search",
                "- per-candidate features for a lightweight learned selector",
                "",
                f"Samples: {len(sample_rows)}",
                f"Grid: N = {N}",
                f"Cutoff: K0 = {K0}",
                f"Budget: {budget}",
                f"Pool hops: {pool_hops}",
                f"Pool max candidates: {pool_max_candidates}",
                f"Eval seeds: {eval_seeds}",
                f"Eval times: {eval_times}",
                "",
                f"Files: {sample_path.name}, {candidate_path.name}",
            ]
        )
    )

    print(f"Wrote samples: {sample_path}")
    print(f"Wrote candidates: {candidate_path}")
    print(f"Wrote note: {note_path}")
    print(
        f"Built {len(sample_rows)} selector samples with "
        f"budget={budget}, pool_max_candidates={pool_max_candidates}, N={N}, K0={K0}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
