"""Utilities for learning budgeted ShadowFluid reference dictionaries.

This module keeps the learning problem narrow and reusable:

- build a candidate reference pool for the bra-side dictionary index set R
- score custom R choices with the existing ShadowFluid rollout engine
- enumerate oracle subsets on small budgets
- expose candidate/sample features for lightweight ML selectors

The learned selector does not replace the projected Heisenberg dynamics. It
chooses which reference modes to retain before the reduced rollout.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json

import numpy as np

from shiftflow import cases, core_v0, core_v1


@dataclass(frozen=True)
class ReferenceEval:
    """Aggregated quality metrics for one custom reference-set choice."""

    reference_text: str
    objective: float
    mean_err_rho_vs_full: float
    mean_err_E_LP: float
    mean_err_Z_frob: float
    mean_leakage: float
    mean_bound: float
    mean_err_rho_lp_vs_full: float
    n_rollouts: int


def reference_set_to_text(R_flat: np.ndarray) -> str:
    """Serialize a reference set as a compact CSV-like string."""
    arr = core_v1.canonicalize_reference_set(R_flat)
    return ",".join(str(int(x)) for x in arr.tolist())


def reference_set_from_text(text: str) -> np.ndarray:
    """Inverse of reference_set_to_text()."""
    vals = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    return core_v1.canonicalize_reference_set(np.asarray(vals, dtype=int))


def components_to_json(components: list[core_v1.FourierPotential]) -> str:
    """Serialize a list of FourierPotential entries to JSON."""
    payload = [
        {"alpha": float(c.alpha), "qx": int(c.qx), "qy": int(c.qy)}
        for c in components
    ]
    return json.dumps(payload, separators=(",", ":"))


def components_from_json(text: str) -> list[core_v1.FourierPotential]:
    """Parse a FourierPotential list from components_to_json()."""
    payload = json.loads(str(text))
    return [
        core_v1.FourierPotential(
            alpha=float(item["alpha"]),
            qx=int(item["qx"]),
            qy=int(item["qy"]),
        )
        for item in payload
    ]


def anchor_flat_from_r0(r0: tuple[int, int], N: int) -> int:
    """Map an (iy, ix) anchor to the flattened Fourier-grid index."""
    return int((int(r0[0]) % N) * N + (int(r0[1]) % N))


def flat_to_k_coords(flat_idx: int, N: int) -> tuple[int, int]:
    """Return centered Fourier coordinates (kx, ky) for one flat index."""
    iy, ix = divmod(int(flat_idx), int(N))
    k = np.fft.fftfreq(N) * N
    return int(k[ix]), int(k[iy])


def candidate_direct_coupling_strength(
    flat_idx: int,
    components: list[core_v1.FourierPotential],
    N: int,
    *,
    r0: tuple[int, int] = (0, 0),
) -> float:
    """Total direct coupling magnitude between the anchor and one candidate."""
    anchor = anchor_flat_from_r0(r0, N)
    if int(flat_idx) == anchor:
        return 0.0

    iy0, ix0 = divmod(anchor, int(N))
    iy, ix = divmod(int(flat_idx), int(N))

    total = 0.0
    for comp in components:
        for sy, sx in [(comp.qy, comp.qx), (-comp.qy, -comp.qx)]:
            if ((iy0 + sy) % N == iy) and ((ix0 + sx) % N == ix):
                total += abs(float(comp.alpha)) / 2.0
    return float(total)


def build_candidate_reference_pool(
    *,
    N: int,
    components: list[core_v1.FourierPotential],
    r0: tuple[int, int] = (0, 0),
    max_hops: int = 1,
    max_candidates: int | None = None,
) -> np.ndarray:
    """Build and optionally clip a candidate reference pool for selector learning."""
    pool = core_v1.build_R_closure(r0, components, N, max_hops=max_hops)
    pool = core_v1.canonicalize_reference_set(pool)

    if max_candidates is None or len(pool) <= int(max_candidates):
        return pool

    keep = max(int(max_candidates), 1)
    anchor = anchor_flat_from_r0(r0, N)
    remaining = [int(x) for x in pool.tolist() if int(x) != anchor]
    remaining.sort(
        key=lambda flat: (
            -candidate_direct_coupling_strength(flat, components, N, r0=r0),
            abs(flat_to_k_coords(flat, N)[0]) + abs(flat_to_k_coords(flat, N)[1]),
            flat,
        )
    )
    clipped = np.asarray([anchor, *remaining[: max(keep - 1, 0)]], dtype=int)
    return core_v1.canonicalize_reference_set(clipped)


def make_budgeted_reference_set(
    pool_flat: np.ndarray,
    budget: int,
    *,
    anchor_flat: int,
    scores: np.ndarray,
) -> np.ndarray:
    """Pick `budget` reference modes from a pool using per-candidate scores."""
    pool = core_v1.canonicalize_reference_set(pool_flat)
    budget_i = max(1, min(int(budget), len(pool)))

    other = [int(x) for x in pool.tolist() if int(x) != int(anchor_flat)]
    if budget_i == 1 or not other:
        return np.asarray([int(anchor_flat)], dtype=int)

    if len(scores) != len(other):
        raise ValueError("scores must align with the non-anchor candidate list")

    ranked = sorted(
        zip(other, np.asarray(scores, dtype=float).tolist()),
        key=lambda item: (-float(item[1]), item[0]),
    )
    chosen = [int(anchor_flat)]
    chosen.extend(int(flat) for flat, _score in ranked[: budget_i - 1])
    return core_v1.canonicalize_reference_set(np.asarray(chosen, dtype=int))


def heuristic_reference_set(
    pool_flat: np.ndarray,
    budget: int,
    *,
    components: list[core_v1.FourierPotential],
    N: int,
    anchor_flat: int,
    strategy: str,
    rng_seed: int = 0,
) -> np.ndarray:
    """Select a budgeted reference set using a simple hand-crafted heuristic."""
    other = [int(x) for x in core_v1.canonicalize_reference_set(pool_flat).tolist() if int(x) != int(anchor_flat)]

    if not other:
        return np.asarray([int(anchor_flat)], dtype=int)

    if strategy == "coupling_greedy":
        scores = np.asarray(
            [
                candidate_direct_coupling_strength(flat, components, N, r0=divmod(anchor_flat, N))
                for flat in other
            ],
            dtype=float,
        )
    elif strategy == "low_energy":
        scores = np.asarray(
            [
                -(
                    flat_to_k_coords(flat, N)[0] ** 2
                    + flat_to_k_coords(flat, N)[1] ** 2
                )
                for flat in other
            ],
            dtype=float,
        )
    elif strategy == "random":
        rng = np.random.default_rng(int(rng_seed))
        scores = rng.random(len(other))
    else:
        raise ValueError(f"Unknown heuristic strategy: {strategy}")

    return make_budgeted_reference_set(
        pool_flat=np.asarray([anchor_flat, *other], dtype=int),
        budget=budget,
        anchor_flat=anchor_flat,
        scores=scores,
    )


def enumerate_reference_subsets(
    pool_flat: np.ndarray,
    budget: int,
    *,
    anchor_flat: int,
) -> list[np.ndarray]:
    """Enumerate all anchor-preserving subsets of a fixed budget."""
    pool = core_v1.canonicalize_reference_set(pool_flat)
    budget_i = max(1, min(int(budget), len(pool)))
    other = [int(x) for x in pool.tolist() if int(x) != int(anchor_flat)]

    if budget_i == 1 or not other:
        return [np.asarray([int(anchor_flat)], dtype=int)]

    subsets: list[np.ndarray] = []
    for combo in combinations(other, budget_i - 1):
        arr = np.asarray([int(anchor_flat), *combo], dtype=int)
        subsets.append(core_v1.canonicalize_reference_set(arr))
    return subsets


def evaluate_reference_set(
    *,
    nx: int,
    K0: float,
    components: list[core_v1.FourierPotential],
    R_flat: np.ndarray,
    seeds: list[int],
    times: list[float],
    density_weight: float = 1.0,
    leakage_weight: float = 0.1,
    task_weight: float = 0.0,
    z_weight: float = 0.0,
    H_dense: np.ndarray | None = None,
    eig: tuple[np.ndarray, np.ndarray] | None = None,
) -> ReferenceEval:
    """Score one custom reference set by averaging over seeds and times."""
    N = 2 ** int(nx)
    H_dense_local = H_dense if H_dense is not None else core_v1.build_H_dense(N, components)
    eig_local = eig if eig is not None else core_v1.eigendecompose(H_dense_local)

    rho_vals: list[float] = []
    task_vals: list[float] = []
    z_vals: list[float] = []
    leak_vals: list[float] = []
    bound_vals: list[float] = []
    lp_gap_vals: list[float] = []

    for seed in seeds:
        psi1_0, psi2_0, _stacked, _meta = cases.vortex_case(nx=nx, seed=int(seed))
        for t in times:
            result = core_v1.run_single_with_reference_set(
                N=N,
                components=components,
                K0=K0,
                t=float(t),
                psi1_0=psi1_0,
                psi2_0=psi2_0,
                R_flat=R_flat,
                H_dense=H_dense_local,
                eig=eig_local,
                use_qiskit=False,
            )
            rho_vals.append(float(result.err_rho_vs_full))
            task_vals.append(float(result.err_E_LP))
            z_vals.append(float(result.err_Z_frob))
            leak_vals.append(float(result.leakage_apriori))
            bound_vals.append(float(result.bound_apriori))
            lp_gap_vals.append(float(result.err_rho_lp_vs_full))

    mean_rho = float(np.mean(rho_vals)) if rho_vals else float("nan")
    mean_task = float(np.mean(task_vals)) if task_vals else float("nan")
    mean_z = float(np.mean(z_vals)) if z_vals else float("nan")
    mean_leak = float(np.mean(leak_vals)) if leak_vals else float("nan")
    mean_bound = float(np.mean(bound_vals)) if bound_vals else float("nan")
    mean_lp_gap = float(np.mean(lp_gap_vals)) if lp_gap_vals else float("nan")

    objective = (
        float(density_weight) * mean_rho
        + float(leakage_weight) * mean_leak
        + float(task_weight) * mean_task
        + float(z_weight) * mean_z
    )
    return ReferenceEval(
        reference_text=reference_set_to_text(R_flat),
        objective=float(objective),
        mean_err_rho_vs_full=mean_rho,
        mean_err_E_LP=mean_task,
        mean_err_Z_frob=mean_z,
        mean_leakage=mean_leak,
        mean_bound=mean_bound,
        mean_err_rho_lp_vs_full=mean_lp_gap,
        n_rollouts=int(len(rho_vals)),
    )


def find_oracle_reference_set(
    *,
    nx: int,
    K0: float,
    components: list[core_v1.FourierPotential],
    pool_flat: np.ndarray,
    budget: int,
    seeds: list[int],
    times: list[float],
    density_weight: float = 1.0,
    leakage_weight: float = 0.1,
    task_weight: float = 0.0,
    z_weight: float = 0.0,
    r0: tuple[int, int] = (0, 0),
    H_dense: np.ndarray | None = None,
    eig: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, ReferenceEval, list[ReferenceEval]]:
    """Exhaustively search a small candidate pool for the best reference set."""
    N = 2 ** int(nx)
    anchor = anchor_flat_from_r0(r0, N)
    H_dense_local = H_dense if H_dense is not None else core_v1.build_H_dense(N, components)
    eig_local = eig if eig is not None else core_v1.eigendecompose(H_dense_local)
    subsets = enumerate_reference_subsets(pool_flat, budget, anchor_flat=anchor)

    all_rows: list[ReferenceEval] = []
    best_row: ReferenceEval | None = None
    best_ref: np.ndarray | None = None

    for subset in subsets:
        row = evaluate_reference_set(
            nx=nx,
            K0=K0,
            components=components,
            R_flat=subset,
            seeds=seeds,
            times=times,
            density_weight=density_weight,
            leakage_weight=leakage_weight,
            task_weight=task_weight,
            z_weight=z_weight,
            H_dense=H_dense_local,
            eig=eig_local,
        )
        all_rows.append(row)
        if best_row is None or float(row.objective) < float(best_row.objective):
            best_row = row
            best_ref = subset

    if best_row is None or best_ref is None:
        raise RuntimeError("Oracle search produced no candidate subset")
    return best_ref, best_row, all_rows


def build_candidate_feature_rows(
    *,
    sample_id: int,
    nx: int,
    K0: float,
    budget: int,
    components: list[core_v1.FourierPotential],
    pool_flat: np.ndarray,
    selected_flat: np.ndarray,
    r0: tuple[int, int] = (0, 0),
) -> list[dict[str, object]]:
    """Build per-candidate feature rows for a budgeted selector dataset."""
    N = 2 ** int(nx)
    anchor = anchor_flat_from_r0(r0, N)
    pool = core_v1.canonicalize_reference_set(pool_flat)
    selected = set(core_v1.canonicalize_reference_set(selected_flat).tolist())

    H_dense = core_v1.build_H_dense(N, components)
    mask = core_v0.low_freq_mask(N, K0)
    K_flat = core_v1.mask_to_flat(mask, N)
    pool_set = set(pool.tolist())
    not_pool = np.asarray(sorted(set(range(N * N)) - pool_set), dtype=int)

    alphas = [abs(float(c.alpha)) for c in components]
    alpha_total = float(sum(alphas))
    alpha_l2 = float(np.sqrt(sum(a * a for a in alphas))) if alphas else 0.0
    max_alpha = float(max(alphas)) if alphas else 0.0
    mean_abs_q = float(
        np.mean([abs(int(c.qx)) + abs(int(c.qy)) for c in components])
    ) if components else 0.0

    rows: list[dict[str, object]] = []
    for rank, flat in enumerate(pool.tolist()):
        kx, ky = flat_to_k_coords(int(flat), N)
        single_leak = core_v1.leakage_apriori(
            H_dense=H_dense,
            K_flat=K_flat,
            R_flat=np.asarray([int(flat)], dtype=int),
        )
        pool_coupling = float(
            np.sum(np.abs(H_dense[int(flat), pool])) - abs(H_dense[int(flat), int(flat)])
        )
        outside_coupling = float(
            np.sum(np.abs(H_dense[int(flat), not_pool]))
        ) if len(not_pool) > 0 else 0.0
        task_coupling = float(np.mean(np.abs(H_dense[K_flat, int(flat)])))
        rows.append(
            {
                "sample_id": int(sample_id),
                "candidate_rank": int(rank),
                "flat_idx": int(flat),
                "is_anchor": int(int(flat) == int(anchor)),
                "label_selected": int(int(flat) in selected),
                "kx": int(kx),
                "ky": int(ky),
                "feat_abs_kx": float(abs(kx)),
                "feat_abs_ky": float(abs(ky)),
                "feat_radius2": float(kx * kx + ky * ky),
                "feat_is_task_mode": int(int(flat) in set(K_flat.tolist())),
                "feat_direct_coupling": float(
                    candidate_direct_coupling_strength(int(flat), components, N, r0=r0)
                ),
                "feat_pool_coupling": pool_coupling,
                "feat_outside_pool_coupling": outside_coupling,
                "feat_task_coupling_mean": task_coupling,
                "feat_single_ref_leakage": float(single_leak),
                "feat_alpha_total": alpha_total,
                "feat_alpha_l2": alpha_l2,
                "feat_alpha_max": max_alpha,
                "feat_mean_abs_q": mean_abs_q,
                "feat_num_components": int(len(components)),
                "feat_budget": int(budget),
                "feat_pool_size": int(len(pool)),
                "feat_task_size": int(len(K_flat)),
            }
        )
    return rows
