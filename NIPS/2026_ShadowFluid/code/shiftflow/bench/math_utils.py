"""Numerical helpers shared by data generation and analytical baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

from shiftflow import core_v0, core_v1, selector_learning


def psi_components_to_channels(psi1: np.ndarray, psi2: np.ndarray) -> np.ndarray:
    """Pack complex spinor components into 4 real channels."""
    return np.stack(
        [
            np.asarray(psi1.real, dtype=np.float32),
            np.asarray(psi1.imag, dtype=np.float32),
            np.asarray(psi2.real, dtype=np.float32),
            np.asarray(psi2.imag, dtype=np.float32),
        ],
        axis=0,
    )


def channels_to_psi_components(channels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of psi_components_to_channels()."""
    arr = np.asarray(channels, dtype=np.float64)
    psi1 = arr[0] + 1j * arr[1]
    psi2 = arr[2] + 1j * arr[3]
    return np.asarray(psi1, dtype=np.complex128), np.asarray(psi2, dtype=np.complex128)


def density_to_channels(rho: np.ndarray) -> np.ndarray:
    return np.asarray(rho[None, :, :], dtype=np.float32)


def build_spatial_grid(N: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 2.0 * np.pi, int(N), endpoint=False)
    X, Y = np.meshgrid(x, x)
    return X, Y


def potential_field_from_components(
    N: int, components: list[core_v1.FourierPotential]
) -> np.ndarray:
    """Evaluate the potential field on the periodic spatial grid."""
    X, Y = build_spatial_grid(N)
    V = np.zeros((N, N), dtype=np.float64)
    for comp in components:
        V += float(comp.alpha) * np.cos(float(comp.qx) * X + float(comp.qy) * Y)
    return V


def components_cache_key(
    components: list[core_v1.FourierPotential],
) -> tuple[tuple[float, int, int], ...]:
    """Stable hashable representation for one potential specification."""
    return tuple(
        (
            float(comp.alpha),
            int(comp.qx),
            int(comp.qy),
        )
        for comp in components
    )


def components_to_param_array(
    components: list[core_v1.FourierPotential],
    max_modes: int,
) -> np.ndarray:
    """Pad a potential specification into a fixed-size array."""
    out = np.zeros((int(max_modes), 3), dtype=np.float32)
    for idx, comp in enumerate(components[: int(max_modes)]):
        out[idx, 0] = float(comp.alpha)
        out[idx, 1] = float(comp.qx)
        out[idx, 2] = float(comp.qy)
    return out


def param_array_to_components(params: np.ndarray) -> list[core_v1.FourierPotential]:
    """Inverse of components_to_param_array(), ignoring zero-alpha rows."""
    arr = np.asarray(params, dtype=np.float64).reshape(-1, 3)
    out: list[core_v1.FourierPotential] = []
    for alpha, qx, qy in arr:
        if abs(float(alpha)) < 1e-12:
            continue
        out.append(
            core_v1.FourierPotential(
                alpha=float(alpha),
                qx=int(round(float(qx))),
                qy=int(round(float(qy))),
            )
        )
    return out


def hamiltonian_feature_vector(
    components: list[core_v1.FourierPotential],
    max_modes: int,
) -> np.ndarray:
    """Compact Hamiltonian/potential descriptor for conditioned baselines."""
    params = components_to_param_array(components, max_modes=max_modes).reshape(-1)
    alphas = np.asarray([abs(float(c.alpha)) for c in components], dtype=np.float64)
    q_norms = np.asarray(
        [abs(int(c.qx)) + abs(int(c.qy)) for c in components],
        dtype=np.float64,
    )
    stats = np.asarray(
        [
            float(np.sum(alphas)) if alphas.size else 0.0,
            float(np.sqrt(np.sum(alphas**2))) if alphas.size else 0.0,
            float(np.max(alphas)) if alphas.size else 0.0,
            float(len(components)),
            float(np.mean(q_norms)) if q_norms.size else 0.0,
            float(np.max(q_norms)) if q_norms.size else 0.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([params.astype(np.float32), stats], axis=0)


def lowfreq_density_flat(rho: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Flatten low-frequency density Fourier coefficients as real/imag parts."""
    N = int(rho.shape[-1])
    coeffs = np.fft.fft2(rho) / N
    coeffs_keep = coeffs[mask]
    return np.concatenate(
        [
            np.asarray(coeffs_keep.real, dtype=np.float32),
            np.asarray(coeffs_keep.imag, dtype=np.float32),
        ],
        axis=0,
    )


def lowpass_energy_from_coeffs(
    b1_flat: np.ndarray,
    b2_flat: np.ndarray,
    mask: np.ndarray,
    N: int,
) -> float:
    b1 = np.asarray(b1_flat, dtype=np.complex128).reshape(N, N)
    b2 = np.asarray(b2_flat, dtype=np.complex128).reshape(N, N)
    return float(np.real(np.sum(np.abs(b1[mask]) ** 2 + np.abs(b2[mask]) ** 2)))


def truncated_energy_from_coeffs(
    b1_trunc: np.ndarray,
    b2_trunc: np.ndarray,
) -> float:
    """Energy in an already low-pass-truncated coefficient vector."""
    b1 = np.asarray(b1_trunc, dtype=np.complex128).reshape(-1)
    b2 = np.asarray(b2_trunc, dtype=np.complex128).reshape(-1)
    return float(np.real(np.sum(np.abs(b1) ** 2 + np.abs(b2) ** 2)))


@dataclass(frozen=True)
class FullTrajectory:
    psi_channels: np.ndarray
    rho: np.ndarray
    lowfreq: np.ndarray
    energy: np.ndarray


def exact_rollout(
    psi1_0: np.ndarray,
    psi2_0: np.ndarray,
    components: list[core_v1.FourierPotential],
    time_grid: list[float],
    K0: float,
    *,
    precomputed: dict[str, Any] | None = None,
) -> FullTrajectory:
    """Compute exact full-state trajectories for one initial condition."""
    N = int(psi1_0.shape[0])
    if precomputed is None:
        precomputed = {}
    backend = str(precomputed.get("exact_backend", "eigh"))
    mask = precomputed.get("mask")
    if mask is None:
        mask = core_v0.low_freq_mask(N, K0)
    H_dense = precomputed.get("H_dense")
    if H_dense is None:
        H_dense = core_v1.build_H_dense(N, components)

    b1_0 = core_v0.unitary_fft2(psi1_0).reshape(-1)
    b2_0 = core_v0.unitary_fft2(psi2_0).reshape(-1)
    if backend == "expm_multiply":
        H_sparse = precomputed.get("H_sparse")
        if H_sparse is None:
            H_sparse = csr_matrix(H_dense)
    else:
        eig_vals = precomputed.get("eig_vals")
        eig_vecs = precomputed.get("eig_vecs")
        if eig_vals is None or eig_vecs is None:
            eig_vals, eig_vecs = core_v1.eigendecompose(H_dense)
        coeffs1 = eig_vecs.conj().T @ b1_0
        coeffs2 = eig_vecs.conj().T @ b2_0

    psi_list: list[np.ndarray] = []
    rho_list: list[np.ndarray] = []
    lowfreq_list: list[np.ndarray] = []
    energy_list: list[float] = []

    for t in time_grid:
        if backend == "expm_multiply":
            A_t = (-1j * float(t)) * H_sparse
            b1_t = np.asarray(expm_multiply(A_t, b1_0), dtype=np.complex128)
            b2_t = np.asarray(expm_multiply(A_t, b2_0), dtype=np.complex128)
        else:
            phase = np.exp(-1j * eig_vals * float(t))
            b1_t = eig_vecs @ (coeffs1 * phase)
            b2_t = eig_vecs @ (coeffs2 * phase)
        psi1_t = core_v0.unitary_ifft2(b1_t.reshape(N, N))
        psi2_t = core_v0.unitary_ifft2(b2_t.reshape(N, N))
        rho_t = core_v0.density_from_components(psi1_t, psi2_t)

        psi_list.append(psi_components_to_channels(psi1_t, psi2_t))
        rho_list.append(density_to_channels(rho_t))
        lowfreq_list.append(lowfreq_density_flat(rho_t, mask))
        energy_list.append(lowpass_energy_from_coeffs(b1_t, b2_t, mask, N))

    return FullTrajectory(
        psi_channels=np.stack(psi_list, axis=0),
        rho=np.stack(rho_list, axis=0),
        lowfreq=np.stack(lowfreq_list, axis=0),
        energy=np.asarray(energy_list, dtype=np.float32),
    )


@dataclass(frozen=True)
class ShadowPoolTrajectory:
    features: np.ndarray
    candidate_mask: np.ndarray
    candidate_leakage: np.ndarray
    candidate_flat: np.ndarray
    outside_pool_bra_leakage: np.ndarray
    pair_bra_coupling_sq: np.ndarray


def build_shadow_pool_trajectory(
    psi1_0: np.ndarray,
    psi2_0: np.ndarray,
    components: list[core_v1.FourierPotential],
    time_grid: list[float],
    K0: float,
    *,
    max_hops: int,
    max_candidates: int,
    r0: tuple[int, int] = (0, 0),
    precomputed: dict[str, Any] | None = None,
) -> ShadowPoolTrajectory:
    """Projected-Heisenberg candidate-pool trajectory for NSO-style models."""
    N = int(psi1_0.shape[0])
    if precomputed is None:
        precomputed = {}
    mask = precomputed.get("mask")
    if mask is None:
        mask = core_v0.low_freq_mask(N, K0)
    K_flat = precomputed.get("K_flat")
    if K_flat is None:
        K_flat = core_v1.mask_to_flat(mask, N)

    pool_flat = precomputed.get("pool_flat")
    if pool_flat is None:
        pool_flat = selector_learning.build_candidate_reference_pool(
            N=N,
            components=components,
            r0=r0,
            max_hops=max_hops,
            max_candidates=max_candidates,
        )
    P = int(max_candidates)
    M_K = int(len(K_flat))
    F_shadow = int(4 * M_K)

    candidate_mask = np.zeros(P, dtype=np.float32)
    candidate_leakage = np.zeros(P, dtype=np.float32)
    candidate_flat = np.full(P, -1, dtype=np.int64)
    features = np.zeros((len(time_grid), P, F_shadow), dtype=np.float32)
    outside_pool_bra_leakage = np.zeros(P, dtype=np.float32)
    pair_bra_coupling_sq = np.zeros((P, P), dtype=np.float32)

    pool_len = min(len(pool_flat), P)
    candidate_mask[:pool_len] = 1.0
    candidate_flat[:pool_len] = pool_flat[:pool_len]

    H_dense = precomputed.get("H_dense")
    if H_dense is None:
        H_dense = core_v1.build_H_dense(N, components)
    H_K = precomputed.get("H_K")
    if H_K is None:
        H_K = core_v1.extract_submatrix(H_dense, K_flat)
    H_R = precomputed.get("H_R")
    if H_R is None:
        H_R = core_v1.extract_submatrix(H_dense, pool_flat[:pool_len])
    pool_set = set(int(x) for x in pool_flat[:pool_len].tolist())
    not_pool = np.asarray(sorted(set(range(N * N)) - pool_set), dtype=int)
    b1_0 = core_v0.unitary_fft2(psi1_0).reshape(-1)
    b2_0 = core_v0.unitary_fft2(psi2_0).reshape(-1)
    b1_K_0 = b1_0[K_flat]
    b2_K_0 = b2_0[K_flat]
    b1_R_0 = b1_0[pool_flat[:pool_len]]
    b2_R_0 = b2_0[pool_flat[:pool_len]]

    pre_leak = precomputed.get("candidate_leakage")
    if pre_leak is not None:
        candidate_leakage[:pool_len] = np.asarray(pre_leak[:pool_len], dtype=np.float32)
    else:
        for idx, flat in enumerate(pool_flat[:pool_len]):
            candidate_leakage[idx] = float(
                core_v1.leakage_apriori(
                    H_dense=H_dense,
                    K_flat=K_flat,
                    R_flat=np.asarray([int(flat)], dtype=int),
                )
            )
    pre_outside = precomputed.get("outside_pool_bra_leakage")
    if pre_outside is not None:
        outside_pool_bra_leakage[:pool_len] = np.asarray(pre_outside[:pool_len], dtype=np.float32)
    else:
        for idx, flat in enumerate(pool_flat[:pool_len]):
            if len(not_pool) > 0:
                outside_pool_bra_leakage[idx] = float(
                    np.sum(np.abs(H_dense[np.ix_(not_pool, np.asarray([int(flat)], dtype=int))]) ** 2)
                )
            else:
                outside_pool_bra_leakage[idx] = 0.0
    pre_pair = precomputed.get("pair_bra_coupling_sq")
    if pre_pair is not None:
        pair_bra_coupling_sq[:pool_len, :pool_len] = np.asarray(
            pre_pair[:pool_len, :pool_len],
            dtype=np.float32,
        )
    else:
        pair_sq = np.abs(H_R) ** 2
        np.fill_diagonal(pair_sq, 0.0)
        pair_bra_coupling_sq[:pool_len, :pool_len] = pair_sq.astype(np.float32, copy=False)

    U_K_cache = precomputed.get("U_K_cache")
    if U_K_cache is None:
        U_K_cache = {float(t): expm(-1j * H_K * float(t)) for t in time_grid}
    U_R_cache = precomputed.get("U_R_cache")
    if U_R_cache is None:
        U_R_cache = {float(t): expm(-1j * H_R * float(t)) for t in time_grid}

    for t_idx, t in enumerate(time_grid):
        b1_K_t = U_K_cache[float(t)] @ b1_K_0
        b2_K_t = U_K_cache[float(t)] @ b2_K_0
        b1_R_t = U_R_cache[float(t)] @ b1_R_0
        b2_R_t = U_R_cache[float(t)] @ b2_R_0

        z1 = np.outer(b1_K_t, np.conj(b1_R_t))
        z2 = np.outer(b2_K_t, np.conj(b2_R_t))
        feat = np.concatenate(
            [
                z1.real.T,
                z1.imag.T,
                z2.real.T,
                z2.imag.T,
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        features[t_idx, :pool_len, :] = feat

    return ShadowPoolTrajectory(
        features=features,
        candidate_mask=candidate_mask,
        candidate_leakage=candidate_leakage,
        candidate_flat=candidate_flat,
        outside_pool_bra_leakage=outside_pool_bra_leakage,
        pair_bra_coupling_sq=pair_bra_coupling_sq,
    )


@dataclass(frozen=True)
class FixedShadowRollout:
    rho: np.ndarray
    lowfreq: np.ndarray
    energy: np.ndarray
    leakage: float


def fixed_shadow_rollout(
    psi1_0: np.ndarray,
    psi2_0: np.ndarray,
    components: list[core_v1.FourierPotential],
    time_grid: list[float],
    K0: float,
    *,
    reference_mode: str = "bfs",
    budget: int | None = None,
    r0: tuple[int, int] = (0, 0),
    R_hops: int = 1,
    pool_hops: int = 2,
    pool_max_candidates: int = 8,
) -> FixedShadowRollout:
    """Roll out the hand-crafted ShadowFluid baseline over a time grid."""
    N = int(psi1_0.shape[0])
    mask = core_v0.low_freq_mask(N, K0)
    K_flat = core_v1.mask_to_flat(mask, N)
    H_dense = core_v1.build_H_dense(N, components)

    if reference_mode == "bfs" and budget is None:
        R_flat = core_v1.build_R_closure(r0, components, N, max_hops=R_hops)
    else:
        pool = selector_learning.build_candidate_reference_pool(
            N=N,
            components=components,
            r0=r0,
            max_hops=pool_hops,
            max_candidates=pool_max_candidates,
        )
        anchor = selector_learning.anchor_flat_from_r0(r0, N)
        if budget is None:
            budget = len(pool)
        strategy = "coupling_greedy" if reference_mode != "random" else "random"
        R_flat = selector_learning.heuristic_reference_set(
            pool_flat=pool,
            budget=int(budget),
            components=components,
            N=N,
            anchor_flat=anchor,
            strategy=strategy,
            rng_seed=0,
        )

    H_K = core_v1.extract_submatrix(H_dense, K_flat)
    H_R = core_v1.extract_submatrix(H_dense, R_flat)

    b1_0 = core_v0.unitary_fft2(psi1_0).reshape(-1)
    b2_0 = core_v0.unitary_fft2(psi2_0).reshape(-1)
    b1_K_0 = b1_0[K_flat]
    b2_K_0 = b2_0[K_flat]

    rho_pred: list[np.ndarray] = []
    lowfreq_pred: list[np.ndarray] = []
    energy_pred: list[float] = []

    U_K_cache = {float(t): expm(-1j * H_K * float(t)) for t in time_grid}
    for t in time_grid:
        b1_K_t = U_K_cache[float(t)] @ b1_K_0
        b2_K_t = U_K_cache[float(t)] @ b2_K_0
        psi1_lp, psi2_lp = core_v1.reconstruct_lowpass(b1_K_t, b2_K_t, K_flat, N)
        rho_t = core_v0.density_from_components(psi1_lp, psi2_lp)
        rho_pred.append(density_to_channels(rho_t))
        lowfreq_pred.append(lowfreq_density_flat(rho_t, mask))
        energy_pred.append(truncated_energy_from_coeffs(b1_K_t, b2_K_t))

    leakage = float(core_v1.leakage_apriori(H_dense=H_dense, K_flat=K_flat, R_flat=R_flat))
    return FixedShadowRollout(
        rho=np.stack(rho_pred, axis=0),
        lowfreq=np.stack(lowfreq_pred, axis=0),
        energy=np.asarray(energy_pred, dtype=np.float32),
        leakage=leakage,
    )
