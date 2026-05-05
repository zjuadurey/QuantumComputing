"""shiftflow/cases.py

Deterministic initial-condition generator for SHIFT-FLOW sweeps.

We start from the vortex initial condition used in `test/shadow_test_v4.py` and
apply small, controlled, seed-deterministic variations:
- sigma jitter
- integer roll shift on the periodic grid
- optional tiny complex perturbation

All outputs are globally normalized as a single quantum state (stacked psi1/psi2
vector has L2 norm 1), matching v4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shiftflow import core_v0


def normalize_components(psi1: np.ndarray, psi2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Global quantum-state normalization across both components."""
    N = psi1.shape[0]
    stacked = np.concatenate([psi1.reshape(-1), psi2.reshape(-1)]).astype(np.complex128, copy=False)
    nrm = float(np.linalg.norm(stacked))
    if nrm == 0.0:
        raise ValueError("Cannot normalize: zero norm")
    stacked = stacked / nrm
    psi1_n = stacked[: N * N].reshape(N, N)
    psi2_n = stacked[N * N :].reshape(N, N)
    return psi1_n, psi2_n, stacked


@dataclass(frozen=True)
class CaseMeta:
    nx: int
    N: int
    seed: int
    sigma_base: float
    sigma_eff: float
    shift_x: int
    shift_y: int
    noise_eps: float


def vortex_case(
    nx: int,
    seed: int,
    *,
    sigma_base: float = 2.0,
    sigma_jitter: float = 0.05,
    shift_max: int = 2,
    noise_eps: float = 5e-4,
    canonical_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, CaseMeta]:
    """Generate a reproducible vortex IC with small, controlled variations.

    The `canonical_seed` (default 0) returns the unmodified base vortex IC.
    """
    nx_i = int(nx)
    N = 2**nx_i
    seed_i = int(seed)
    rng = np.random.default_rng(seed_i)

    if seed_i == int(canonical_seed):
        sigma_eff = float(sigma_base)
        shift_x = 0
        shift_y = 0
        noise_eff = 0.0
    else:
        # sigma jitter (keep positive)
        sigma_eff = float(sigma_base) * (1.0 + float(sigma_jitter) * float(rng.standard_normal()))
        sigma_eff = max(sigma_eff, 1e-3)

        # integer roll shifts on periodic grid
        sm = int(shift_max)
        shift_x = int(rng.integers(-sm, sm + 1)) if sm > 0 else 0
        shift_y = int(rng.integers(-sm, sm + 1)) if sm > 0 else 0

        # tiny perturbation strength
        noise_eff = float(noise_eps)

    psi1, psi2, _ = core_v0.vortex_initial_condition(N=N, sigma=sigma_eff)

    if shift_x != 0 or shift_y != 0:
        psi1 = np.roll(psi1, shift=(shift_y, shift_x), axis=(0, 1))
        psi2 = np.roll(psi2, shift=(shift_y, shift_x), axis=(0, 1))

    if noise_eff > 0.0:
        n1 = rng.standard_normal(size=psi1.shape) + 1j * rng.standard_normal(size=psi1.shape)
        n2 = rng.standard_normal(size=psi2.shape) + 1j * rng.standard_normal(size=psi2.shape)
        noise_stack = np.concatenate([n1.reshape(-1), n2.reshape(-1)]).astype(np.complex128, copy=False)
        noise_norm = float(np.linalg.norm(noise_stack))
        if noise_norm > 0.0:
            scale = noise_eff / noise_norm
            psi1 = psi1 + scale * n1
            psi2 = psi2 + scale * n2

    psi1_n, psi2_n, stacked = normalize_components(psi1, psi2)

    meta = CaseMeta(
        nx=nx_i,
        N=N,
        seed=seed_i,
        sigma_base=float(sigma_base),
        sigma_eff=float(sigma_eff),
        shift_x=int(shift_x),
        shift_y=int(shift_y),
        noise_eps=float(noise_eff),
    )
    return psi1_n, psi2_n, stacked, meta
