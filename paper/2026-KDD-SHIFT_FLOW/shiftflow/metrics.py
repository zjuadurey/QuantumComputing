"""shiftflow/metrics.py

Metrics and cost proxies for SHIFT-FLOW experiments.

All physics/field definitions must match the reference implementation in
`test/shadow_test_v4.py`. In particular, momentum is represented by the
probability current components (Jx, Jy) computed via periodic central
differences.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from shiftflow import core_v0


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    """Relative L2 error ||a-b|| / ||b||. Returns NaN if ||b|| == 0."""
    num = float(np.linalg.norm(a - b))
    den = float(np.linalg.norm(b))
    return num / den if den != 0.0 else float("nan")


def rel_l2_vec(a_x: np.ndarray, a_y: np.ndarray, b_x: np.ndarray, b_y: np.ndarray) -> float:
    """Relative L2 error for a 2D vector field (x,y components)."""
    num = math.sqrt(float(np.linalg.norm(a_x - b_x) ** 2 + np.linalg.norm(a_y - b_y) ** 2))
    den = math.sqrt(float(np.linalg.norm(b_x) ** 2 + np.linalg.norm(b_y) ** 2))
    return num / den if den != 0.0 else float("nan")


def err_rho_from_components(
    psi1_pred: np.ndarray,
    psi2_pred: np.ndarray,
    psi1_ref: np.ndarray,
    psi2_ref: np.ndarray,
) -> float:
    """err_rho = relL2(rho_pred, rho_ref), rho = |psi1|^2 + |psi2|^2."""
    rho_p = core_v0.density_from_components(psi1_pred, psi2_pred)
    rho_r = core_v0.density_from_components(psi1_ref, psi2_ref)
    return rel_l2(rho_p, rho_r)


def err_momentum_from_components(
    psi1_pred: np.ndarray,
    psi2_pred: np.ndarray,
    psi1_ref: np.ndarray,
    psi2_ref: np.ndarray,
    dx: float,
    dy: float,
) -> float:
    """err_momentum for v4-defined momentum/current fields.

    Uses J = Im(psi* grad psi) summed over both components.
    Returns relative L2 on the vector field (Jx, Jy).
    """
    Jx_p, Jy_p = core_v0.current_from_components(psi1_pred, psi2_pred, dx=dx, dy=dy)
    Jx_r, Jy_r = core_v0.current_from_components(psi1_ref, psi2_ref, dx=dx, dy=dy)
    return rel_l2_vec(Jx_p, Jy_p, Jx_r, Jy_r)


def err_omega_from_components(
    psi1_pred: np.ndarray,
    psi2_pred: np.ndarray,
    psi1_ref: np.ndarray,
    psi2_ref: np.ndarray,
    dx: float,
    dy: float,
) -> float:
    """Optional err_omega = relL2(omega_pred, omega_ref) using v4 definition."""
    omg_p = core_v0.vorticity_from_components(psi1_pred, psi2_pred, dx=dx, dy=dy)
    omg_r = core_v0.vorticity_from_components(psi1_ref, psi2_ref, dx=dx, dy=dy)
    return rel_l2(omg_p, omg_r)


def E_LP_from_coeffs(b1: np.ndarray, b2: np.ndarray, mask: np.ndarray) -> float:
    """Task-only low-frequency mass/power E_LP = sum_{mask} (|b1|^2 + |b2|^2).

    Assumes unitary FFT conventions (as in v4), but does not require any
    reconstruction.
    """
    m = mask.astype(bool)
    return float(np.sum(np.abs(b1[m]) ** 2 + np.abs(b2[m]) ** 2))


def E_LP_from_components(psi1: np.ndarray, psi2: np.ndarray, mask: np.ndarray) -> float:
    """Convenience wrapper computing E_LP from (psi1, psi2) via unitary FFT."""
    b1 = core_v0.unitary_fft2(psi1)
    b2 = core_v0.unitary_fft2(psi2)
    return E_LP_from_coeffs(b1, b2, mask)


def q_base(nx: int) -> int:
    """Baseline qubit proxy: x,y registers plus spin qubit."""
    return 2 * int(nx) + 1


def q_shift(M: int, include_spin: bool = True) -> int:
    """SHIFT-FLOW qubit proxy based on retained mode count M.

    We use q_shift = ceil(log2(M)) plus an optional +1 spin qubit.
    """
    M_i = int(M)
    if M_i <= 0:
        q_spatial = 0
    else:
        q_spatial = int(math.ceil(math.log2(M_i)))
    return q_spatial + (1 if include_spin else 0)


@dataclass(frozen=True)
class CostProxies:
    nx: int
    L: int
    M: int
    q_base: int
    q_shift: int
    measurement_proxy: float
    postprocess_task_proxy: float
    postprocess_full_proxy: float


def cost_proxies(nx: int, mask: np.ndarray, include_spin_shift: bool = True) -> CostProxies:
    """Compute common cost proxies given nx and a low-frequency mask."""
    nx_i = int(nx)
    L = 2**nx_i
    M = int(np.count_nonzero(mask))
    qb = q_base(nx_i)
    qs = q_shift(M, include_spin=include_spin_shift)

    # Simple KDD-style proxies (dimensionless)
    measurement_proxy = float(M)
    postprocess_task_proxy = float(M)
    postprocess_full_proxy = float(L * L)

    return CostProxies(
        nx=nx_i,
        L=L,
        M=M,
        q_base=qb,
        q_shift=qs,
        measurement_proxy=measurement_proxy,
        postprocess_task_proxy=postprocess_task_proxy,
        postprocess_full_proxy=postprocess_full_proxy,
    )
