"""shiftflow/core_v0.py

Core math for SHIFT-FLOW (free evolution V=0) experiments.

Source of truth: `test/shadow_test_v4.py`.
This module refactors reusable functions out of that script without changing the
math. It also adds an FFT baseline evolution routine for fast sweeps.

Important:
- "Shadow" here means truncated-mode / shadow-observable dynamics (no classical
  shadow tomography).
"""

from __future__ import annotations

import numpy as np


# ============================================================
# 1) Optional Qiskit baseline (spot-check only)
# ============================================================


def kinetic_operator(n: int, dt: float):
    """Same kinetic operator as in `test/shadow_test_v4.py` / `test/circuit_2D.py`.

    This is only needed for optional Qiskit spot-checks.
    """
    try:
        from qiskit import QuantumCircuit
    except Exception as e:  # pragma: no cover
        raise ImportError("qiskit is required for kinetic_operator()") from e

    qc = QuantumCircuit(n)
    qc.rz(-2 ** (n - 1) * dt, n - 1)
    for i in range(n):
        qc.rz(2 ** (n - i - 2) * dt, n - i - 1)
    for i in range(1, n):
        qc.cx(n - 1, n - i - 1)
        qc.rz(-2 ** (2 * n - i - 2) * dt, n - i - 1)
        qc.cx(n - 1, n - i - 1)
    for i in range(n):
        for j in range(n):
            if i != j:
                qc.cx(n - i - 1, n - j - 1)
                qc.rz(2 ** (2 * n - i - j - 4) * dt, n - j - 1)
                qc.cx(n - i - 1, n - j - 1)
    return qc


def evolve_statevector_v0(nx: int, ny: int, t: float, initial_state: np.ndarray) -> np.ndarray:
    """Baseline full-state evolution for V=0 using Qiskit statevector simulation.

    initialize -> QFTx QFTy -> kinetic phase -> iQFTx iQFTy

    Returns the final statevector (length 2^(nx+ny+1)).
    """
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
        from qiskit.circuit.library import QFT
    except Exception as e:  # pragma: no cover
        raise ImportError("qiskit and qiskit-aer are required for evolve_statevector_v0()") from e

    q_num = nx + ny + 1
    circ = QuantumCircuit(q_num)

    # Initialize full state (2 components stacked: sigma is the MSB qubit)
    circ.initialize(initial_state)
    circ.barrier()

    # QFT on x and y registers
    circ.append(QFT(nx), range(nx))
    circ.append(QFT(ny), range(nx, nx + ny))
    circ.barrier()

    # kinetic phases (free evolution)
    circ.append(kinetic_operator(nx, t), range(nx))
    circ.append(kinetic_operator(ny, t), range(nx, nx + ny))
    circ.barrier()

    # inverse QFT back to position basis
    circ.append(QFT(nx).inverse(), range(nx))
    circ.append(QFT(ny).inverse(), range(nx, nx + ny))

    circ.save_state()

    simulator = AerSimulator(method="statevector")
    circ = transpile(circ, simulator)
    result = simulator.run(circ).result()
    return np.asarray(result.data(0)["statevector"], dtype=np.complex128)


def statevector_to_components(sv: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
    """Split a stacked statevector into (psi1, psi2) components with shape (N, N)."""
    tmp = sv.reshape(2, N, N)
    return tmp[0, :, :], tmp[1, :, :]


# ============================================================
# 2) Vortex initial condition (reference)
# ============================================================


def vortex_initial_condition(N: int, sigma: float = 3.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (psi1_0, psi2_0, initial_state_vector), globally normalized."""
    pi = np.pi
    x = np.linspace(-pi, pi, N, endpoint=False)
    y = np.linspace(-pi, pi, N, endpoint=False)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    f = np.exp(-(R / sigma) ** 4)
    u = 2 * (X + 1j * Y) * f / (1 + R**2)
    v = 1j * (R**2 + 1 - 2 * f) / (1 + R**2)

    psi1 = u / np.sqrt(np.abs(u) ** 2 + np.abs(v) ** 4)
    psi2 = v**2 / np.sqrt(np.abs(u) ** 2 + np.abs(v) ** 4)

    # Quantum-state global normalization (and sync back to psi1/psi2)
    stacked = np.array([psi1, psi2]).reshape(-1).astype(np.complex128)
    stacked /= np.linalg.norm(stacked)
    psi1_n = stacked[: N * N].reshape(N, N)
    psi2_n = stacked[N * N :].reshape(N, N)
    return psi1_n, psi2_n, stacked


# ============================================================
# 3) Fourier conventions + masks
# ============================================================


def k_grid(N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (kx, ky, KX, KY) with k = fftfreq(N) * N in integer units."""
    kx = np.fft.fftfreq(N) * N
    ky = np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(kx, ky)
    return kx, ky, KX, KY


def energy_grid_free(N: int) -> np.ndarray:
    """Free-particle energy grid E(k) = (kx^2 + ky^2)/2 in integer-k convention."""
    _, _, KX, KY = k_grid(N)
    return 0.5 * (KX**2 + KY**2)


def low_freq_mask(N: int, K0: float) -> np.ndarray:
    """Circular low-frequency set: keep modes with sqrt(kx^2+ky^2) <= K0."""
    _, _, KX, KY = k_grid(N)
    return (KX**2 + KY**2) <= (K0**2)


def unitary_fft2(field_xy: np.ndarray) -> np.ndarray:
    """Unitary 2D FFT consistent with QFT normalization: b = FFT2(field)/N."""
    N = field_xy.shape[0]
    return np.fft.fft2(field_xy) / N


def unitary_ifft2(field_k: np.ndarray) -> np.ndarray:
    """Inverse of unitary_fft2: field = IFFT2(b) * N."""
    N = field_k.shape[0]
    return np.fft.ifft2(field_k) * N


def lowpass_filter_coeffs(bk: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero out k-modes outside mask."""
    return np.where(mask, bk, 0.0)


def lowpass_filter_components(
    psi1: np.ndarray,
    psi2: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Low-pass filter the (psi1, psi2) fields by truncating Fourier coefficients."""
    b1 = unitary_fft2(psi1)
    b2 = unitary_fft2(psi2)
    b1_lp = lowpass_filter_coeffs(b1, mask)
    b2_lp = lowpass_filter_coeffs(b2, mask)
    return unitary_ifft2(b1_lp), unitary_ifft2(b2_lp)


def evolve_components_fft_v0(
    psi1_0: np.ndarray,
    psi2_0: np.ndarray,
    t: float,
    E: np.ndarray | None = None,
    return_coeffs: bool = False,
):
    """FFT baseline evolution for V=0.

    b(t) = b(0) * exp(-i E t), with unitary FFT conventions.
    """
    N = psi1_0.shape[0]
    E_arr = energy_grid_free(N) if E is None else E

    phase = np.exp(-1j * E_arr * t)
    b1_0 = unitary_fft2(psi1_0)
    b2_0 = unitary_fft2(psi2_0)
    b1_t = b1_0 * phase
    b2_t = b2_0 * phase
    psi1_t = unitary_ifft2(b1_t)
    psi2_t = unitary_ifft2(b2_t)
    if return_coeffs:
        return psi1_t, psi2_t, b1_t, b2_t
    return psi1_t, psi2_t


# ============================================================
# 4) Shadow evolution (coherences)
# ============================================================


def choose_reference_mode(
    bk: np.ndarray,
    mask: np.ndarray,
    prefer: tuple[int, int] = (0, 0),
    min_rel: float = 1e-3,
) -> tuple[int, int]:
    """Pick reference k0 index (iy, ix). Prefer (0,0) unless too small."""
    iy0, ix0 = prefer
    amp0 = np.abs(bk[iy0, ix0])
    amps = np.abs(bk[mask])
    max_amp = float(np.max(amps)) if amps.size else 0.0
    if max_amp == 0.0:
        return iy0, ix0
    if amp0 >= min_rel * max_amp:
        return iy0, ix0

    idx_flat = np.argmax(np.abs(bk) * mask)
    iy, ix = np.unravel_index(idx_flat, bk.shape)
    return int(iy), int(ix)


def shadow_evolve_lowpass_from_coherences(
    b0: np.ndarray,
    mask: np.ndarray,
    t: float,
    k0_idx: tuple[int, int],
    E: np.ndarray,
) -> np.ndarray:
    """Shadow-style evolution of low-frequency coherences relative to k0.

    Returns b_shadow(k,t) with zeros outside mask.
    """
    iy0, ix0 = k0_idx
    E0 = float(E[iy0, ix0])

    b_k0_0 = b0[iy0, ix0]
    z0 = b0 * np.conj(b_k0_0)
    phase = np.exp(-1j * (E - E0) * t)
    zt = z0 * phase

    b_k0_t = b_k0_0 * np.exp(-1j * E0 * t)

    b_t = np.zeros_like(b0)
    b_t[mask] = zt[mask] / np.conj(b_k0_t)
    b_t[iy0, ix0] = b_k0_t
    return b_t


def shadow_evolve_components_lowpass_coherences(
    psi1_0: np.ndarray,
    psi2_0: np.ndarray,
    mask: np.ndarray,
    t: float,
    E: np.ndarray | None = None,
    prefer_k0: tuple[int, int] = (0, 0),
    min_rel: float = 1e-3,
    return_coeffs: bool = False,
):
    """Convenience wrapper: evolve shadow low-pass (psi1, psi2) from coherences."""
    N = psi1_0.shape[0]
    E_arr = energy_grid_free(N) if E is None else E

    b1_0 = unitary_fft2(psi1_0)
    b2_0 = unitary_fft2(psi2_0)
    k0_1 = choose_reference_mode(b1_0, mask, prefer=prefer_k0, min_rel=min_rel)
    k0_2 = choose_reference_mode(b2_0, mask, prefer=prefer_k0, min_rel=min_rel)

    b1_shadow = shadow_evolve_lowpass_from_coherences(b0=b1_0, mask=mask, t=t, k0_idx=k0_1, E=E_arr)
    b2_shadow = shadow_evolve_lowpass_from_coherences(b0=b2_0, mask=mask, t=t, k0_idx=k0_2, E=E_arr)
    psi1_shadow = unitary_ifft2(b1_shadow)
    psi2_shadow = unitary_ifft2(b2_shadow)
    if return_coeffs:
        return psi1_shadow, psi2_shadow, b1_shadow, b2_shadow, k0_1, k0_2
    return psi1_shadow, psi2_shadow, k0_1, k0_2


def shadow_evolve_components_lowpass_qiskit_v0(
    psi1_0: np.ndarray,
    psi2_0: np.ndarray,
    mask: np.ndarray,
    t: float,
    E: np.ndarray | None = None,
    order: str = "energy",
):
    """Qiskit shadow evolution for V=0 on a compressed mode register.

    This is the "real" quantum implementation path used by experiments.
    """
    from shiftflow import qiskit_shadow_v0

    psi1_t, psi2_t, b1_t, b2_t, _enc = qiskit_shadow_v0.shadow_evolve_components_qiskit_v0(
        psi1_0,
        psi2_0,
        mask=mask,
        t=t,
        E=E,
        order=order,
    )
    return psi1_t, psi2_t, b1_t, b2_t


# ============================================================
# 5) Diagnostics: density, current (momentum), vorticity
# ============================================================


def density_from_components(psi1: np.ndarray, psi2: np.ndarray) -> np.ndarray:
    return np.abs(psi1) ** 2 + np.abs(psi2) ** 2


def _ddx_periodic(f: np.ndarray, dx: float) -> np.ndarray:
    """Periodic central difference d/dx (x is axis=1)."""
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * dx)


def _ddy_periodic(f: np.ndarray, dy: float) -> np.ndarray:
    """Periodic central difference d/dy (y is axis=0)."""
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * dy)


def current_from_components(
    psi1: np.ndarray,
    psi2: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute probability current J = Im(psi* grad psi), summed over both components."""
    dpsi1_dx = _ddx_periodic(psi1, dx)
    dpsi1_dy = _ddy_periodic(psi1, dy)
    dpsi2_dx = _ddx_periodic(psi2, dx)
    dpsi2_dy = _ddy_periodic(psi2, dy)

    Jx = np.imag(np.conj(psi1) * dpsi1_dx) + np.imag(np.conj(psi2) * dpsi2_dx)
    Jy = np.imag(np.conj(psi1) * dpsi1_dy) + np.imag(np.conj(psi2) * dpsi2_dy)
    return Jx, Jy


def vorticity_from_components(
    psi1: np.ndarray,
    psi2: np.ndarray,
    dx: float,
    dy: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute omega = d u_y / dx - d u_x / dy, with u = J/rho."""
    Jx, Jy = current_from_components(psi1, psi2, dx=dx, dy=dy)
    rho = density_from_components(psi1, psi2)
    rho_safe = np.where(rho > eps, rho, eps)
    ux = Jx / rho_safe
    uy = Jy / rho_safe
    return _ddx_periodic(uy, dx) - _ddy_periodic(ux, dy)


def momentum_magnitude_from_components(
    psi1: np.ndarray,
    psi2: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Return |J| = sqrt(Jx^2 + Jy^2)."""
    Jx, Jy = current_from_components(psi1, psi2, dx=dx, dy=dy)
    return np.hypot(Jx, Jy)
