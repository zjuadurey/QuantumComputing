"""
shadow_vortex_demo.py

A proof-of-concept "shadow simulation" pipeline for the 2D vortex initial condition
in circuit_2D.py (Communications Physics style Schrödinger-flow, V=0 / free evolution).

Key idea:
- Baseline (CP-style): prepare full wavefunction |psi(0)>, apply QFT -> kinetic phase -> iQFT,
  and (in simulation) read full statevector to compute density rho(x,y,t).
- Shadow-style (what you asked for): do NOT evolve the full state. Instead, evolve a reduced set
  of low-frequency k-space coherences (shadow observables), then reconstruct a low-pass density map.

This script produces:
1) Baseline density rho_full(x,y,t) from qiskit statevector simulation.
2) Baseline low-pass density rho_lp_baseline from truncating baseline's k-space coefficients.
3) Shadow density rho_lp_shadow from evolving ONLY low-frequency coherences.
4) Error metrics + plots.

Notes:
- This file requires qiskit + qiskit-aer.
- We prioritize density (rho). Extending to velocity/vorticity is straightforward using your
  compute_fluid_quantities(), but density is the cleanest first target.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Qiskit imports (required)
# ----------------------------
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT


# ============================================================
# 1) Baseline circuit pieces from circuit_2D.py
# ============================================================

def kinetic_operator(n: int, dt: float) -> QuantumCircuit:
    """Same kinetic operator as in circuit_2D.py."""
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
    """
    Baseline full-state evolution for V=0 using qiskit:
      initialize -> QFTx QFTy -> kinetic phase -> iQFTx iQFTy
    Returns the full final statevector (length 2^(nx+ny+1)).
    """
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
    sv = np.asarray(result.data(0)["statevector"], dtype=np.complex128)
    return sv


# ============================================================
# 2) Vortex initial condition (same as circuit_2D.py)
# ============================================================

def vortex_initial_condition(N: int, sigma: float = 3.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (psi1_0, psi2_0, initial_state_vector) where:
    - psi1_0, psi2_0 have shape (N,N)
    - initial_state_vector has length 2*N*N (psi1 flattened then psi2 flattened), normalized
    """
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

    # 关键：按“量子态”整体归一化，且把归一化同步回 psi1, psi2
    stacked = np.array([psi1, psi2]).reshape(-1).astype(np.complex128)
    stacked /= np.linalg.norm(stacked)

    psi1_n = stacked[: N * N].reshape(N, N)
    psi2_n = stacked[N * N :].reshape(N, N)

    return psi1_n, psi2_n, stacked


# ============================================================
# 3) Low-frequency mask + "shadow" observables
# ============================================================

def k_grid(N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (kx, ky, KX, KY) using the same convention as circuit_2D.py:
    kx = fftfreq(N)*N in integer units.
    """
    kx = np.fft.fftfreq(N) * N
    ky = np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(kx, ky)
    return kx, ky, KX, KY


def low_freq_mask(N: int, K0: float) -> np.ndarray:
    """
    Circular low-frequency set: keep modes with sqrt(kx^2+ky^2) <= K0.
    """
    _, _, KX, KY = k_grid(N)
    return (KX**2 + KY**2) <= (K0**2)


def unitary_fft2(field_xy: np.ndarray) -> np.ndarray:
    """
    Unitary 2D Fourier transform coefficients consistent with QFT normalization:
      b(k) = FFT2(field)/N, so that sum|b|^2 = sum|field|^2.
    """
    N = field_xy.shape[0]
    return np.fft.fft2(field_xy) / N


def unitary_ifft2(field_k: np.ndarray) -> np.ndarray:
    """
    Inverse of unitary_fft2:
      field = IFFT2(b) * N
    """
    N = field_k.shape[0]
    return np.fft.ifft2(field_k) * N


def choose_reference_mode(bk: np.ndarray, mask: np.ndarray, prefer: tuple[int, int] = (0, 0),
                          min_rel: float = 1e-3) -> tuple[int, int]:
    """
    Pick reference k0 index (iy, ix). Prefer (0,0) unless it's too small.
    We only need k0 to have a reasonable amplitude to avoid numerical blow-ups.
    """
    iy0, ix0 = prefer
    amp0 = np.abs(bk[iy0, ix0])
    amps = np.abs(bk[mask])
    max_amp = float(np.max(amps)) if amps.size else 0.0
    if max_amp == 0.0:
        return iy0, ix0
    if amp0 >= min_rel * max_amp:
        return iy0, ix0

    # otherwise choose the max-amplitude mode within mask
    idx_flat = np.argmax(np.abs(bk) * mask)
    iy, ix = np.unravel_index(idx_flat, bk.shape)
    return int(iy), int(ix)


def energy_grid_free(N: int) -> np.ndarray:
    """
    Free-particle energy E_k = (kx^2+ky^2)/2 in the integer-k convention used here.
    """
    _, _, KX, KY = k_grid(N)
    return 0.5 * (KX**2 + KY**2)


def shadow_evolve_lowpass_from_coherences(
    b0: np.ndarray,
    mask: np.ndarray,
    t: float,
    k0_idx: tuple[int, int],
    E: np.ndarray,
) -> np.ndarray:
    """
    Shadow-style evolution: evolve ONLY coherences relative to k0, then reconstruct b(k,t) on mask.
    - b0: unitary k-space coefficients at t=0
    - mask: which k modes we keep
    - k0_idx: reference mode indices (iy0, ix0)
    - E: energy grid E(k)
    Returns b_shadow(k,t) with zeros outside mask.
    """
    iy0, ix0 = k0_idx
    E0 = float(E[iy0, ix0])

    # Coherence z_k(t) = b_k(t) * conj(b_k0(t))
    # For free evolution b_k(t)=b_k(0) e^{-iE_k t}, so
    # z_k(t)= z_k(0) e^{-i(E_k-E0)t}
    b_k0_0 = b0[iy0, ix0]
    z0 = b0 * np.conj(b_k0_0)
    phase = np.exp(-1j * (E - E0) * t)
    zt = z0 * phase

    # We also track b_k0(t) itself (one complex number)
    b_k0_t = b_k0_0 * np.exp(-1j * E0 * t)

    # Reconstruct b_k(t) on the retained mask: b_k(t) = z_k(t) / conj(b_k0(t))
    b_t = np.zeros_like(b0)
    b_t[mask] = zt[mask] / np.conj(b_k0_t)
    b_t[iy0, ix0] = b_k0_t  # make sure reference is set
    return b_t


# ============================================================
# 4) Helpers: reshape qiskit statevector -> (psi1, psi2)
# ============================================================

def statevector_to_components(sv: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
    """
    sv is length 2*N*N, ordered as [sigma=0 block (N*N), sigma=1 block (N*N)]
    consistent with circuit_2D.py initialization.
    Returns psi1, psi2 with shape (N,N).
    """
    tmp = sv.reshape(2, N, N)
    return tmp[0, :, :], tmp[1, :, :]


def density_from_components(psi1: np.ndarray, psi2: np.ndarray) -> np.ndarray:
    return np.abs(psi1)**2 + np.abs(psi2)**2


# ============================================================
# 5) Main demo
# ============================================================

def main():
    # -------------------------
    # Tunable parameters
    # -------------------------
    N = 2**6         # grid size (32)
    nx = ny = 6      # qubits per spatial dim (since N=2^5)
    t = 3.0          # same as dt in your example
    K0 = 6        # low-frequency cutoff (adjustable)

    # -------------------------
    # Initial condition
    # -------------------------
    psi1_0, psi2_0, initial_state = vortex_initial_condition(N=N, sigma=3.0)

    # -------------------------
    # Baseline: full evolution by qiskit statevector
    # -------------------------
    sv_t = evolve_statevector_v0(nx=nx, ny=ny, t=t, initial_state=initial_state)
    psi1_full, psi2_full = statevector_to_components(sv_t, N)
    rho_full = density_from_components(psi1_full, psi2_full)

    # Also baseline low-pass (truncate the *baseline* at time t for fair comparison)
    mask = low_freq_mask(N, K0)
    b1_full = unitary_fft2(psi1_full)
    b2_full = unitary_fft2(psi2_full)
    b1_full_lp = np.where(mask, b1_full, 0.0)
    b2_full_lp = np.where(mask, b2_full, 0.0)
    psi1_full_lp = unitary_ifft2(b1_full_lp)
    psi2_full_lp = unitary_ifft2(b2_full_lp)
    rho_full_lp = density_from_components(psi1_full_lp, psi2_full_lp)

    # -------------------------
    # Shadow: evolve only low-frequency coherences (from t=0 data)
    # -------------------------
    E = energy_grid_free(N)
    b1_0 = unitary_fft2(psi1_0)
    b2_0 = unitary_fft2(psi2_0)

    k0_1 = choose_reference_mode(b1_0, mask, prefer=(0, 0), min_rel=1e-3)
    k0_2 = choose_reference_mode(b2_0, mask, prefer=(0, 0), min_rel=1e-3)

    b1_shadow = shadow_evolve_lowpass_from_coherences(b0=b1_0, mask=mask, t=t, k0_idx=k0_1, E=E)
    b2_shadow = shadow_evolve_lowpass_from_coherences(b0=b2_0, mask=mask, t=t, k0_idx=k0_2, E=E)

    psi1_shadow = unitary_ifft2(b1_shadow)
    psi2_shadow = unitary_ifft2(b2_shadow)
    rho_shadow = density_from_components(psi1_shadow, psi2_shadow)

    # -------------------------
    # Metrics: shadow vs baseline low-pass
    # -------------------------
    def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
        num = float(np.linalg.norm(a - b))
        den = float(np.linalg.norm(b))
        return num / den if den != 0 else np.nan

    err_shadow_vs_lp = rel_l2(rho_shadow, rho_full_lp)
    err_shadow_vs_full = rel_l2(rho_shadow, rho_full)

    print(f"N={N}, t={t}, K0={K0}")
    print(f"reference k0 (psi1) = {k0_1}, |b_k0| = {abs(b1_0[k0_1]):.3e}")
    print(f"reference k0 (psi2) = {k0_2}, |b_k0| = {abs(b2_0[k0_2]):.3e}")
    print(f"rel L2 error: shadow vs baseline(low-pass) = {err_shadow_vs_lp:.3e}")
    print(f"rel L2 error: shadow vs baseline(full) = {err_shadow_vs_full:.3e}")

    # -------------------------
    # Plot: density maps
    # -------------------------
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    y = np.linspace(0, 2*np.pi, N, endpoint=False)
    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), constrained_layout=True)

    im0 = axes[0].pcolormesh(X, Y, rho_full, shading="auto")
    axes[0].set_title("Baseline (full) ρ")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].pcolormesh(X, Y, rho_full_lp, shading="auto")
    axes[1].set_title(f"Baseline (low-pass) ρ, K0={K0}")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].pcolormesh(X, Y, rho_shadow, shading="auto")
    axes[2].set_title("Shadow (coherences) ρ")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    diff = rho_shadow - rho_full_lp
    im3 = axes[3].pcolormesh(X, Y, diff, shading="auto")
    axes[3].set_title("Shadow − Baseline(low-pass)")
    fig.colorbar(im3, ax=axes[3], fraction=0.046)

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


if __name__ == "__main__":
    main()
