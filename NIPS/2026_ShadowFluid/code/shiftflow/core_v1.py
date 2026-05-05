"""shiftflow/core_v1.py

V!=0 SHIFT-FLOW: Galerkin-truncated dynamics with nontrivial potential.

Extends core_v0 (V=0, diagonal closure) to H = H0 + V where
V(x) = sum_j alpha_j cos(q_j . x).

Capabilities:
  1. Hamiltonian construction in 2D Fourier basis
  2. Index-set management (K task modes, R reference closure)
  3. Full-state reference evolution via eigendecomposition
  4. Galerkin-truncated evolution on subspace K
  5. Commutator leakage residual (a priori, state-independent)
  6. Linear error bound (no exponential blowup)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from shiftflow import core_v0


# ================================================================
# 1) Potential specification
# ================================================================


@dataclass(frozen=True)
class FourierPotential:
    """V_j(x) = alpha * cos(qx*x1 + qy*x2)."""

    alpha: float
    qx: int
    qy: int


def potential_single(
    alpha: float, qx: int = 1, qy: int = 0
) -> list[FourierPotential]:
    """Tier 1: single cosine V(x) = alpha cos(qx x1 + qy x2)."""
    return [FourierPotential(alpha=alpha, qx=qx, qy=qy)]


def potential_multi_deterministic(
    alphas: list[float], qxs: list[int], qys: list[int]
) -> list[FourierPotential]:
    """Tier 2: explicit multi-component potential."""
    return [
        FourierPotential(alpha=a, qx=qx, qy=qy)
        for a, qx, qy in zip(alphas, qxs, qys)
    ]


def potential_multi_random(
    J: int,
    alpha_scale: float = 0.5,
    q_max: int = 3,
    seed: int = 42,
) -> list[FourierPotential]:
    """Tier 2: J random Fourier components with |q| <= q_max."""
    rng = np.random.default_rng(seed)
    comps: list[FourierPotential] = []
    for _ in range(J):
        a = alpha_scale * rng.uniform(0.5, 1.5)
        qx = int(rng.integers(-q_max, q_max + 1))
        qy = int(rng.integers(-q_max, q_max + 1))
        if qx == 0 and qy == 0:
            qx = 1
        comps.append(FourierPotential(alpha=a, qx=qx, qy=qy))
    return comps


def potential_label(components: list[FourierPotential]) -> str:
    """Short human-readable label for a potential configuration."""
    if not components:
        return "V=0"
    parts = []
    for c in components:
        parts.append(f"{c.alpha:.2g}cos({c.qx},{c.qy})")
    return "V=" + "+".join(parts)


# ================================================================
# 2) Hamiltonian construction (dense, 2D Fourier basis)
# ================================================================


def build_H_dense(
    N: int, components: list[FourierPotential]
) -> np.ndarray:
    """Build H = H0 + V as dense (N^2 x N^2) Hermitian matrix.

    Fourier basis with flat index = iy * N + ix.
    H0 diagonal: E(kx,ky) = (kx^2 + ky^2)/2.
    V couples (iy,ix) -> ((iy +/- qy)%N, (ix +/- qx)%N) with alpha/2.
    """
    dim = N * N
    E = core_v0.energy_grid_free(N)
    H = np.diag(E.reshape(-1).astype(np.complex128))

    iy_all = np.arange(N).repeat(N)
    ix_all = np.tile(np.arange(N), N)

    for comp in components:
        ha = comp.alpha / 2.0
        for sy, sx in [(comp.qy, comp.qx), (-comp.qy, -comp.qx)]:
            iy2 = (iy_all + sy) % N
            ix2 = (ix_all + sx) % N
            src = iy_all * N + ix_all
            dst = iy2 * N + ix2
            H[dst, src] += ha

    return H


# ================================================================
# 3) Eigendecomposition
# ================================================================


def eigendecompose(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecompose Hermitian H -> (eigenvalues, eigenvectors).

    Returns (vals, vecs) where H = vecs @ diag(vals) @ vecs^dagger.
    """
    vals, vecs = np.linalg.eigh(H)
    return vals, vecs


# ================================================================
# 4) Index-set management
# ================================================================


def mask_to_flat(mask: np.ndarray, N: int) -> np.ndarray:
    """Boolean (N,N) mask -> sorted flat index array."""
    iy, ix = np.nonzero(mask)
    return np.sort(iy * N + ix).astype(int)


def flat_to_2d(flat_indices: np.ndarray, N: int):
    """Flat indices -> (iy_array, ix_array)."""
    return np.divmod(flat_indices, N)


def build_R_closure(
    r0: tuple[int, int],
    components: list[FourierPotential],
    N: int,
    max_hops: int = 1,
) -> np.ndarray:
    """Build reference set R by coupling-graph BFS from r0.

    Returns sorted flat indices. At hop h, adds all modes reachable
    from current R via one coupling vector +/- q_j.
    """
    current = {(r0[0] % N) * N + (r0[1] % N)}
    for _ in range(max_hops):
        frontier = set()
        for flat in current:
            iy, ix = divmod(flat, N)
            for comp in components:
                for sy, sx in [
                    (comp.qy, comp.qx),
                    (-comp.qy, -comp.qx),
                ]:
                    iy2 = (iy + sy) % N
                    ix2 = (ix + sx) % N
                    frontier.add(iy2 * N + ix2)
        current |= frontier
    return np.array(sorted(current), dtype=int)


def extract_submatrix(
    H_dense: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    """Extract H[indices, :][:, indices] as dense sub-block."""
    return H_dense[np.ix_(indices, indices)].copy()


# ================================================================
# 5) Evolution
# ================================================================


def evolve_full(
    b0_flat: np.ndarray,
    eig_vals: np.ndarray,
    eig_vecs: np.ndarray,
    t: float,
) -> np.ndarray:
    """Full-state evolution: b(t) = U exp(-i D t) U^dag b(0)."""
    coeffs = eig_vecs.conj().T @ b0_flat
    coeffs = coeffs * np.exp(-1j * eig_vals * t)
    return eig_vecs @ coeffs


def evolve_galerkin(
    b0_sub: np.ndarray,
    H_sub: np.ndarray,
    t: float,
) -> np.ndarray:
    """Galerkin-truncated evolution: expm(-i H_sub t) @ b0_sub."""
    U = expm(-1j * H_sub * t)
    return U @ b0_sub


# ================================================================
# 6) Leakage residual (a priori, state-independent)
# ================================================================


def leakage_apriori(
    H_dense: np.ndarray,
    K_flat: np.ndarray,
    R_flat: np.ndarray | None = None,
) -> float:
    """A priori commutator leakage (state-independent, RMS Frobenius).

    For each operator O_{k,r} = |k><r| in the dictionary S(K,R):
      [H, O_{k,r}] = sum_{k'} H_{k'k} |k'><r| - sum_{r'} H_{rr'} |k><r'|
    The residual outside S has squared norm:
      ||R_{k,r}||^2 = sum_{k' not in K} |H_{k'k}|^2
                     + sum_{r' not in R} |H_{rr'}|^2
    Returns RMS over all (k,r) pairs = sqrt(mean_leak_K + mean_leak_R).
    """
    if R_flat is None:
        R_flat = K_flat
    dim = H_dense.shape[0]
    all_set = set(range(dim))

    K_set = set(K_flat.tolist())
    R_set = set(R_flat.tolist())
    notK = np.array(sorted(all_set - K_set), dtype=int)
    notR = np.array(sorted(all_set - R_set), dtype=int)

    # Leakage along K (ket) direction
    if len(notK) > 0:
        H_notK_K = H_dense[np.ix_(notK, K_flat)]
        leak_K = np.sum(np.abs(H_notK_K) ** 2, axis=0)  # (|K|,)
    else:
        leak_K = np.zeros(len(K_flat))

    # Leakage along R (bra) direction
    if len(notR) > 0:
        H_notR_R = H_dense[np.ix_(notR, R_flat)]
        leak_R = np.sum(np.abs(H_notR_R) ** 2, axis=0)  # (|R|,)
    else:
        leak_R = np.zeros(len(R_flat))

    # RMS: mean over (k,r) of (leak_K[k] + leak_R[r])
    #     = mean(leak_K) + mean(leak_R)
    mean_leak = float(np.mean(leak_K)) + float(np.mean(leak_R))
    return float(np.sqrt(max(mean_leak, 0.0)))


def apriori_bound(leakage: float, t: float) -> float:
    """A priori error bound: ||DeltaZ(t)||_F <= t * leakage."""
    return abs(t) * leakage


# ================================================================
# 7) Actual error computation (requires full reference)
# ================================================================


def actual_Z_error(
    b_full_K: np.ndarray,
    b_full_R: np.ndarray,
    b_trunc_K: np.ndarray,
    b_trunc_R: np.ndarray,
) -> float:
    """||Z^full - Z^trunc||_F where Z_{k,r} = b_k * conj(b_r).

    Z^full = b_full_K @ b_full_R^dag  (rank-1, |K| x |R|)
    Z^trunc = b_trunc_K @ b_trunc_R^dag
    """
    Z_full = np.outer(b_full_K, b_full_R.conj())
    Z_trunc = np.outer(b_trunc_K, b_trunc_R.conj())
    return float(np.linalg.norm(Z_full - Z_trunc, "fro"))


def actual_coeff_error(
    b_full_sub: np.ndarray,
    b_trunc_sub: np.ndarray,
) -> float:
    """Relative L2 error of Fourier coefficients on a subspace."""
    num = float(np.linalg.norm(b_full_sub - b_trunc_sub))
    den = float(np.linalg.norm(b_full_sub))
    return num / den if den > 0 else float("nan")


# ================================================================
# 8) Convenience: reconstruct fields from truncated coefficients
# ================================================================


def scatter_to_2d(
    b_sub: np.ndarray, flat_indices: np.ndarray, N: int
) -> np.ndarray:
    """Place subspace coefficients into an (N,N) array (zeros elsewhere)."""
    out = np.zeros((N, N), dtype=np.complex128)
    iy, ix = flat_to_2d(flat_indices, N)
    out[iy, ix] = b_sub
    return out


def reconstruct_lowpass(
    b1_sub: np.ndarray,
    b2_sub: np.ndarray,
    K_flat: np.ndarray,
    N: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct low-pass (psi1, psi2) from K-subspace coefficients."""
    b1_2d = scatter_to_2d(b1_sub, K_flat, N)
    b2_2d = scatter_to_2d(b2_sub, K_flat, N)
    psi1 = core_v0.unitary_ifft2(b1_2d)
    psi2 = core_v0.unitary_ifft2(b2_2d)
    return psi1, psi2


# ================================================================
# 9) Full experiment runner (single configuration)
# ================================================================


@dataclass
class V1Result:
    """Results from a single V!=0 experiment point."""

    # Config
    N: int
    K0: float
    M_K: int
    R_size: int
    t: float
    alpha_total: float
    J: int
    V_label: str

    # Errors (truncated vs full)
    err_b_K_rel: float       # relative L2 of K-mode coefficients
    err_rho_vs_full: float   # density error
    err_E_LP: float          # task energy error

    # Leakage and bounds
    leakage_apriori: float   # state-independent RMS leakage
    bound_apriori: float     # t * leakage_apriori
    err_Z_frob: float        # actual ||DeltaZ||_F

    # Low-pass baseline errors (full low-pass vs full)
    err_rho_lp_vs_full: float

    # Qiskit timing (seconds); NaN when classical path used
    rt_qiskit_K_s: float = float("nan")
    rt_qiskit_R_s: float = float("nan")


def run_single(
    N: int,
    components: list[FourierPotential],
    K0: float,
    t: float,
    psi1_0: np.ndarray,
    psi2_0: np.ndarray,
    *,
    r0: tuple[int, int] = (0, 0),
    R_hops: int = 1,
    H_dense: np.ndarray | None = None,
    eig: tuple[np.ndarray, np.ndarray] | None = None,
    use_qiskit: bool = False,
    qiskit_sim=None,
) -> V1Result:
    """Run full + Galerkin evolution and compute all metrics.

    Parameters
    ----------
    H_dense, eig : precomputed Hamiltonian and eigendecomposition (for speed).
    use_qiskit : if True, use Qiskit Aer for shadow (Galerkin) evolution.
    qiskit_sim : shared AerSimulator instance (avoids repeated creation).
    """
    # Build / reuse Hamiltonian
    if H_dense is None:
        H_dense = build_H_dense(N, components)
    if eig is None:
        eig = eigendecompose(H_dense)
    eig_vals, eig_vecs = eig

    # Index sets
    mask = core_v0.low_freq_mask(N, K0)
    K_flat = mask_to_flat(mask, N)
    M_K = len(K_flat)

    if len(components) > 0:
        R_flat = build_R_closure(r0, components, N, max_hops=R_hops)
    else:
        R_flat = K_flat  # V=0: block dictionary

    # Initial Fourier coefficients
    b1_0 = core_v0.unitary_fft2(psi1_0).reshape(-1)
    b2_0 = core_v0.unitary_fft2(psi2_0).reshape(-1)

    b1_K_0 = b1_0[K_flat]
    b2_K_0 = b2_0[K_flat]
    b1_R_0 = b1_0[R_flat]
    b2_R_0 = b2_0[R_flat]

    # Hamiltonian sub-blocks
    H_K = extract_submatrix(H_dense, K_flat)
    H_R = extract_submatrix(H_dense, R_flat)

    # ---- Full evolution ----
    b1_full_t = evolve_full(b1_0, eig_vals, eig_vecs, t)
    b2_full_t = evolve_full(b2_0, eig_vals, eig_vecs, t)

    b1_full_K = b1_full_t[K_flat]
    b2_full_K = b2_full_t[K_flat]
    b1_full_R = b1_full_t[R_flat]
    b2_full_R = b2_full_t[R_flat]

    # ---- Galerkin-truncated evolution ----
    rt_qiskit_K_s = float("nan")
    rt_qiskit_R_s = float("nan")

    if use_qiskit:
        from shiftflow.qiskit_shadow_v1 import shadow_evolve_v1

        (b1_trunc_K, b2_trunc_K,
         b1_trunc_R, b2_trunc_R,
         rt_qiskit_K_s, rt_qiskit_R_s) = shadow_evolve_v1(
            b1_K_0, b2_K_0, b1_R_0, b2_R_0,
            H_K, H_R, t, sim=qiskit_sim,
        )
    else:
        b1_trunc_K = evolve_galerkin(b1_K_0, H_K, t)
        b2_trunc_K = evolve_galerkin(b2_K_0, H_K, t)
        b1_trunc_R = evolve_galerkin(b1_R_0, H_R, t)
        b2_trunc_R = evolve_galerkin(b2_R_0, H_R, t)

    # ---- Errors: coefficients ----
    err1 = actual_coeff_error(b1_full_K, b1_trunc_K)
    err2 = actual_coeff_error(b2_full_K, b2_trunc_K)
    err_b_K_rel = max(err1, err2)

    # ---- Errors: Z matrix (Frobenius) ----
    ez1 = actual_Z_error(b1_full_K, b1_full_R, b1_trunc_K, b1_trunc_R)
    ez2 = actual_Z_error(b2_full_K, b2_full_R, b2_trunc_K, b2_trunc_R)
    err_Z_frob = float(np.sqrt(ez1**2 + ez2**2))

    # ---- Errors: density ----
    dx = 2.0 * np.pi / N
    psi1_full_lp, psi2_full_lp = reconstruct_lowpass(
        b1_full_K, b2_full_K, K_flat, N
    )
    psi1_trunc_lp, psi2_trunc_lp = reconstruct_lowpass(
        b1_trunc_K, b2_trunc_K, K_flat, N
    )
    psi1_full = core_v0.unitary_ifft2(b1_full_t.reshape(N, N))
    psi2_full = core_v0.unitary_ifft2(b2_full_t.reshape(N, N))

    rho_full = core_v0.density_from_components(psi1_full, psi2_full)
    rho_trunc = core_v0.density_from_components(psi1_trunc_lp, psi2_trunc_lp)
    rho_full_lp = core_v0.density_from_components(psi1_full_lp, psi2_full_lp)

    from shiftflow.metrics import rel_l2

    err_rho_vs_full = rel_l2(rho_trunc, rho_full)
    err_rho_lp_vs_full = rel_l2(rho_full_lp, rho_full)

    # ---- Task energy E_LP ----
    E_LP_full = float(
        np.sum(np.abs(b1_full_K) ** 2 + np.abs(b2_full_K) ** 2)
    )
    E_LP_trunc = float(
        np.sum(np.abs(b1_trunc_K) ** 2 + np.abs(b2_trunc_K) ** 2)
    )
    err_E_LP = (
        abs(E_LP_trunc - E_LP_full) / abs(E_LP_full)
        if abs(E_LP_full) > 0
        else float("nan")
    )

    # ---- Leakage and bound ----
    leak = leakage_apriori(H_dense, K_flat, R_flat)
    bound = apriori_bound(leak, t)

    # ---- Summary ----
    alpha_total = sum(abs(c.alpha) for c in components)

    return V1Result(
        N=N,
        K0=K0,
        M_K=M_K,
        R_size=len(R_flat),
        t=t,
        alpha_total=alpha_total,
        J=len(components),
        V_label=potential_label(components),
        err_b_K_rel=err_b_K_rel,
        err_rho_vs_full=err_rho_vs_full,
        err_E_LP=err_E_LP,
        leakage_apriori=leak,
        bound_apriori=bound,
        err_Z_frob=err_Z_frob,
        err_rho_lp_vs_full=err_rho_lp_vs_full,
        rt_qiskit_K_s=rt_qiskit_K_s,
        rt_qiskit_R_s=rt_qiskit_R_s,
    )
