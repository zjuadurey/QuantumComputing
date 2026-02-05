"""
Shadow Hamiltonian Simulation: construct reduced generator H_S from commutator closure.

The key idea is to find observables O_m such that [H, O_m] ≈ -∑_{m'} h_{mm'} O_{m'}
(approximate invariance property). Then H_S = (h_{mm'}) is the reduced generator.
"""
import numpy as np
from numpy import pi, exp
from scipy import linalg


def build_kinetic_hamiltonian(N):
    """
    Build the kinetic Hamiltonian H = -1/2 (d^2/dx^2 + d^2/dy^2) in Fourier space.

    For the free particle (V=0), H is diagonal in momentum space:
    H = 1/2 (kx^2 + ky^2)

    In position space on an N x N grid with x,y in [-pi, pi), this becomes
    a matrix acting on the wavefunction.

    Args:
        N: Grid size (N x N)

    Returns:
        H: Hamiltonian as (N*N, N*N) matrix acting on flattened wavefunction
    """
    # Momentum grid
    kx = np.fft.fftfreq(N) * N  # k = 0, 1, ..., N/2-1, -N/2, ..., -1
    ky = np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(kx, ky)

    # Kinetic energy in momentum space: T = 1/2 (kx^2 + ky^2)
    T_k = 0.5 * (KX**2 + KY**2)

    # Build H in position space: H = F^{-1} T_k F
    # where F is 2D FFT
    # For a vector psi (flattened), H @ psi = ifft2(T_k * fft2(psi.reshape(N,N))).flatten()

    # Build full matrix representation
    H = np.zeros((N*N, N*N), dtype='complex128')

    for j in range(N*N):
        # Unit vector e_j
        e_j = np.zeros(N*N, dtype='complex128')
        e_j[j] = 1.0

        # Apply H to e_j
        psi = e_j.reshape(N, N)
        psi_k = np.fft.fft2(psi)
        H_psi_k = T_k * psi_k
        H_psi = np.fft.ifft2(H_psi_k)

        H[:, j] = H_psi.flatten()

    return H


def build_fourier_observable(N, kx_idx, ky_idx):
    """
    Build Fourier mode observable O_k = exp(-i k·X) as a diagonal matrix.

    For density observable, we use ρ_k = ∫ exp(-i k·x) |ψ(x)|^2 dx
    which corresponds to ⟨ψ| O_k |ψ⟩ where O_k is diagonal with entries exp(-i k·x_j).

    Args:
        N: Grid size
        kx_idx, ky_idx: Frequency indices (integers)

    Returns:
        O_k: Observable as (N*N, N*N) diagonal matrix
    """
    x = np.linspace(-pi, pi, N, endpoint=False)
    y = np.linspace(-pi, pi, N, endpoint=False)
    X, Y = np.meshgrid(x, y)

    # Wave vector
    kx = kx_idx
    ky = ky_idx

    # exp(-i k·x) at each grid point
    phase = exp(-1j * (kx * X + ky * Y))

    # Diagonal matrix
    O_k = np.diag(phase.flatten())

    return O_k


def build_observable_set(N, K):
    """
    Build set of low-frequency Fourier observables.

    For k in [-K, K] x [-K, K], we have (2K+1)^2 observables.

    Args:
        N: Grid size
        K: Maximum frequency index

    Returns:
        observables: List of (N*N, N*N) matrices
        k_indices: List of (kx, ky) tuples
    """
    observables = []
    k_indices = []

    for kx in range(-K, K+1):
        for ky in range(-K, K+1):
            O_k = build_fourier_observable(N, kx, ky)
            observables.append(O_k)
            k_indices.append((kx, ky))

    return observables, k_indices


def compute_commutator(A, B):
    """Compute [A, B] = AB - BA."""
    return A @ B - B @ A


def build_shadow_hamiltonian(H, observables, ridge_lambda=1e-6):
    """
    Build shadow Hamiltonian H_S by solving for commutator closure.

    For each observable O_m, we want:
        [H, O_m] ≈ -∑_{m'} h_{mm'} O_{m'}

    This is a least-squares problem for each row of H_S.

    Args:
        H: Full Hamiltonian (N*N, N*N)
        observables: List of M observable matrices
        ridge_lambda: Regularization parameter

    Returns:
        H_S: Shadow Hamiltonian (M, M) complex matrix
        residuals: List of Frobenius norm residuals for each observable
    """
    M = len(observables)
    n = H.shape[0]

    # Vectorize observables for least squares
    # Each observable O_m is n x n, flatten to n^2
    O_vecs = np.zeros((M, n*n), dtype='complex128')
    for m, O_m in enumerate(observables):
        O_vecs[m, :] = O_m.flatten()

    # For each O_m, compute [H, O_m] and solve for coefficients
    H_S = np.zeros((M, M), dtype='complex128')
    residuals = []

    for m, O_m in enumerate(observables):
        # Compute commutator [H, O_m]
        comm = compute_commutator(H, O_m)
        comm_vec = comm.flatten()

        # Solve: -∑_{m'} h_{mm'} O_{m'} ≈ [H, O_m]
        # i.e., O_vecs.T @ h_m ≈ -comm_vec
        # Least squares: h_m = -(O_vecs @ O_vecs.H + lambda*I)^{-1} @ O_vecs @ comm_vec

        # Using real formulation for stability
        # Stack real and imaginary parts
        O_real = np.vstack([O_vecs.real, O_vecs.imag])  # (2M, n^2)
        comm_real = np.concatenate([comm_vec.real, comm_vec.imag])  # (2*n^2,)

        # Solve: O_real.T @ h ≈ -comm_real
        # h = -(O_real @ O_real.T + lambda*I)^{-1} @ O_real @ comm_real

        A_mat = O_vecs.conj() @ O_vecs.T  # (M, M)
        b_vec = O_vecs.conj() @ comm_vec  # (M,)

        reg = ridge_lambda * np.eye(M)
        h_m = -np.linalg.solve(A_mat + reg, b_vec)

        H_S[m, :] = h_m

        # Compute residual: ||[H, O_m] + ∑_{m'} h_{mm'} O_{m'}||_F
        reconstructed = sum(h_m[mp] * observables[mp] for mp in range(M))
        residual = np.linalg.norm(comm + reconstructed, 'fro')
        residuals.append(residual)

    return H_S, residuals


def compute_expectation(O, psi):
    """
    Compute expectation value ⟨ψ|O|ψ⟩.

    Args:
        O: Observable matrix (n, n)
        psi: State vector (n,)

    Returns:
        Expectation value (complex)
    """
    return psi.conj() @ O @ psi


def extract_expectations(observables, psi):
    """
    Extract expectation values for all observables.

    Args:
        observables: List of observable matrices
        psi: State vector

    Returns:
        o: Vector of expectation values
    """
    return np.array([compute_expectation(O, psi) for O in observables])


def evolve_shadow(H_S, o0, t_list):
    """
    Evolve observable expectations using shadow dynamics.

    d/dt o(t) = -i H_S o(t)
    Solution: o(t) = exp(-i H_S t) o(0)

    Args:
        H_S: Shadow Hamiltonian (M, M)
        o0: Initial expectations (M,)
        t_list: List of time points

    Returns:
        o_list: List of expectation vectors at each time
    """
    o_list = []

    for t in t_list:
        if t == 0.0:
            o_list.append(o0.copy())
        else:
            propagator = linalg.expm(-1j * H_S * t)
            o_t = propagator @ o0
            o_list.append(o_t)

    return o_list


def evolve_shadow_stepwise(H_S, o0, dt, n_steps):
    """
    Evolve observable expectations step by step.

    Args:
        H_S: Shadow Hamiltonian (M, M)
        o0: Initial expectations (M,)
        dt: Time step
        n_steps: Number of steps

    Returns:
        o_list: List of expectation vectors (length n_steps + 1)
    """
    o_list = [o0.copy()]

    # Precompute propagator
    propagator = linalg.expm(-1j * H_S * dt)

    o_current = o0.copy()
    for _ in range(n_steps):
        o_current = propagator @ o_current
        o_list.append(o_current.copy())

    return o_list
