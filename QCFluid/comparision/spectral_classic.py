# === spectral_classic.py ===
from scipy.fft import fft2, ifft2
from config import N, pi
import numpy as np


# 和量子版本数学上等价
def evolve_spectral_quantum_style(psi, t, N):
    """
    Evolves a wavefunction psi under the Hamiltonian H = -∇²/2 for time t,
    using classical spectral method in a way that matches the quantum simulation
    with QFT-based k² operator.

    psi : 2D complex array of shape (N, N)
    t   : evolution time
    N   : number of grid points (assumed square domain)
    """
    # Use unit grid spacing: x ∈ [0, N-1], so dx = 1
    # QFT in quantum sim assumes k ∈ {0, 1, ..., N-1}
    kx = np.fft.fftfreq(N, d=1.0) * N  # Gives: [0, 1, ..., N/2-1, -N/2, ..., -1]
    ky = np.fft.fftfreq(N, d=1.0) * N
    KX, KY = np.meshgrid(kx, ky, indexing='ij')  # indexing='ij' for matrix-aligned axes
    K2 = KX**2 + KY**2

    # Apply kinetic phase shift: matches e^{-i k^2 t / 2}
    U = np.exp(-1j * K2 * t / 2)

    # Spectral evolution
    psi_hat = fft2(psi)
    psi_hat_evolved = psi_hat * U
    psi_new = ifft2(psi_hat_evolved)

    return psi_new

import numpy as np
from numpy.fft import fft2, ifft2

def evolve_rk4_baseline(psi0, t, N, nsteps=100, safety=2.5):
    """
    RK4 evolution of i∂t ψ = -½∇²ψ on a NxN grid over total time t.
    Internally subdivides each outer step so that |λ dt_sub| ≤ safety
    (default safety=2.5; RK4 stability limit≈2.8).

    Parameters
    ----------
    psi0   : complex ndarray (N, N)
    t      : float   total physical time
    N      : int     grid size
    nsteps : int     number of *outer* steps (each of length dt = t/nsteps)
    safety : float   stability margin (≤2.8 for classic RK4)

    Returns
    -------
    psi_t  : complex ndarray (N, N)  wave-function at time t
    """
    dt_big = t / nsteps
    psi = psi0.copy()

    # ----- spectral Laplacian -----
    k = np.fft.fftfreq(N) * N         # spacing d=1 ⇒ k_max = N/2
    KX, KY = np.meshgrid(k, k, indexing='ij')
    K2 = KX**2 + KY**2
    lam_max = 0.5 * K2.max()          # |λ|max = ½ k_max²

    def laplacian(arr):
        return ifft2(-K2 * fft2(arr))

    # ----- sub-step size ensuring stability -----
    dt_sub = safety / lam_max
    n_sub  = max(1, int(np.ceil(dt_big / dt_sub)))
    dt_sub = dt_big / n_sub           # exactly divides outer step

    for _ in range(nsteps):
        for _ in range(n_sub):
            k1 = -0.5j * laplacian(psi)
            k2 = -0.5j * laplacian(psi + 0.5*dt_sub*k1)
            k3 = -0.5j * laplacian(psi + 0.5*dt_sub*k2)
            k4 = -0.5j * laplacian(psi +       dt_sub*k3)
            psi += dt_sub/6.0 * (k1 + 2*k2 + 2*k3 + k4)

    return psi

