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

def evolve_rk4_baseline(psi0, t, N, nsteps=100):
    """
    Evolve psi under i∂t ψ = -1/2 ∇²ψ using RK4 method in time domain.
    Input:
        psi0 : ndarray (N x N), initial wavefunction
        t    : float, total evolution time
        N    : int, grid size
        nsteps : int, number of RK4 time steps
    Output:
        psi_t : ndarray (N x N), evolved wavefunction at time t
    """
    dt = t / nsteps
    psi = psi0.copy()
    
    # Precompute k² operator in spectral space
    kx = np.fft.fftfreq(N, d=1.0) * N
    ky = np.fft.fftfreq(N, d=1.0) * N
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2

    def laplacian(psi):
        psi_hat = fft2(psi)
        lap_psi_hat = -K2 * psi_hat
        return ifft2(lap_psi_hat)

    for _ in range(nsteps):
        k1 = -1j * 0.5 * laplacian(psi)
        k2 = -1j * 0.5 * laplacian(psi + 0.5 * dt * k1)
        k3 = -1j * 0.5 * laplacian(psi + 0.5 * dt * k2)
        k4 = -1j * 0.5 * laplacian(psi + dt * k3)
        psi += dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    return psi
