"""
Shadow Hamiltonian Simulation with Fourier Mode Observables.

For free particle H = k²/2 (discrete, periodic BC), the Fourier mode observables
O_q = e^{-i q·x} have EXACT closure:

  [H, O_q] = (energy shift) * O_q

This means each Fourier mode evolves independently with a phase factor.
The shadow Hamiltonian H_S is diagonal in the Fourier basis.

For the density Fourier modes rho_q = <e^{-i q·x}> = <psi| e^{-i q·X} |psi>,
the evolution depends on the wavefunction structure, not just a simple phase.

Actually, for density observables we need to be more careful:
rho_q(t) = ∫ e^{-i q·x} |psi(x,t)|² dx

This is NOT a simple linear observable of |psi>, so we need a different approach.

Instead, let's work with the wavefunction Fourier modes directly:
psi_k(t) = <k|psi(t)> evolves as psi_k(t) = psi_k(0) * e^{-i k² t / 2}

Then rho_q = ∑_k psi_k^* psi_{k+q} (convolution in Fourier space)
"""
import numpy as np
from numpy import pi
from scipy import linalg


def evolve_wavefunction_fourier(psi0, N, t):
    """
    Evolve wavefunction in Fourier space for free particle H = k²/2.

    psi_k(t) = psi_k(0) * exp(-i k² t / 2)

    Args:
        psi0: Initial wavefunction (N*N,) or (N, N)
        N: Grid size
        t: Time

    Returns:
        psi_t: Evolved wavefunction (N, N)
    """
    psi = psi0.reshape(N, N)

    # Transform to Fourier space
    psi_k = np.fft.fft2(psi)

    # Momentum grid (matching circuit_2D.py convention)
    kx = np.fft.fftfreq(N) * N
    ky = np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(kx, ky)

    # Kinetic energy: k² = kx² + ky²
    K2 = KX**2 + KY**2

    # Time evolution phase
    phase = np.exp(-0.5j * K2 * t)

    # Evolve in Fourier space
    psi_k_t = psi_k * phase

    # Transform back
    psi_t = np.fft.ifft2(psi_k_t)

    return psi_t


def extract_density(psi, N):
    """Extract density rho = |psi|² from wavefunction."""
    return np.abs(psi.reshape(N, N))**2


def extract_fourier_features(rho, K):
    """
    Extract low-frequency Fourier block from density.

    Args:
        rho: Density (N, N)
        K: Half-width of frequency block

    Returns:
        block: (2K+1, 2K+1) complex Fourier coefficients
        features: Real vector [Re(block), Im(block)]
    """
    N = rho.shape[0]
    rho_hat = np.fft.fft2(rho)
    rho_hat_c = np.fft.fftshift(rho_hat)

    center = N // 2
    block = rho_hat_c[center-K:center+K+1, center-K:center+K+1]

    vec = block.flatten()
    features = np.concatenate([np.real(vec), np.imag(vec)])

    return block, features


def compute_low_freq_energy(block):
    """Compute low-frequency energy E = sum(|block|²)."""
    return np.sum(np.abs(block)**2)


def build_density_fourier_shadow_hamiltonian(N, K):
    """
    Build shadow Hamiltonian for density Fourier modes.

    For free particle, the density Fourier mode rho_q evolves as:
    rho_q(t) = ∑_{k} psi_k^*(0) psi_{k+q}(0) * exp(-i (|k+q|² - |k|²) t / 2)

    This is NOT a simple linear evolution in rho_q space because it involves
    a sum over all k modes with different phases.

    However, for a Gaussian wavepacket with narrow momentum spread around k0,
    we can approximate:
    rho_q(t) ≈ rho_q(0) * exp(-i q·k0 t) * (spreading corrections)

    For the POC, we'll compute H_S numerically by:
    1. Computing [H, O_q] for each density observable O_q
    2. Projecting onto the observable space

    But actually, for density observables this is tricky because
    rho_q = <psi| e^{-iq·X} |psi> is quadratic in psi.

    Let's use a different approach: direct numerical differentiation.
    """
    # This is complex - let's use the moment approach instead
    # or compute H_S from short-time evolution
    pass


def build_shadow_from_short_time_evolution(psi0, N, K, dt_small=0.001, n_samples=10):
    """
    Build H_S by observing how density Fourier features evolve for short times.

    This computes d(features)/dt numerically and fits H_S such that:
    d(features)/dt = H_S @ features

    Note: This is still "data-driven" but uses the EXACT Hamiltonian evolution,
    not the quantum circuit. It's a way to extract H_S from the known H.

    Args:
        psi0: Initial wavefunction
        N: Grid size
        K: Fourier block half-width
        dt_small: Small time step for numerical derivative
        n_samples: Number of samples for fitting

    Returns:
        H_S: Shadow Hamiltonian for features
    """
    # Get initial features
    rho0 = extract_density(psi0, N)
    block0, features0 = extract_fourier_features(rho0, K)
    d = len(features0)

    # Collect samples at different short times
    times = np.linspace(0, dt_small * n_samples, n_samples + 1)
    features_list = [features0]

    for t in times[1:]:
        psi_t = evolve_wavefunction_fourier(psi0, N, t)
        rho_t = extract_density(psi_t, N)
        _, features_t = extract_fourier_features(rho_t, K)
        features_list.append(features_t)

    # Fit H_S from d(features)/dt = H_S @ features
    # Using finite differences: (f_{i+1} - f_i) / dt ≈ H_S @ f_i
    X = np.zeros((d, n_samples))
    Y = np.zeros((d, n_samples))

    for i in range(n_samples):
        X[:, i] = features_list[i]
        Y[:, i] = (features_list[i+1] - features_list[i]) / dt_small

    # Least squares: H_S = Y @ X^T @ (X @ X^T)^{-1}
    XXT = X @ X.T
    reg = 1e-10 * np.eye(d)
    H_S = Y @ X.T @ np.linalg.inv(XXT + reg)

    return H_S


def evolve_features_with_shadow(H_S, features0, dt, n_steps):
    """
    Evolve features using shadow Hamiltonian.

    d(features)/dt = H_S @ features
    Solution: features(t) = expm(H_S * t) @ features(0)
    """
    features_list = [features0.copy()]

    propagator = linalg.expm(H_S * dt)

    f = features0.copy()
    for _ in range(n_steps):
        f = propagator @ f
        features_list.append(f.copy())

    return features_list
