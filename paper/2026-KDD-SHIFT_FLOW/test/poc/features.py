"""
Feature extraction: low-frequency Fourier modes of density.
"""
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift


def extract_density(statevector, N):
    """
    Extract density rho = |psi1|^2 from statevector.

    Args:
        statevector: Full statevector of length 2*N*N
        N: Grid size (N x N)

    Returns:
        rho: Density array of shape (N, N)
    """
    tmp = statevector.reshape(2, N, N)
    psi1 = tmp[0, :, :]
    rho = np.abs(psi1)**2
    return rho


def extract_fourier_features(rho, K=2):
    """
    Extract low-frequency Fourier features from density.

    Args:
        rho: Density array of shape (N, N)
        K: Half-width of the frequency block (total width = 2K+1)

    Returns:
        features: Real vector of concatenated real and imaginary parts
        block: The complex Fourier block (for energy computation)
    """
    N = rho.shape[0]

    # Compute 2D FFT
    rho_hat = fft2(rho)

    # Shift to center frequencies
    rho_hat_c = fftshift(rho_hat)

    # Extract centered block of size (2K+1) x (2K+1)
    center = N // 2
    block = rho_hat_c[center-K:center+K+1, center-K:center+K+1]

    # Vectorize in fixed order (row-major)
    vec = block.flatten()

    # Concatenate real and imaginary parts
    features = np.concatenate([np.real(vec), np.imag(vec)])

    return features, block


def compute_low_freq_energy(block):
    """
    Compute low-frequency energy E_low = sum(|block|^2).

    Args:
        block: Complex Fourier block

    Returns:
        E_low: Low-frequency energy (scalar)
    """
    return np.sum(np.abs(block)**2)


def features_to_block(features, K=2):
    """
    Convert feature vector back to complex Fourier block.

    Args:
        features: Real vector of concatenated real and imaginary parts
        K: Half-width of the frequency block

    Returns:
        block: Complex Fourier block of shape (2K+1, 2K+1)
    """
    block_size = (2*K + 1)**2
    real_part = features[:block_size]
    imag_part = features[block_size:]
    vec = real_part + 1j * imag_part
    block = vec.reshape(2*K+1, 2*K+1)
    return block


def reconstruct_low_freq_density(block, N, K=2):
    """
    Reconstruct low-frequency density by zeroing out all modes except the block.

    Args:
        block: Complex Fourier block of shape (2K+1, 2K+1)
        N: Grid size
        K: Half-width of the frequency block

    Returns:
        rho_low: Reconstructed low-frequency density (N, N)
    """
    # Create full spectrum with zeros
    rho_hat_c = np.zeros((N, N), dtype='complex128')

    # Insert block at center
    center = N // 2
    rho_hat_c[center-K:center+K+1, center-K:center+K+1] = block

    # Inverse shift and inverse FFT
    rho_hat = ifftshift(rho_hat_c)
    rho_low = np.real(ifft2(rho_hat))

    return rho_low
