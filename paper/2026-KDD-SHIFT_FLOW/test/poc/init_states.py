"""
Initial state preparation: Gaussian wavepacket with plane wave.
"""
import numpy as np
from numpy import pi, exp


def create_gaussian_wavepacket(nx, ny, sigma=0.8, x0=0.0, y0=0.0, k0x=2.0, k0y=1.0):
    """
    Create a Gaussian wavepacket with plane wave modulation.

    psi(x,y) ∝ exp(-((x-x0)^2+(y-y0)^2)/(4*sigma^2)) * exp(1j*(k0x*x + k0y*y))

    Args:
        nx, ny: Number of qubits for x and y dimensions
        sigma: Gaussian width
        x0, y0: Center position
        k0x, k0y: Wave vector components

    Returns:
        Normalized statevector of length 2*N*N where N=2^nx=2^ny
    """
    N = 2**nx
    assert 2**ny == N, "nx and ny must be equal for this POC"

    # Coordinates matching baseline convention
    x = np.linspace(-pi, pi, N, endpoint=False)
    y = np.linspace(-pi, pi, N, endpoint=False)
    X, Y = np.meshgrid(x, y)

    # Gaussian envelope with plane wave
    gaussian = exp(-((X - x0)**2 + (Y - y0)**2) / (4 * sigma**2))
    plane_wave = exp(1j * (k0x * X + k0y * Y))
    psi = gaussian * plane_wave

    # Normalize so sum(|psi|^2) = 1 on the grid
    psi = psi / np.sqrt(np.sum(np.abs(psi)**2))

    # Embed into full statevector: [psi, zeros] then flatten
    # Matches baseline's reshape convention tmp.reshape(2, N, N)
    psi_full = np.zeros((2, N, N), dtype='complex128')
    psi_full[0, :, :] = psi

    initial_state = psi_full.reshape(-1)

    return initial_state


def get_grid_coordinates(nx, ny):
    """Return the grid coordinates matching baseline convention."""
    N = 2**nx
    x = np.linspace(-pi, pi, N, endpoint=False)
    y = np.linspace(-pi, pi, N, endpoint=False)
    X, Y = np.meshgrid(x, y)
    return X, Y, x, y
