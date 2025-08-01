"""
Mesh helpers and FFT shortcuts (all purely NumPy).
"""
import numpy as np
from numpy import pi
from .config import N

def mesh() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-pi, pi, N, endpoint=False)
    y = np.linspace(-pi, pi, N, endpoint=False)
    return np.meshgrid(x, y, indexing="ij")
