"""
Generate initial two–component wave-function for a single ideal vortex.
"""
import numpy as np
from numpy import sqrt, abs, exp, pi

# ------- grid helpers -------
GRID_POW = 5           # 2**5 = 32 points per axis
N        = 2 ** GRID_POW

def mesh():
    x = np.linspace(-pi, pi, N, endpoint=False)
    y = np.linspace(-pi, pi, N, endpoint=False)
    return np.meshgrid(x, y, indexing="ij")

# ------- vortex state -------
def single_vortex(sigma: float = 3.0, center=(0.0, 0.0)):
    X, Y = mesh()
    x0, y0 = center
    R  = np.sqrt((X - x0)**2 + (Y - y0)**2)
    f  = exp(-(R / sigma) ** 4)
    u  = 2 * (X + 1j * Y) * f / (1 + R**2)
    v  = 1j * (R**2 + 1 - 2 * f) / (1 + R**2)
    psi1 = u / sqrt(abs(u)**2 + abs(v)**4)
    psi2 = v**2 / sqrt(abs(u)**2 + abs(v)**4)
    return psi1, psi2
