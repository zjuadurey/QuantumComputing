"""
Very small Fourier-spectral solver for the V=0 Schrödinger (hydrodynamic) equation.
"""
import numpy as np
from numpy.fft import fft2, ifft2
from stateprep import GRID_POW

N   = 2 ** GRID_POW
K   = np.fft.fftfreq(N) * N                # 0,1,2,…,-1

class SpectralSolver:
    def evolve(self, psi1_0, psi2_0, t_list):
        out = {}
        K2   = (K[:, None]**2 + K[None, :]**2)
        psi1, psi2 = psi1_0.copy(), psi2_0.copy()

        last_t = t_list[0]
        out[last_t] = (psi1.copy(), psi2.copy())

        for t in t_list[1:]:
            dt = t - last_t
            phase = np.exp(-0.5j * K2 * dt)

            psi1 = ifft2(phase * fft2(psi1))
            psi2 = ifft2(phase * fft2(psi2))
            out[t] = (psi1.copy(), psi2.copy())
            last_t = t
        return out
