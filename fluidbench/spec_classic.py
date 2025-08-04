# spec_classic.py  ─────────────────────────────────────────────
import numpy as np
from numpy.fft import fft2, ifft2
from stateprep import GRID_POW

N  = 2 ** GRID_POW
K  = np.fft.fftfreq(N) * N
K2 = (K[:, None]**2 + K[None, :]**2)      # |k|² 预计算一次

class SpectralSolver:
    """
    Fourier-spectral solver for V=0 Schrödinger.
    After each time step, the state is renormalised so that
        Σ (|ψ₁|²+|ψ₂|²) = 1
    matching the quantum-spectral solver’s convention.
    """
    def evolve(self, psi1_0, psi2_0, t_list):
        out  = {}
        psi1 = psi1_0.copy()
        psi2 = psi2_0.copy()

        last_t = t_list[0]
        out[last_t] = (psi1.copy(), psi2.copy())

        for t in t_list[1:]:
            dt    = t - last_t
            phase = np.exp(-0.5j * K2 * dt)

            psi1 = ifft2(phase * fft2(psi1))
            psi2 = ifft2(phase * fft2(psi2))

            # -------- 归一化 --------
            norm = np.sqrt((np.abs(psi1)**2 + np.abs(psi2)**2).sum())
            psi1 /= norm
            psi2 /= norm
            # ------------------------

            out[t] = (psi1.copy(), psi2.copy())
            last_t = t

        return out
