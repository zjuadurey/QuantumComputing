# rk4_realspace.py  ---------------------------------------------------------
# 高稳定性 RK4：自动把外层步长再细分，确保 |λ·dt_sub| ≤ safety
#   i ∂ψ/∂t = -½ ∇² ψ       (V = 0)
# 单文件即可 import，无包依赖。
# --------------------------------------------------------------------------
import numpy as np
from numpy.fft import fft2, ifft2
from stateprep import GRID_POW
# --------------------------------------------------------------------------
N  = 2 ** GRID_POW
k  = np.fft.fftfreq(N) * N
KX, KY = np.meshgrid(k, k, indexing='ij')
K2 = KX**2 + KY**2
lam_max = 0.5 * K2.max()                 # |λ|max = ½ k_max²

def _laplacian(field: np.ndarray) -> np.ndarray:
    return ifft2(-K2 * fft2(field))

# ---------------- 用户给出的改进版核心 ----------------
def evolve_rk4_baseline(psi0: np.ndarray,
                        t: float,
                        nsteps: int = 100,
                        safety: float = 2.5) -> np.ndarray:
    """
    RK4 evolution of i∂t ψ = -½∇²ψ on an NxN grid over total time t.
    Sub-steps so that |λ·dt_sub| ≤ safety (stability limit≈2.8).
    """
    dt_big = t / nsteps
    psi = psi0.copy()

    # --- 计算子步长 ---
    dt_sub = safety / lam_max
    n_sub  = max(1, int(np.ceil(dt_big / dt_sub)))
    dt_sub = dt_big / n_sub            # 整除外层步

    for _ in range(nsteps):
        for _ in range(n_sub):
            k1 = -0.5j * _laplacian(psi)
            k2 = -0.5j * _laplacian(psi + 0.5*dt_sub*k1)
            k3 = -0.5j * _laplacian(psi + 0.5*dt_sub*k2)
            k4 = -0.5j * _laplacian(psi +       dt_sub*k3)
            psi += dt_sub/6.0 * (k1 + 2*k2 + 2*k3 + k4)
    return psi
# --------------------------------------------------------------------------

class RK4Solver:
    """
    Wrapper so compare_time_series.py can call:

        rk4 = RK4Solver(nsteps=120, safety=2.5)
        result = rk4.evolve(psi1_0, psi2_0, t_list)
    """
    def __init__(self, nsteps: int = 100, safety: float = 2.5):
        self.nsteps = nsteps
        self.safety = safety

    def evolve(self, psi1_0: np.ndarray, psi2_0: np.ndarray, t_list):
        out = {}
        psi1, psi2 = psi1_0.copy(), psi2_0.copy()
        t_prev = t_list[0]
        out[t_prev] = (psi1.copy(), psi2.copy())

        for t in t_list[1:]:
            dt = t - t_prev
            psi1 = evolve_rk4_baseline(psi1, dt, self.nsteps, self.safety)
            psi2 = evolve_rk4_baseline(psi2, dt, self.nsteps, self.safety)
            out[t] = (psi1.copy(), psi2.copy())
            t_prev = t
        return out
