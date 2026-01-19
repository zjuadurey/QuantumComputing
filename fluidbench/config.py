"""
Centralised physical & numerical constants.
Edit here once, impact everywhere.
"""
import numpy as np
from numpy import pi

# ---------------- grid ----------------
GRID_POW   = 5                  # 2**GRID_POW points per axis
N          = 2 ** GRID_POW
DX         = 2 * pi / N
DY         = DX
KX         = np.fft.fftfreq(N) * N   # 常用频率数组
KY         = KX.copy()

# ---------------- default vortex ----------------
DEFAULT_SIGMA   = 0.25
DEFAULT_CENTER  = (0.39, 1.42)

# ---------------- misc ----------------
SEED = 42

PARAM_LIST = [
    (-0.39,  1.42, 0.18), (-0.39,  1.42, 0.25), (-0.39,  1.42, 0.54),
    (-1.08, -1.08, 0.18), (-1.08, -1.08, 0.25), (-1.08, -1.08, 0.54),
    (-1.39,  1.15, 0.18), (-1.39,  1.15, 0.25), (-1.39,  1.15, 0.54),
    ( 0.73,  0.31, 0.18), ( 0.73,  0.31, 0.25), ( 0.73,  0.31, 0.54),
]

T_GRID = np.linspace(0, 10, 11)  