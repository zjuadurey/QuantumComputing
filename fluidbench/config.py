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
DEFAULT_SIGMA   = 3.0
DEFAULT_CENTER  = (0.0, 0.0)

# ---------------- misc ----------------
SEED = 42
