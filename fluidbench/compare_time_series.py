"""
Classic vs Quantum vs Quantum+ (占位) 五图对比，一行生成 PNG 到 figs/.
可直接 python compare_time_series.py 或在 Jupyter 单元运行本文件内容。
"""
import sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent if "__file__" in globals() \
       else pathlib.Path().cwd()
sys.path.insert(0, str(HERE))   # 保证同目录模块可 import

import numpy as np, matplotlib.pyplot as plt
from pathlib import Path

from stateprep      import single_vortex
from spec_classic   import SpectralSolver
from qspec_naive    import QuantumSpectralSolver
from qspec_pod      import QuantumPODSolver
from plot_fivepanel import five_panel

# ---------------- 参数 ----------------
T_GRID   = np.linspace(0, np.pi/2, 5)   # five snapshots
SIGMA    = 3.0
CENTER   = (0.0, 0.0)
OUT_DIR  = Path("figs"); OUT_DIR.mkdir(exist_ok=True)

# ---------------- 初态 ----------------
psi1_0, psi2_0 = single_vortex(SIGMA, CENTER)

# ---------------- 求解 ----------------
classic = SpectralSolver().evolve(psi1_0, psi2_0, T_GRID)
quantum = QuantumSpectralSolver().evolve(psi1_0, psi2_0, T_GRID)

try:
    q_plus = QuantumPODSolver().evolve(psi1_0, psi2_0, T_GRID)
except NotImplementedError:
    q_plus = quantum      # 先用 quantum 结果占位

# ---------------- 绘图 ----------------
for t in T_GRID:
    rho_c = np.abs(classic[t][0])**2 + np.abs(classic[t][1])**2
    rho_q = np.abs(quantum[t][0])**2 + np.abs(quantum[t][1])**2
    rho_p = np.abs(q_plus[t][0])**2 + np.abs(q_plus[t][1])**2

    fig = five_panel(rho_c, rho_q, rho_p, title=f"t = {t:.2f}")
    save_path = OUT_DIR / f"compare_t{t:.2f}.png"
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print("✓ saved", save_path)
