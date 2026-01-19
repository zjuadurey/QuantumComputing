# === config.py ===
import numpy as np
from numpy import pi

N = 2**5
x = np.linspace(-pi, pi, N, endpoint=False)
y = np.linspace(-pi, pi, N, endpoint=False)
X, Y = np.meshgrid(x, y)


# ---------- 时间步 ----------
#t_values = [0.0, 0.1, 0.2, 0.3]
t_values = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

# ---------- 工具函数 ----------
def is_sigma_valid(x0, y0, sigma):
    """
    判断高斯核直径 (≈2σ) 是否落在 [−π, π) 周期边界内
    """
    r_max = np.sqrt((pi - abs(x0))**2 + (pi - abs(y0))**2)
    return 2 * sigma < r_max

# ---------- 抽样 4 组合法参数 ----------
np.random.seed(42)           # 结果可复现
param_sets = []              # [(x0, y0, sigma), ...]
while len(param_sets) < 4:
    sigma = np.random.uniform(0, pi)       # 0 – π
    x0    = np.random.uniform(-pi, pi)     # −π – π
    y0    = np.random.uniform(-pi, pi)
    if is_sigma_valid(x0, y0, sigma):
        param_sets.append((x0, y0, sigma))

# 拆分成两个列表，保持与旧脚本接口兼容
position_seeds = [(x0, y0) for x0, y0, _ in param_sets]
sigma_values   = [sigma     for _,  _,  sigma in param_sets]
