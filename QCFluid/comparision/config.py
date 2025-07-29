# === config.py ===
import numpy as np
from numpy import pi

N = 2**5
x = np.linspace(-pi, pi, N, endpoint=False)
y = np.linspace(-pi, pi, N, endpoint=False)
X, Y = np.meshgrid(x, y)

#t_values = [0.0, 0.1, 0.2, 0.3]
t_values = [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4]
sigma_values = np.random.uniform(0, np.pi / 2, size=4)
np.random.seed(42)
position_seeds = [
    (np.random.uniform(-pi/2, pi/2), np.random.uniform(-pi/2, pi/2))
    for _ in range(4)
]
