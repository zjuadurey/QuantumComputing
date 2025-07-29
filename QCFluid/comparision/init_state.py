# === init_state.py ===
import numpy as np
from config import X, Y, N

def generate_initial_state(x0, y0, sigma):
    Xs = X - x0
    Ys = Y - y0
    R = np.sqrt(Xs**2 + Ys**2)
    f = np.exp(-(R/sigma)**4)
    u = 2*(Xs + 1j*Ys)*f / (1 + R**2)
    v = 1j*(R**2 + 1 - 2*f) / (1 + R**2)
    psi1 = u / np.sqrt(np.abs(u)**2 + np.abs(v)**4)
    psi2 = v**2 / np.sqrt(np.abs(u)**2 + np.abs(v)**4)
    return psi1, psi2