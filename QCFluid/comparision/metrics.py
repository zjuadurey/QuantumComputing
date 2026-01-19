# === metrics.py ===
import numpy as np

def compute_fidelity(psi1_q, psi2_q, psi1_c, psi2_c):
    psi_q = psi1_q.flatten() + 1j * psi2_q.flatten()
    psi_c = psi1_c.flatten() + 1j * psi2_c.flatten()
    psi_q /= np.linalg.norm(psi_q)
    psi_c /= np.linalg.norm(psi_c)
    return np.abs(np.vdot(psi_q, psi_c))**2

def compute_rho(psi1, psi2):
    return np.abs(psi1)**2 + np.abs(psi2)**2

def compute_relative_error(psi1_q, psi2_q, psi1_c, psi2_c):
    psi_q = psi1_q.flatten() + 1j * psi2_q.flatten()
    psi_c = psi1_c.flatten() + 1j * psi2_c.flatten()
    return np.linalg.norm(psi_q - psi_c) / np.linalg.norm(psi_c)

def compute_density_mse(psi1_q, psi2_q, psi1_c, psi2_c):
    rho_q = np.abs(psi1_q)**2 + np.abs(psi2_q)**2
    rho_c = np.abs(psi1_c)**2 + np.abs(psi2_c)**2
    diff = rho_q - rho_c
    diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
    return np.mean(diff**2)


def compute_norm_difference(psi1_q, psi2_q, psi1_c, psi2_c):
    norm_q = np.sum(np.abs(psi1_q)**2 + np.abs(psi2_q)**2)
    norm_c = np.sum(np.abs(psi1_c)**2 + np.abs(psi2_c)**2)
    return np.abs(norm_q - norm_c)

def compute_max_density_shift(psi1_q, psi2_q, psi1_c, psi2_c):
    rho_q = np.abs(psi1_q)**2 + np.abs(psi2_q)**2
    rho_c = np.abs(psi1_c)**2 + np.abs(psi2_c)**2
    idx_q = np.unravel_index(np.argmax(rho_q), rho_q.shape)
    idx_c = np.unravel_index(np.argmax(rho_c), rho_c.shape)
    return np.sqrt((idx_q[0] - idx_c[0])**2 + (idx_q[1] - idx_c[1])**2)

# spectrum diff
def compute_spectrum_l2_error(psi1_q, psi2_q, psi1_c, psi2_c):
    fft_q = np.fft.fft2(psi1_q + 1j*psi2_q)
    fft_c = np.fft.fft2(psi1_c + 1j*psi2_c)
    spec_q = np.abs(fft_q)
    spec_c = np.abs(fft_c)

    numerator = np.linalg.norm(spec_q - spec_c)
    denominator = np.linalg.norm(spec_c)

    if denominator == 0 or not np.isfinite(denominator):
        return 0.0  # or np.nan to flag unusable results
    return numerator / denominator

