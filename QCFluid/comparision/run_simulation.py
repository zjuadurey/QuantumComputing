# === run_simulation.py ===
import numpy as np
from init_state import generate_initial_state
from spectral_classic import evolve_spectral
from quantum_evolve import evolve_quantum
from metrics import compute_fidelity, compute_rho
from plot_utils import plot_density_diff
from config import N, sigma_values, position_seeds, t_values
import os
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

# 判断 sigma 是否超出周期边界（以中心为准）
def is_sigma_valid(x0, y0, sigma):
    r_max = np.sqrt((np.pi - abs(x0))**2 + (np.pi - abs(y0))**2)
    return 2 * sigma < r_max

# 保存 fidelity vs time 曲线图
fidelity_log = {}

for idx, (x0, y0) in enumerate(position_seeds):
    for sigma in sigma_values:
        if not is_sigma_valid(x0, y0, sigma):
            continue

        psi1_0, psi2_0 = generate_initial_state(x0, y0, sigma)
        initial_state = np.array([psi1_0, psi2_0]).reshape(-1)
        initial_state /= np.linalg.norm(initial_state)

        fidelity_list = []

        for t in t_values:
            psi1_c = evolve_spectral(psi1_0, t)
            psi2_c = evolve_spectral(psi2_0, t)
            psi1_q, psi2_q = evolve_quantum(5, 5, t, initial_state)

            rho_c = compute_rho(psi1_c, psi2_c)
            rho_q = compute_rho(psi1_q, psi2_q)

            scale = np.sqrt(np.sum(rho_c) / np.sum(rho_q))
            psi1_q *= scale
            psi2_q *= scale

            fidelity = compute_fidelity(psi1_q, psi2_q, psi1_c, psi2_c)
            fidelity_list.append(fidelity)
            print(f"Seed {idx} Sigma {sigma} Time {t:.2f} Fidelity: {fidelity:.5f}")

            plot_density_diff(rho_q, rho_c, f"results/diff_seed{idx}_sigma{sigma}_t{int(t*10)}.png")
            plot_vorticity_diff(psi1_q, psi2_q, psi1_c, psi2_c, f"results/vort_seed{idx}_sigma{sigma}_t{int(t*10)}.png")

        # 绘制曲线
        plt.plot(t_values, fidelity_list, label=f"Seed {idx} σ={sigma}")
        fidelity_log[(idx, sigma)] = fidelity_list

plt.xlabel("Time")
plt.ylabel("Fidelity")
plt.title("Fidelity vs Time")
plt.legend()
plt.tight_layout()
plt.savefig("results/fidelity_vs_time.png", dpi=150)
plt.close()