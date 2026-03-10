"""
plot_v0_sanity.py — V=0 sanity check: density field visualization (1×4 figure*).

Pure numpy/FFT computation, no qiskit dependency.
Panels: (a) Full ρ, (b) Low-pass ρ, (c) Shadow ρ, (d) Shadow − Low-pass.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

OUT = Path(__file__).resolve().parent

# ── Style (match plot_v1_all.py) ──────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "axes.labelweight": "normal",
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# ── Parameters ────────────────────────────────────────────────────────────
N = 64          # grid size (2^6)
t = 0.30 * np.pi
K0 = 6          # low-frequency cutoff
sigma = 3.0     # vortex width


def vortex_initial_condition(N, sigma=3.0):
    """Two-component vortex initial state (same as shadow_test_v4.py)."""
    x = np.linspace(-np.pi, np.pi, N, endpoint=False)
    y = np.linspace(-np.pi, np.pi, N, endpoint=False)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    f = np.exp(-(R / sigma) ** 4)
    u = 2 * (X + 1j * Y) * f / (1 + R**2)
    v = 1j * (R**2 + 1 - 2 * f) / (1 + R**2)
    psi1 = u / np.sqrt(np.abs(u)**2 + np.abs(v)**4)
    psi2 = v**2 / np.sqrt(np.abs(u)**2 + np.abs(v)**4)
    stacked = np.array([psi1, psi2]).reshape(-1).astype(np.complex128)
    stacked /= np.linalg.norm(stacked)
    psi1_n = stacked[:N*N].reshape(N, N)
    psi2_n = stacked[N*N:].reshape(N, N)
    return psi1_n, psi2_n


def unitary_fft2(f):
    return np.fft.fft2(f) / f.shape[0]

def unitary_ifft2(f):
    return np.fft.ifft2(f) * f.shape[0]


def main():
    # Grids
    kx = np.fft.fftfreq(N) * N
    ky = np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(kx, ky)
    E = 0.5 * (KX**2 + KY**2)          # free-particle energy
    mask = (KX**2 + KY**2) <= K0**2     # low-freq set

    # Initial condition
    psi1_0, psi2_0 = vortex_initial_condition(N, sigma)

    # ── Full evolution (analytical, V=0) ──
    b1_0 = unitary_fft2(psi1_0)
    b2_0 = unitary_fft2(psi2_0)
    phase = np.exp(-1j * E * t)
    b1_full = b1_0 * phase
    b2_full = b2_0 * phase
    psi1_full = unitary_ifft2(b1_full)
    psi2_full = unitary_ifft2(b2_full)
    rho_full = np.abs(psi1_full)**2 + np.abs(psi2_full)**2

    # ── Low-pass baseline ──
    b1_lp = np.where(mask, b1_full, 0.0)
    b2_lp = np.where(mask, b2_full, 0.0)
    psi1_lp = unitary_ifft2(b1_lp)
    psi2_lp = unitary_ifft2(b2_lp)
    rho_lp = np.abs(psi1_lp)**2 + np.abs(psi2_lp)**2

    # ── Shadow evolution (coherences, V=0) ──
    def shadow_evolve(b0, mask, E, t):
        k0_idx = (0, 0)
        E0 = E[k0_idx]
        b_k0_0 = b0[k0_idx]
        z0 = b0 * np.conj(b_k0_0)
        zt = z0 * np.exp(-1j * (E - E0) * t)
        b_k0_t = b_k0_0 * np.exp(-1j * E0 * t)
        b_t = np.zeros_like(b0)
        b_t[mask] = zt[mask] / np.conj(b_k0_t)
        b_t[k0_idx] = b_k0_t
        return b_t

    b1_sh = shadow_evolve(b1_0, mask, E, t)
    b2_sh = shadow_evolve(b2_0, mask, E, t)
    psi1_sh = unitary_ifft2(b1_sh)
    psi2_sh = unitary_ifft2(b2_sh)
    rho_sh = np.abs(psi1_sh)**2 + np.abs(psi2_sh)**2

    # Difference
    diff = rho_sh - rho_lp

    # Metrics
    def rel_l2(a, b):
        return np.linalg.norm(a - b) / np.linalg.norm(b)
    print(f"Shadow vs LP:   rel L2 = {rel_l2(rho_sh, rho_lp):.2e}")
    print(f"Shadow vs Full: rel L2 = {rel_l2(rho_sh, rho_full):.2e}")
    print(f"LP vs Full:     rel L2 = {rel_l2(rho_lp, rho_full):.2e}")

    # ── Plot 1×4 ──
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    y = np.linspace(0, 2*np.pi, N, endpoint=False)
    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.8))

    panels = [
        (rho_full, r"(a) Full $\rho$", None),
        (rho_lp,   rf"(b) Low-pass $\rho$ ($K_0\!=\!{K0}$)", None),
        (rho_sh,   r"(c) Shadow $\rho$", None),
        (diff,     r"(d) Shadow $-$ Low-pass", None),
    ]

    # Shared color range for (a)-(c)
    vmin_rho = min(rho_full.min(), rho_lp.min(), rho_sh.min())
    vmax_rho = max(rho_full.max(), rho_lp.max(), rho_sh.max())

    for i, (data, title, cmap) in enumerate(panels):
        ax = axes[i]
        im = ax.pcolormesh(X, Y, data, shading="auto",
                           vmin=vmin_rho, vmax=vmax_rho, cmap="viridis")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=9, pad=3)
        if i == 3:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=7)

    fig.tight_layout(w_pad=0.5)
    fig.savefig(OUT / "v0_sanity_density.pdf")
    print("Saved v0_sanity_density.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
