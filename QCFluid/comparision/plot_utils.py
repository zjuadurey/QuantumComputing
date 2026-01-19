# === plot_utils.py ===
import matplotlib.pyplot as plt
import numpy as np
import os
from config import N

def show_density_comparison(rho_q, rho_c, title="Density Comparison", save_path=None):
    """
    Display classic, quantum, and difference density side-by-side.
    Parameters:
        rho_q: 2D array (quantum result)
        rho_c: 2D array (classical result)
        title: string, title for the full figure
        save_path: if given, save the figure to this file
    """
    diff = rho_q - rho_c

    fig, axs = plt.subplots(1, 3, figsize=(12, 3))

    im0 = axs[0].pcolormesh(rho_c, cmap='viridis', shading='auto')
    axs[0].set_title('Classic')
    axs[0].axis('off')
    fig.colorbar(im0, ax=axs[0])

    im1 = axs[1].pcolormesh(rho_q, cmap='viridis', shading='auto')
    axs[1].set_title('Quantum')
    axs[1].axis('off')
    fig.colorbar(im1, ax=axs[1])

    im2 = axs[2].pcolormesh(diff.real, cmap='seismic', shading='auto')
    axs[2].set_title('Difference')
    axs[2].axis('off')
    fig.colorbar(im2, ax=axs[2])

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()


# === 计算涡度 ===
def compute_vorticity(psi1, psi2):
    kx = np.fft.fftfreq(N)*N
    ky = np.fft.fftfreq(N)*N
    KX, KY = np.meshgrid(kx, ky)
    dpsi1_x = np.fft.ifft2(1j*KX * np.fft.fft2(psi1))
    dpsi1_y = np.fft.ifft2(1j*KY * np.fft.fft2(psi1))
    dpsi2_x = np.fft.ifft2(1j*KX * np.fft.fft2(psi2))
    dpsi2_y = np.fft.ifft2(1j*KY * np.fft.fft2(psi2))
    rho = np.abs(psi1)**2 + np.abs(psi2)**2
    ux = (np.real(psi1)*np.imag(dpsi1_x) - np.imag(psi1)*np.real(dpsi1_x)
        + np.real(psi2)*np.imag(dpsi2_x) - np.imag(psi2)*np.real(dpsi2_x)) / rho
    uy = (np.real(psi1)*np.imag(dpsi1_y) - np.imag(psi1)*np.real(dpsi1_y)
        + np.real(psi2)*np.imag(dpsi2_y) - np.imag(psi2)*np.real(dpsi2_y)) / rho
    vort = np.fft.ifft2(1j*KX*np.fft.fft2(uy) - 1j*KY*np.fft.fft2(ux))
    return np.real(vort)

# === 显示拼图函数 ===
def show_vorticity_comparison(psi1_q, psi2_q, psi1_c, psi2_c, title):
    vort_q = compute_vorticity(psi1_q, psi2_q)
    vort_c = compute_vorticity(psi1_c, psi2_c)
    diff = vort_q - vort_c

    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    im0 = axs[0].pcolormesh(vort_c, cmap='RdBu', shading='auto')
    axs[0].set_title('Classic')
    axs[0].axis('off')
    fig.colorbar(im0, ax=axs[0])

    im1 = axs[1].pcolormesh(vort_q, cmap='RdBu', shading='auto')
    axs[1].set_title('Quantum')
    axs[1].axis('off')
    fig.colorbar(im1, ax=axs[1])

    im2 = axs[2].pcolormesh(diff, cmap='seismic', shading='auto')
    axs[2].set_title('Difference')
    axs[2].axis('off')
    fig.colorbar(im2, ax=axs[2])

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()