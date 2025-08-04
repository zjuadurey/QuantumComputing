"""
metrics.py
--------------------------------------------------------------------------
一站式指标库（Classic vs. 其它）——共 6 项：
    density_mse, vorticity_mse, fidelity,
    relative_L2, spectrum_L2, energy_spec_L2
所有函数签名:
    fn(psi1_q, psi2_q, psi1_c, psi2_c) -> float
--------------------------------------------------------------------------"""

import numpy as np
from numpy.fft import fft, fft2, ifft2
from compute_utils import compute_fluid_quantities, compute_vorticity
from numpy.fft import fft2, fftshift
# ------------------------------------------------------------
# 公共工具
# ------------------------------------------------------------
def _concat(psi1: np.ndarray, psi2: np.ndarray) -> np.ndarray:
    """把两分量拼接为 1-D 复向量"""
    return np.concatenate([psi1.flatten(), psi2.flatten()])

def _concat_norm(psi1, psi2):
    vec = _concat(psi1, psi2)
    return vec / np.linalg.norm(vec)

def _align_phase(vecA: np.ndarray, vecB: np.ndarray):
    """把 vecA 乘全局相位，使 ⟨A|B⟩ 实且非负"""
    phi = np.angle(np.vdot(vecA, vecB))
    return vecA * np.exp(-1j * phi)

def _rho(psi1, psi2):
    return np.abs(psi1)**2 + np.abs(psi2)**2

def _vorticity(psi1, psi2):
    N = psi1.shape[0]
    k = np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(k, k, indexing="ij")
    psi1_k, psi2_k = fft2(psi1), fft2(psi2)
    dpsi1_x = ifft2(1j*KX*psi1_k);  dpsi1_y = ifft2(1j*KY*psi1_k)
    dpsi2_x = ifft2(1j*KX*psi2_k);  dpsi2_y = ifft2(1j*KY*psi2_k)
    rho = _rho(psi1, psi2)
    ux  = (np.real(psi1)*np.imag(dpsi1_x) - np.imag(psi1)*np.real(dpsi1_x)
          +np.real(psi2)*np.imag(dpsi2_x) - np.imag(psi2)*np.real(dpsi2_x)) / rho
    uy  = (np.real(psi1)*np.imag(dpsi1_y) - np.imag(psi1)*np.real(dpsi1_y)
          +np.real(psi2)*np.imag(dpsi2_y) - np.imag(psi2)*np.real(dpsi2_y)) / rho
    vor = np.real(ifft2(1j*KX*fft2(uy) - 1j*KY*fft2(ux)))
    return vor

# ------------------------------------------------------------
# 1. Fidelity  (1 表示完全一致)
# ------------------------------------------------------------
def compute_fidelity(p1_q, p2_q, p1_c, p2_c):
    vq = _concat_norm(p1_q, p2_q)
    vc = _concat_norm(p1_c, p2_c)
    vq = _align_phase(vq, vc)
    return np.abs(np.vdot(vq, vc))**2

# ------------------------------------------------------------
# 2. Relative L2  (波函数向量差)
# ------------------------------------------------------------
def compute_relative_error(p1_q, p2_q, p1_c, p2_c):
    vq = _align_phase(_concat_norm(p1_q, p2_q),
                      _concat_norm(p1_c, p2_c))
    vc = _concat_norm(p1_c, p2_c)
    return np.linalg.norm(vq - vc)            # 分母=1

# ------------------------------------------------------------
# 3. Density MSE
# ------------------------------------------------------------
def compute_density_mse(p1_q, p2_q, p1_c, p2_c):
    return np.mean((_rho(p1_q, p2_q) - _rho(p1_c, p2_c))**2)

# ------------------------------------------------------------
# 4. Vorticity MSE
# ------------------------------------------------------------
def compute_vorticity_mse(p1_q, p2_q, p1_c, p2_c):
    return np.mean((_vorticity(p1_q, p2_q) - _vorticity(p1_c, p2_c))**2)

# ------------------------------------------------------------
# 5. Spectrum L2  (一维 FFT, 幅度谱差)
# ------------------------------------------------------------
def compute_spectrum_l2_error(ψ1_q, ψ2_q, ψ1_c, ψ2_c):
    """
    对两分量波函数分别做 2D FFT，得到能量谱 E = |FFT ψ1|^2 + |FFT ψ2|^2，
    然后将量子／经典的二维能量谱归一化并做 L2 差异。
    """
    # 1) 计算各分量的 2D FFT
    F1q = fft2(ψ1_q)
    F2q = fft2(ψ2_q)
    F1c = fft2(ψ1_c)
    F2c = fft2(ψ2_c)

    # 2) 频谱移中心
    F1q = fftshift(F1q)
    F2q = fftshift(F2q)
    F1c = fftshift(F1c)
    F2c = fftshift(F2c)

    # 3) 构造能量谱：两分量能量叠加
    Eq = np.abs(F1q)**2 + np.abs(F2q)**2
    Ec = np.abs(F1c)**2 + np.abs(F2c)**2

    # 4) 归一化（保证 ∑ Eq = ∑ Ec = 1）
    Eq /= Eq.sum()
    Ec /= Ec.sum()

    # 5) L2 差异
    return float(np.linalg.norm(Eq - Ec))

# ------------------------------------------------------------
# 6. Energy-spectrum L2  (ρ(k)² 差)
# ------------------------------------------------------------
def compute_energy_spec_l2(p1_q, p2_q, p1_c, p2_c):
    eq = np.abs(fft2(_rho(p1_q, p2_q)))**2
    ec = np.abs(fft2(_rho(p1_c, p2_c)))**2
    eq /= np.linalg.norm(eq);  ec /= np.linalg.norm(ec)
    return np.linalg.norm(eq - ec)

def compute_vorticity_l2_error(ψ1_q, ψ2_q, ψ1_c, ψ2_c):
    """
    Compute L2 relative error between quantum and classical vorticity fields.
    """

    def compute_vorticity(ψ1, ψ2):
        N = ψ1.shape[0]
        dx = 2 * np.pi / N
        p = np.abs(ψ1)**2 + np.abs(ψ2)**2

        def J_comp(ψ):
            grad_ψx = (np.roll(ψ, -1, axis=0) - np.roll(ψ, 1, axis=0)) / (2 * dx)
            grad_ψy = (np.roll(ψ, -1, axis=1) - np.roll(ψ, 1, axis=1)) / (2 * dx)
            return 0.5j * (ψ * np.conj(grad_ψx) - np.conj(ψ) * grad_ψx), \
                   0.5j * (ψ * np.conj(grad_ψy) - np.conj(ψ) * grad_ψy)

        Jx1, Jy1 = J_comp(ψ1)
        Jx2, Jy2 = J_comp(ψ2)
        Jx = Jx1 + Jx2
        Jy = Jy1 + Jy2

        ux = np.real(Jx / p)
        uy = np.real(Jy / p)

        dudy = (np.roll(ux, -1, axis=1) - np.roll(ux, 1, axis=1)) / (2 * dx)
        dvdx = (np.roll(uy, -1, axis=0) - np.roll(uy, 1, axis=0)) / (2 * dx)
        return dvdx - dudy

    ω_q = compute_vorticity(ψ1_q, ψ2_q)
    ω_c = compute_vorticity(ψ1_c, ψ2_c)

    return float(np.linalg.norm(ω_q - ω_c) / np.linalg.norm(ω_c))

def compute_vorticity_correlation(ψ1_q, ψ2_q, ψ1_c, ψ2_c):
    """
    Compute Pearson correlation between quantum and classical vorticity fields.
    """
    ω_q = compute_vorticity(ψ1_q, ψ2_q)
    ω_c = compute_vorticity(ψ1_c, ψ2_c)

    ω_q = ω_q.flatten()
    ω_c = ω_c.flatten()
    return float(np.corrcoef(ω_q, ω_c)[0, 1])


# new

import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_vorticity(ψ1, ψ2, dx=1.0):
    u = np.imag(np.conj(ψ1) * np.gradient(ψ1, axis=1))  # ∂ψ/∂x
    v = np.imag(np.conj(ψ2) * np.gradient(ψ2, axis=0))  # ∂ψ/∂y
    ω = np.gradient(v, axis=1) - np.gradient(u, axis=0)
    return ω / dx

def compute_vorticity_l2(ψ1_q, ψ2_q, ψ1_c, ψ2_c):
    ω_q = compute_vorticity(ψ1_q, ψ2_q)
    ω_c = compute_vorticity(ψ1_c, ψ2_c)
    return np.linalg.norm(ω_q - ω_c) / np.linalg.norm(ω_c)

def compute_vorticity_corr(ψ1_q, ψ2_q, ψ1_c, ψ2_c):
    ω_q = compute_vorticity(ψ1_q, ψ2_q)
    ω_c = compute_vorticity(ψ1_c, ψ2_c)
    return np.corrcoef(ω_q.flatten(), ω_c.flatten())[0, 1]

def compute_momentum_mse(ψ1_q, ψ2_q, ψ1_c, ψ2_c):
    def momentum(ψ1, ψ2):
        jx = np.imag(np.conj(ψ1) * np.gradient(ψ1, axis=1) +
                     np.conj(ψ2) * np.gradient(ψ2, axis=1))
        jy = np.imag(np.conj(ψ1) * np.gradient(ψ1, axis=0) +
                     np.conj(ψ2) * np.gradient(ψ2, axis=0))
        return jx, jy
    jqx, jqy = momentum(ψ1_q, ψ2_q)
    jcx, jcy = momentum(ψ1_c, ψ2_c)
    return np.mean((jqx - jcx)**2 + (jqy - jcy)**2)

def compute_mass_diff(ψ1_q, ψ2_q, ψ1_c, ψ2_c):
    ρ_q = np.abs(ψ1_q)**2 + np.abs(ψ2_q)**2
    ρ_c = np.abs(ψ1_c)**2 + np.abs(ψ2_c)**2
    return abs(np.sum(ρ_q) - np.sum(ρ_c))

def compute_com_error(ψ1_q, ψ2_q, ψ1_c, ψ2_c):
    ρ_q = np.abs(ψ1_q)**2 + np.abs(ψ2_q)**2
    ρ_c = np.abs(ψ1_c)**2 + np.abs(ψ2_c)**2
    N = ρ_q.shape[0]
    x = np.arange(N)
    y = np.arange(N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    def com(ρ): return (np.sum(X * ρ) / np.sum(ρ), np.sum(Y * ρ) / np.sum(ρ))
    xq, yq = com(ρ_q)
    xc, yc = com(ρ_c)
    return np.sqrt((xq - xc)**2 + (yq - yc)**2)


