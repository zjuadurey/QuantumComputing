# qspec_pod.py  -------------------------------------------------------------
# Improved “quantum” solver: 频域截断 + 低维对角演化
# 接口与 qspec_naive.QuantumSpectralSolver 保持一致：
#     evolve(psi1_0, psi2_0, t_list)  →  {t: (psi1, psi2)}
#
# 依赖：
#   numpy, qiskit, qiskit-aer     （和 qspec_naive 相同）
# --------------------------------------------------------------------------
from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np
from numpy.fft import fft2, ifft2
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import Diagonal
from stateprep import GRID_POW                       # 与其他模块共用
N = 2 ** GRID_POW

# ———————————————————————— 工具函数 ————————————————————————
def _auto_select_K(psi1_k: np.ndarray,
                   psi2_k: np.ndarray,
                   ratio: float = 0.99) -> int:
    """最小 K 使累计能量 ≥ ratio·总能量"""
    energy   = np.abs(psi1_k)**2 + np.abs(psi2_k)**2
    total_E  = energy.sum()
    kx = np.fft.fftfreq(N) * N
    ky = kx.copy()
    KX, KY = np.meshgrid(kx, ky)
    k_abs = np.sqrt(KX**2 + KY**2)
    for K in range(1, N // 2):
        if energy[k_abs <= K].sum() / total_E >= ratio:
            return K
    return N // 2 - 1

def _make_mask(K: int,
               window: str = "cross",
               include_edges: bool = False) -> List[Tuple[int, int]]:
    """返回被保留的 (kx, ky) 频率索引列表"""
    mask: List[Tuple[int, int]] = []

    # 周期边界 Nyquist 频率
    if include_edges:
        for ix in (-N//2, 0, N//2):
            for iy in (-N//2, 0, N//2):
                if not (ix == iy == 0):
                    mask.append((ix, iy))

    # 中心窗口
    for ix in range(-K, K + 1):
        for iy in range(-K, K + 1):
            cond = (
                (window == "cross"   and (abs(ix) <= K or abs(iy) <= K)) or
                (window == "square"  and (abs(ix) <= K and abs(iy) <= K)) or
                (window == "diamond" and (abs(ix) + abs(iy) <= K))
            )
            if cond and (ix, iy) not in mask:
                mask.append((ix, iy))
    return mask

def _evolve_trunc(psi1: np.ndarray,
                  psi2: np.ndarray,
                  dt: float,
                  mask: List[Tuple[int, int]],
                  backend: AerSimulator):
    """
    单步 dt 频域截断演化；返回 (psi1_t, psi2_t)
    """
    # ----- 1. 取截断系数 -----
    psi1_k, psi2_k = fft2(psi1), fft2(psi2)
    coeff_list = []
    for ix, iy in mask: coeff_list.append(psi1_k[(iy+N)%N, (ix+N)%N])
    for ix, iy in mask: coeff_list.append(psi2_k[(iy+N)%N, (ix+N)%N])
    coeff_raw   = np.array(coeff_list, dtype=complex)
    orig_norm   = np.linalg.norm(coeff_raw)
    coeff_norm  = coeff_raw / orig_norm

    r_modes = len(mask) * 2
    q       = int(np.ceil(np.log2(r_modes)))
    dim     = 1 << q

    state   = np.zeros(dim, dtype=complex)
    state[:r_modes] = coeff_norm

    # ----- 2. 对角相位 -----
    kx_modes = [ix for ix, _ in mask] * 2
    ky_modes = [iy for _, iy in mask] * 2
    phase = np.exp(-0.5j * dt * (np.array(kx_modes)**2
                                 + np.array(ky_modes)**2))
    diag = np.ones(dim, dtype=complex); diag[:r_modes] = phase

    # ----- 3. 构电路并跑 -----
    qc = QuantumCircuit(q)
    qc.initialize(state, qc.qubits)
    qc.append(Diagonal(diag), qc.qubits)
    qc.save_state()
    qc = transpile(qc, backend, optimization_level=0)
    sv = backend.run(qc).result().get_statevector(qc)

    # ----- 4. 写回截断谱，再 ifft -----
    psi1_k_new = np.zeros_like(psi1_k)
    psi2_k_new = np.zeros_like(psi2_k)
    m = 0
    for ix, iy in mask:
        psi1_k_new[(iy+N)%N, (ix+N)%N] = sv[m]; m += 1
    for ix, iy in mask:
        psi2_k_new[(iy+N)%N, (ix+N)%N] = sv[m]; m += 1

    psi1_t = ifft2(psi1_k_new) * orig_norm
    psi2_t = ifft2(psi2_k_new) * orig_norm

    # 归一化到总概率 1
    total_norm = np.sqrt((np.abs(psi1_t)**2 + np.abs(psi2_t)**2).sum())
    psi1_t /= total_norm; psi2_t /= total_norm
    return psi1_t, psi2_t
# ————————————————————————————————————————————————————————————————

class QuantumPODSolver:
    """
    频域截断 (≥ratio 能量) + 低维对角演化。
    与 naive 量子谱方法同接口：evolve(...)
    """
    def __init__(self,
                 energy_ratio: float = 0.99,
                 window: str = "cross",
                 include_edges: bool = False):
        self.energy_ratio  = energy_ratio
        self.window        = window
        self.include_edges = include_edges
        self.backend       = AerSimulator(method="statevector")

    def evolve(self,
               psi1_0: np.ndarray,
               psi2_0: np.ndarray,
               t_list: np.ndarray | list[float]
               ) -> Dict[float, tuple[np.ndarray, np.ndarray]]:

        # -------- 0. 预处理：确定截断窗口 --------
        psi1_k0, psi2_k0 = fft2(psi1_0), fft2(psi2_0)
        K_cut = _auto_select_K(psi1_k0, psi2_k0, self.energy_ratio)
        mask  = _make_mask(K_cut, self.window, self.include_edges)

        out: Dict[float, tuple[np.ndarray, np.ndarray]] = {}
        psi1, psi2 = psi1_0.copy(), psi2_0.copy()
        t_prev = t_list[0]
        out[t_prev] = (psi1.copy(), psi2.copy())

        for t in t_list[1:]:
            dt  = t - t_prev
            psi1, psi2 = _evolve_trunc(
                psi1, psi2, dt, mask, self.backend)
            out[t] = (psi1.copy(), psi2.copy())
            t_prev = t
        return out
