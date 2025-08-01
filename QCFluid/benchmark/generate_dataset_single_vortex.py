
# -*- coding: utf-8 -*-
"""
Generate single-vortex dataset:
  • classical spectral  （必生成）
  • quantum spectral    （若 ENABLE_QSPEC=True）
数据写入：<脚本所在目录>/dataset_root/single_vortex/…
"""
from pathlib import Path
import json, hashlib, numpy as np
from numpy import pi, sqrt, exp
from tqdm import tqdm
from solvers import (run_spec_step, run_quantum_step,
                     run_rk4_step,              # ← 新增
                     compute_fluid_quantities)


# ──────────────────────────────────────────────────────────────
# 全局参数
# ──────────────────────────────────────────────────────────────
N            = 2 ** 5          # 32 × 32 网格
# 时间步长
DT           = 0.1
# 时间步数
NSTEPS       = 100
# 数据集大小
NSAMPLES     = 100
SIGMA_MIN    = 2 * pi / N      # 一个格宽

# 输出目录：dataset_root/single_vortex（位于脚本同级）
ROOT = Path(__file__).resolve().parent / "dataset_root" / "single_vortex"
ROOT.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# 网格与初始态
# ──────────────────────────────────────────────────────────────
x = np.linspace(-pi, pi, N, endpoint=False)
X, Y = np.meshgrid(x, x, indexing="xy")

def init_wavefunction(x0, y0, sigma):
    R = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
    f = exp(-(R / sigma) ** 4)
    u = 2 * ((X - x0) + 1j * (Y - y0)) * f / (1 + R ** 2)
    v = 1j * (R ** 2 + 1 - 2 * f) / (1 + R ** 2)
    norm = sqrt(np.abs(u) ** 2 + np.abs(v) ** 4)
    return (u / norm).astype(np.complex128), (v ** 2 / norm).astype(np.complex128)

# ──────────────────────────────────────────────────────────────
# Stratified 采样：k×k 等分，每格随机 1 个中心
# ──────────────────────────────────────────────────────────────
def stratified_centres(rng, nsamples):
    k = int(np.ceil(np.sqrt(nsamples)))
    edges = -pi + np.linspace(0, 2 * pi, k + 1)
    cells = [(ix, iy) for ix in range(k) for iy in range(k)]
    rng.shuffle(cells)

    samples = []
    for ix, iy in cells:
        if len(samples) == nsamples:
            break
        for _ in range(20):                              # 最多重试 20 次找合格 σ_max
            x0 = rng.uniform(edges[ix], edges[ix + 1])
            y0 = rng.uniform(edges[iy], edges[iy + 1])
            sigma_max = min(x0 + pi, pi - x0, y0 + pi, pi - y0)
            if sigma_max > SIGMA_MIN:
                sigma = rng.uniform(SIGMA_MIN, sigma_max)
                samples.append((x0, y0, sigma))
                break
    return samples

# ──────────────────────────────────────────────────────────────
# 单方法完整时间序列
# ──────────────────────────────────────────────────────────────
def simulate(psi1_0, psi2_0, step_fn):
    Nt = NSTEPS + 1
    psi1 = np.empty((Nt, N, N), np.complex128); psi1[0] = psi1_0
    psi2 = np.empty_like(psi1);                 psi2[0] = psi2_0

    rho  = np.empty((Nt, N, N), np.float32)
    jx   = np.empty_like(rho)
    jy   = np.empty_like(rho)
    vort = np.empty_like(rho)
    rho[0], jx[0], jy[0], vort[0] = compute_fluid_quantities(psi1_0, psi2_0)

    for n in range(1, Nt):
        psi1[n], psi2[n] = step_fn(psi1[n - 1], psi2[n - 1], DT)
        rho[n], jx[n], jy[n], vort[n] = compute_fluid_quantities(psi1[n], psi2[n])

    return dict(rho=rho, jx=jx, jy=jy, vort=vort)

# ──────────────────────────────────────────────────────────────
# 写盘
# ──────────────────────────────────────────────────────────────
def dump(method, arrays, params):
    uid = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
    folder = ROOT / method / f"case_{uid}"
    folder.mkdir(parents=True, exist_ok=True)

    for name, arr in arrays.items():
        np.save(folder / f"{name}.npy", arr)
    with open(folder / "meta.json", "w") as fp:
        json.dump({**params, "dt": DT, "nsteps": NSTEPS}, fp, indent=2)

# ──────────────────────────────────────────────────────────────
# 主循环
# ──────────────────────────────────────────────────────────────
def main():
    rng = np.random.default_rng(42)

    for x0, y0, sigma in tqdm(stratified_centres(rng, NSAMPLES),
                              desc="Generating samples"):
        params = dict(x0=float(x0), y0=float(y0), sigma=float(sigma))
        psi1_0, psi2_0 = init_wavefunction(**params)

        # --- classical spectral ---
        dump("classical_spectral",
            simulate(psi1_0, psi2_0, run_spec_step),
            params)

        # --- quantum spectral（若启用） ---
        if run_quantum_step is not None:
            dump("quantum_spectral",
                simulate(psi1_0, psi2_0, run_quantum_step),
                params)

        # --- RK4 baseline ---
        dump("rk4",
            simulate(psi1_0, psi2_0, run_rk4_step),
            params)


if __name__ == "__main__":
    main()
