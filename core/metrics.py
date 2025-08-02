"""
Generic metric suite (可自行扩展)：
    * mse_density
    * l2_relative
    * fidelity
"""
import numpy as np
from typing import Tuple

Ψ = Tuple[np.ndarray, np.ndarray]

def mse_density(ψa: Ψ, ψb: Ψ):
    ρa = np.abs(ψa[0])**2 + np.abs(ψa[1])**2
    ρb = np.abs(ψb[0])**2 + np.abs(ψb[1])**2
    return np.mean((ρa - ρb)**2)

def l2_relative(ψa: Ψ, ψb: Ψ):
    diff = np.linalg.norm(np.concatenate([ψa[0]-ψb[0], ψa[1]-ψb[1]]))
    ref  = np.linalg.norm(np.concatenate([ψb[0], ψb[1]]))
    return diff / ref

def fidelity(ψa: Ψ, ψb: Ψ):
    flat_a = np.concatenate([ψa[0].ravel(), ψa[1].ravel()])
    flat_b = np.concatenate([ψb[0].ravel(), ψb[1].ravel()])
    return np.abs(np.vdot(flat_a, flat_b))**2
