"""shiftflow/qiskit_shadow_v1.py

Qiskit-based implementation of SHIFT-FLOW "shadow" evolution for V!=0.

For V!=0, the Hamiltonian H_K (and H_R) is a *dense* Hermitian matrix on the
truncated subspace, so the time-evolution operator exp(-i H_sub t) is a general
unitary (not diagonal).  We use qiskit.extensions.UnitaryGate to apply it on
the mode register inside an Aer statevector simulation.

Encoding (same spinor convention as V=0):
  - q_mode = ceil(log2(M)) qubits for compact mode index m in [0, M)
  - 1 additional qubit for spinor component sigma in {0,1} (MSB)
  - Basis: index = m + (sigma << q_mode)
  - Padded entries (m >= M) carry identity evolution
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.linalg import expm


def pack_subspace_sv(
    b1_sub: np.ndarray,
    b2_sub: np.ndarray,
) -> tuple[np.ndarray, float, int]:
    """Pack subspace coefficients into a spinor statevector.

    Parameters
    ----------
    b1_sub, b2_sub : complex arrays of length M (one per spinor component)

    Returns
    -------
    sv : normalised statevector of length 2 * 2^q_mode
    scale : norm before normalisation
    q_mode : number of mode-register qubits
    """
    M = len(b1_sub)
    if M <= 0:
        raise ValueError("Empty subspace (M=0)")

    q_mode = int(math.ceil(math.log2(M))) if M > 1 else 1
    dim_mode = 1 << q_mode

    a1 = np.zeros(dim_mode, dtype=np.complex128)
    a2 = np.zeros(dim_mode, dtype=np.complex128)
    a1[:M] = b1_sub
    a2[:M] = b2_sub

    sv = np.concatenate([a1, a2])  # sigma=0 block, then sigma=1 block
    scale = float(np.linalg.norm(sv))
    if scale == 0.0:
        raise ValueError("Subspace state has zero norm")
    sv = sv / scale
    return sv, scale, q_mode


def unpack_subspace_sv(
    sv: np.ndarray,
    M: int,
    q_mode: int,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Unpack a spinor statevector back to subspace coefficient vectors.

    Returns
    -------
    b1_sub, b2_sub : complex arrays of length M
    """
    dim_mode = 1 << q_mode
    a1 = sv[:dim_mode] * scale
    a2 = sv[dim_mode:] * scale
    return a1[:M].copy(), a2[:M].copy()


def build_padded_unitary(
    H_sub: np.ndarray,
    t: float,
) -> np.ndarray:
    """Build padded unitary for mode register.

    Computes U = expm(-i H_sub t)  (M x M)
    and embeds it as the top-left block of a 2^q_mode x 2^q_mode unitary
    with identity on the padding subspace.
    """
    M = H_sub.shape[0]
    U = expm(-1j * H_sub * t)

    q_mode = int(math.ceil(math.log2(M))) if M > 1 else 1
    dim_mode = 1 << q_mode

    if dim_mode == M:
        return U

    U_padded = np.eye(dim_mode, dtype=np.complex128)
    U_padded[:M, :M] = U
    return U_padded


def evolve_subspace_aer(
    sv0: np.ndarray,
    H_sub: np.ndarray,
    t: float,
    q_mode: int,
    *,
    sim=None,
) -> tuple[np.ndarray, float, float]:
    """Evolve a packed spinor statevector via Aer statevector simulation.

    The circuit applies UnitaryGate(U_padded) on the mode-register qubits
    (q0 .. q_{q_mode-1}), leaving the spinor qubit (q_{q_mode}) untouched.

    Returns
    -------
    sv_t : evolved statevector (length 2 * 2^q_mode)
    rt_transpile : transpilation wall-clock time (seconds)
    rt_run : simulation wall-clock time (seconds)
    """
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit.library import UnitaryGate
        from qiskit_aer import AerSimulator
    except Exception as e:  # pragma: no cover
        raise ImportError("qiskit-aer is required for Aer shadow evolution") from e

    U_padded = build_padded_unitary(H_sub, t)

    q_total = q_mode + 1  # mode register + spinor qubit
    circ = QuantumCircuit(q_total)
    circ.initialize(np.asarray(sv0, dtype=np.complex128), list(range(q_total)))
    circ.append(UnitaryGate(U_padded), qargs=list(range(q_mode)))
    circ.save_statevector()

    if sim is None:
        sim = AerSimulator(method="statevector")

    t0 = time.perf_counter()
    circ_t = transpile(circ, sim, optimization_level=0)
    rt_transpile = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = sim.run(circ_t).result()
    rt_run = time.perf_counter() - t0

    sv_t = np.asarray(result.data(0)["statevector"], dtype=np.complex128)
    return sv_t, float(rt_transpile), float(rt_run)


def shadow_evolve_v1(
    b1_K_0: np.ndarray,
    b2_K_0: np.ndarray,
    b1_R_0: np.ndarray,
    b2_R_0: np.ndarray,
    H_K: np.ndarray,
    H_R: np.ndarray,
    t: float,
    *,
    sim=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Shadow evolution for V!=0 via Qiskit Aer.

    Evolves K-subspace and R-subspace independently through their
    respective truncated Hamiltonians using UnitaryGate circuits.

    Parameters
    ----------
    b1_K_0, b2_K_0 : initial K-subspace coefficients (length M_K)
    b1_R_0, b2_R_0 : initial R-subspace coefficients (length M_R)
    H_K : Hamiltonian restricted to K-subspace (M_K x M_K)
    H_R : Hamiltonian restricted to R-subspace (M_R x M_R)
    t : evolution time
    sim : optional shared AerSimulator instance

    Returns
    -------
    b1_K_t, b2_K_t : evolved K-subspace coefficients
    b1_R_t, b2_R_t : evolved R-subspace coefficients
    rt_K_s : total Qiskit wall-clock time for K circuit (transpile + run)
    rt_R_s : total Qiskit wall-clock time for R circuit (transpile + run)
    """
    M_K = len(b1_K_0)
    M_R = len(b1_R_0)

    # --- K subspace ---
    sv_K_0, scale_K, q_mode_K = pack_subspace_sv(b1_K_0, b2_K_0)
    sv_K_t, rt_K_trans, rt_K_run = evolve_subspace_aer(
        sv_K_0, H_K, t, q_mode_K, sim=sim,
    )
    b1_K_t, b2_K_t = unpack_subspace_sv(sv_K_t, M_K, q_mode_K, scale_K)

    # --- R subspace ---
    sv_R_0, scale_R, q_mode_R = pack_subspace_sv(b1_R_0, b2_R_0)
    sv_R_t, rt_R_trans, rt_R_run = evolve_subspace_aer(
        sv_R_0, H_R, t, q_mode_R, sim=sim,
    )
    b1_R_t, b2_R_t = unpack_subspace_sv(sv_R_t, M_R, q_mode_R, scale_R)

    rt_K_s = rt_K_trans + rt_K_run
    rt_R_s = rt_R_trans + rt_R_run

    return b1_K_t, b2_K_t, b1_R_t, b2_R_t, rt_K_s, rt_R_s
