"""shiftflow/qiskit_shadow_v0.py

Qiskit-based implementation of SHIFT-FLOW "shadow" evolution for V=0.

Important:
- This is NOT classical shadow tomography.
- For V=0, the low-frequency Fourier subspace is invariant. The shadow evolution
  therefore reduces to time evolution of the truncated Fourier-mode state.

Encoding used here (compressed / HT-style):
- Let S be the kept k-modes (mask) with size M.
- Allocate q_mode = ceil(log2(M)) qubits for a compact mode index m in [0, M).
- Allocate one additional qubit for the component (spin) sigma in {0,1}.
- Basis ordering follows Qiskit convention with sigma as the MSB:
    index = m + (sigma << q_mode)

We simulate the shadow evolution in Qiskit by applying a diagonal phase gate on
the mode register:
  |m> -> exp(-i E_m t) |m>
and leaving the spin qubit untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import time

import numpy as np

from shiftflow import core_v0


@dataclass(frozen=True)
class ShadowEncoding:
    N: int
    K0: float
    M: int
    q_mode: int
    q_total: int
    order: str


def modes_from_mask(mask: np.ndarray, E: np.ndarray, order: str = "energy") -> tuple[list[tuple[int, int]], np.ndarray]:
    """Return a deterministic list of kept modes and their energies.

    Parameters
    - mask: boolean array (N,N)
    - E: energy grid (N,N)
    - order:
        - "energy": increasing E, tie-break by (iy,ix)
        - "yx": lexicographic by (iy,ix)
    """
    idx = np.argwhere(mask)
    if idx.size == 0:
        return [], np.zeros((0,), dtype=float)

    modes = [(int(iy), int(ix)) for iy, ix in idx]
    if order == "yx":
        modes.sort(key=lambda t: (t[0], t[1]))
    elif order == "energy":
        modes.sort(key=lambda t: (float(E[t[0], t[1]]), t[0], t[1]))
    else:
        raise ValueError(f"Unknown order: {order}")

    energies = np.array([float(E[iy, ix]) for iy, ix in modes], dtype=float)
    return modes, energies


def pack_truncated_statevector(
    b1_0: np.ndarray,
    b2_0: np.ndarray,
    modes: list[tuple[int, int]],
    *,
    normalize: bool = True,
) -> tuple[np.ndarray, float, int]:
    """Pack truncated Fourier coefficients into a (spin ⊗ mode) statevector.

    Returns:
      (sv, scale, q_mode)
    where:
      - sv has length 2^(q_mode+1)
      - scale = sqrt(sum_{modes} (|b1|^2 + |b2|^2))
        (so multiplying the unpacked normalized amplitudes by scale recovers the
        original coefficients on the kept subspace)
    """
    M = int(len(modes))
    if M <= 0:
        raise ValueError("Empty mode set (M=0)")

    q_mode = int(math.ceil(math.log2(M))) if M > 1 else 0
    dim_mode = 1 << q_mode
    dim_total = 2 * dim_mode

    a1 = np.zeros((dim_mode,), dtype=np.complex128)
    a2 = np.zeros((dim_mode,), dtype=np.complex128)
    for m, (iy, ix) in enumerate(modes):
        a1[m] = b1_0[iy, ix]
        a2[m] = b2_0[iy, ix]

    sv = np.concatenate([a1, a2])
    scale = float(np.linalg.norm(sv))
    if normalize:
        if scale == 0.0:
            raise ValueError("Truncated state has zero norm")
        sv = sv / scale
    return sv, scale, q_mode


def unpack_truncated_statevector(
    sv: np.ndarray,
    modes: list[tuple[int, int]],
    *,
    N: int,
    q_mode: int,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Unpack a (spin ⊗ mode) statevector back into full (N,N) coefficient arrays."""
    M = int(len(modes))
    dim_mode = 1 << int(q_mode)
    if sv.shape[0] != 2 * dim_mode:
        raise ValueError("Statevector length does not match q_mode")

    a1 = sv[:dim_mode] * complex(scale)
    a2 = sv[dim_mode:] * complex(scale)

    b1 = np.zeros((N, N), dtype=np.complex128)
    b2 = np.zeros((N, N), dtype=np.complex128)
    for m, (iy, ix) in enumerate(modes):
        b1[iy, ix] = a1[m]
        b2[iy, ix] = a2[m]
    return b1, b2


def unpack_truncated_statevector_into(
    b1_out: np.ndarray,
    b2_out: np.ndarray,
    sv: np.ndarray,
    modes: list[tuple[int, int]],
    *,
    q_mode: int,
    scale: float,
) -> None:
    """In-place variant of unpack_truncated_statevector."""
    N = int(b1_out.shape[0])
    dim_mode = 1 << int(q_mode)
    if sv.shape[0] != 2 * dim_mode:
        raise ValueError("Statevector length does not match q_mode")
    if b1_out.shape != (N, N) or b2_out.shape != (N, N):
        raise ValueError("b_out must be square (N,N)")

    b1_out.fill(0.0)
    b2_out.fill(0.0)

    a1 = sv[:dim_mode] * complex(scale)
    a2 = sv[dim_mode:] * complex(scale)
    for m, (iy, ix) in enumerate(modes):
        b1_out[int(iy), int(ix)] = a1[m]
        b2_out[int(iy), int(ix)] = a2[m]


def evolve_truncated_statevector_qiskit_v0(
    sv0: np.ndarray,
    energies: np.ndarray,
    *,
    t: float,
    q_mode: int,
):
    """Evolve a normalized truncated statevector under diag(E) using Qiskit.

    This function uses qiskit.quantum_info.Statevector to apply a Diagonal gate
    on the mode register.
    """
    try:
        from qiskit.quantum_info import Statevector
        from qiskit.circuit.library import Diagonal
    except Exception as e:  # pragma: no cover
        raise ImportError("qiskit is required for Qiskit shadow evolution") from e

    q_mode_i = int(q_mode)
    dim_mode = 1 << q_mode_i
    M = int(energies.shape[0])
    if M > dim_mode:
        raise ValueError("energies length exceeds mode register dimension")

    phases = np.ones((dim_mode,), dtype=np.complex128)
    if M > 0:
        phases[:M] = np.exp(-1j * energies * float(t))

    gate = Diagonal(phases.tolist())
    sv_t = Statevector(np.asarray(sv0, dtype=np.complex128)).evolve(gate, qargs=list(range(q_mode_i)))
    return np.asarray(sv_t.data, dtype=np.complex128)


def evolve_truncated_statevector_aer_v0(
    sv0: np.ndarray,
    energies: np.ndarray,
    *,
    t: float,
    q_mode: int,
    optimization_level: int = 0,
    sim=None,
) -> tuple[np.ndarray, float, float]:
    """Evolve a normalized truncated statevector using Aer statevector simulation.

    Returns:
      (sv_t, rt_transpile_s, rt_run_s)

    Notes:
    - Requires qiskit-aer.
    - This measures simulator wall-clock time, not hardware time.
    """
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit.library import Diagonal
        from qiskit_aer import AerSimulator
    except Exception as e:  # pragma: no cover
        raise ImportError("qiskit-aer is required for Aer shadow evolution") from e

    q_mode_i = int(q_mode)
    dim_mode = 1 << q_mode_i
    M = int(energies.shape[0])
    if M > dim_mode:
        raise ValueError("energies length exceeds mode register dimension")

    # phases for |m> on the mode register (m < M); remaining padded basis states get phase 1
    phases = np.ones((dim_mode,), dtype=np.complex128)
    if M > 0:
        phases[:M] = np.exp(-1j * energies * float(t))

    q_total = q_mode_i + 1
    circ = QuantumCircuit(q_total)
    circ.initialize(np.asarray(sv0, dtype=np.complex128), list(range(q_total)))
    if q_mode_i > 0:
        circ.append(Diagonal(phases.tolist()), qargs=list(range(q_mode_i)))
    circ.save_statevector()

    if sim is None:
        sim = AerSimulator(method="statevector")

    t0 = time.perf_counter()
    circ_t = transpile(circ, sim, optimization_level=int(optimization_level))
    rt_transpile = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = sim.run(circ_t).result()
    rt_run = time.perf_counter() - t0

    sv_t = np.asarray(result.data(0)["statevector"], dtype=np.complex128)
    return sv_t, float(rt_transpile), float(rt_run)


def shadow_evolve_components_qiskit_v0(
    psi1_0: np.ndarray,
    psi2_0: np.ndarray,
    *,
    mask: np.ndarray,
    t: float,
    E: np.ndarray | None = None,
    order: str = "energy",
):
    """Qiskit shadow evolution for V=0 returning low-pass fields and coeffs.

    Returns:
      (psi1_shadow, psi2_shadow, b1_shadow, b2_shadow, enc)
    where b*_shadow are zero outside mask.
    """
    N = psi1_0.shape[0]
    if E is None:
        E = core_v0.energy_grid_free(N)

    modes, energies = modes_from_mask(mask, E, order=order)
    M = int(len(modes))
    if M <= 0:
        raise ValueError("mask keeps no modes (M=0)")

    b1_0 = core_v0.unitary_fft2(psi1_0)
    b2_0 = core_v0.unitary_fft2(psi2_0)

    sv0, scale, q_mode = pack_truncated_statevector(b1_0, b2_0, modes, normalize=True)
    sv_t = evolve_truncated_statevector_qiskit_v0(sv0, energies, t=float(t), q_mode=q_mode)
    b1_t, b2_t = unpack_truncated_statevector(sv_t, modes, N=N, q_mode=q_mode, scale=scale)

    psi1_t = core_v0.unitary_ifft2(b1_t)
    psi2_t = core_v0.unitary_ifft2(b2_t)

    enc = ShadowEncoding(
        N=int(N),
        K0=float("nan"),
        M=int(M),
        q_mode=int(q_mode),
        q_total=int(q_mode) + 1,
        order=str(order),
    )
    return psi1_t, psi2_t, b1_t, b2_t, enc
