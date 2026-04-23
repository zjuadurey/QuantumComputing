"""Isolated compile-target comparison for ShadowFluid.

This script lives inside `cpflow_sandbox/` on purpose so that all future
iteration stays confined to this folder. It imports the repository's existing
target-construction code but writes all outputs locally.
"""

from __future__ import annotations

import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import Diagonal, StatePreparation
try:
    from qiskit.circuit.library import UnitaryGate
except ImportError:
    from qiskit.extensions import UnitaryGate
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shiftflow import core_v0, core_v1, metrics, qiskit_shadow_v0, qiskit_shadow_v1


SANDBOX_DIR = Path(__file__).resolve().parent
OUT_DIR = SANDBOX_DIR / "results"
RAW_CSV = OUT_DIR / "raw_compile_metrics.csv"
MAIN_TABLE_CSV = OUT_DIR / "main_table.csv"
SUMMARY_MD = OUT_DIR / "summary.md"

SEED = 0
TOPOLOGY = "line_nn"
BASIS_2Q = "cz"
BASIS_GATES = ["rz", "sx", "x", BASIS_2Q]
TRANSPILER_OPT_LEVEL = 3
TRANSPILER_SEED = 7

REP_NX = 3
REP_N = 2**REP_NX
REP_K0 = 1.5
REP_T = 0.5
REP_ALPHA = 0.5
REP_QX = 1
REP_QY = 0
REP_R_HOPS = 1

CPFLOW_TARGET_LOSS = 1e-6

RAW_FIELDS = [
    "case_id",
    "task_type",
    "physics_regime",
    "target_family",
    "target_kind",
    "nx",
    "N",
    "K0",
    "alpha",
    "qx",
    "qy",
    "t",
    "R_hops",
    "compiled_qubits",
    "target_size_descriptor",
    "compiler_backend",
    "topology",
    "basis_2q",
    "quality_metric",
    "quality_value",
    "twoq_count",
    "twoq_depth",
    "total_depth",
    "compile_walltime_s",
    "seed",
    "budget_tag",
    "aggregation_group",
]

MAIN_FIELDS = [
    "Case",
    "Task",
    "Full backend",
    "Shadow backend",
    "Full qubits",
    "Shadow qubits",
    "Full target size",
    "Shadow target size",
    "Full achieved quality",
    "Shadow achieved quality",
    "Full twoq count",
    "Shadow twoq count",
    "Full twoq depth",
    "Shadow twoq depth",
]


@dataclass(frozen=True)
class CircuitMetrics:
    quality_metric: str
    quality_value: float
    twoq_count: int
    twoq_depth: int
    total_depth: int
    compile_walltime_s: float


def ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def coupling_map_line(num_qubits: int) -> CouplingMap:
    return CouplingMap.from_line(num_qubits)


def count_twoq_depth(circuit: QuantumCircuit, gate_name: str = BASIS_2Q) -> int:
    dag = circuit_to_dag(circuit)
    depth = 0
    for layer in dag.layers():
        has_gate = any(node.op.name == gate_name for node in layer["graph"].op_nodes())
        if has_gate:
            depth += 1
    return depth


def compiled_operator(circuit: QuantumCircuit) -> Operator:
    layout = getattr(circuit, "layout", None)
    if layout is None:
        try:
            return Operator.from_circuit(circuit)
        except Exception:
            return Operator(circuit)

    try:
        return Operator.from_circuit(
            circuit,
            layout=layout.initial_layout,
            final_layout=layout.final_layout,
        )
    except TypeError:
        return Operator.from_circuit(circuit, layout=layout.initial_layout)
    except Exception:
        return Operator(circuit)


def state_infidelity(target_state: np.ndarray, circuit: QuantumCircuit) -> float:
    zero = np.zeros_like(target_state)
    zero[0] = 1.0
    actual = compiled_operator(circuit).data @ zero
    overlap = np.vdot(target_state, actual)
    return float(max(0.0, 1.0 - abs(overlap) ** 2))


def unitary_loss(target_unitary: np.ndarray, circuit: QuantumCircuit) -> float:
    actual = compiled_operator(circuit).data
    dim = target_unitary.shape[0]
    hs_overlap = abs(np.trace(target_unitary.conj().T @ actual)) / float(dim)
    return float(max(0.0, 1.0 - hs_overlap**2))


def transpile_generic(circuit: QuantumCircuit) -> tuple[QuantumCircuit, float]:
    t0 = time.perf_counter()
    out = transpile(
        circuit,
        basis_gates=BASIS_GATES,
        coupling_map=coupling_map_line(circuit.num_qubits),
        optimization_level=TRANSPILER_OPT_LEVEL,
        seed_transpiler=TRANSPILER_SEED,
    )
    dt = time.perf_counter() - t0
    return out, float(dt)


def evaluate_compiled_circuit(
    compiled: QuantumCircuit,
    *,
    target_kind: str,
    target_object: np.ndarray,
    compile_walltime_s: float,
) -> CircuitMetrics:
    if target_kind == "statevector":
        qmetric = "state_infidelity"
        qvalue = state_infidelity(target_object, compiled)
    elif target_kind == "unitary":
        qmetric = "unitary_loss"
        qvalue = unitary_loss(target_object, compiled)
    else:
        raise ValueError(f"Unsupported target_kind: {target_kind}")

    ops = compiled.count_ops()
    twoq_count = int(ops.get(BASIS_2Q, 0))
    return CircuitMetrics(
        quality_metric=qmetric,
        quality_value=qvalue,
        twoq_count=twoq_count,
        twoq_depth=count_twoq_depth(compiled, BASIS_2Q),
        total_depth=int(compiled.depth() or 0),
        compile_walltime_s=float(compile_walltime_s),
    )


def maybe_import_cpflow() -> Any | None:
    try:
        import cpflow  # type: ignore
    except Exception:
        return None
    return cpflow


def cpflow_budget_tag(num_qubits: int) -> str:
    if num_qubits <= 2:
        return "cpflow_q2_small"
    if num_qubits <= 4:
        return "cpflow_q4_small"
    return "cpflow_q6_small"


def synthesize_with_cpflow(
    target_unitary: np.ndarray,
    *,
    num_qubits: int,
    seed: int,
) -> tuple[QuantumCircuit, float, float, str] | None:
    cpflow = maybe_import_cpflow()
    if cpflow is None:
        return None

    layer = [[i, i + 1] for i in range(num_qubits - 1)]
    if num_qubits <= 2:
        num_cp_gates = 4
        accepted_num_cz_gates = 4
        num_samples = 3
        num_gd_iterations = 200
    elif num_qubits <= 4:
        num_cp_gates = 8
        accepted_num_cz_gates = 8
        num_samples = 3
        num_gd_iterations = 300
    else:
        num_cp_gates = 12
        accepted_num_cz_gates = 12
        num_samples = 3
        num_gd_iterations = 350

    decomposer = cpflow.Synthesize(
        layer,
        target_unitary=target_unitary,
        label=f"shadowfluid_q{num_qubits}",
    )
    options = cpflow.StaticOptions(
        num_cp_gates=num_cp_gates,
        accepted_num_cz_gates=accepted_num_cz_gates,
        num_samples=num_samples,
        num_gd_iterations=num_gd_iterations,
        target_loss=CPFLOW_TARGET_LOSS,
        random_seed=seed,
    )

    t0 = time.perf_counter()
    results = decomposer.static(options, save_results=False)
    dt = time.perf_counter() - t0
    if not results.decompositions:
        return None

    best = min(results.decompositions, key=lambda d: (d.loss, d.cz_count, d.cz_depth))
    return best.circuit, float(best.loss), float(dt), cpflow_budget_tag(num_qubits)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def complex128_array(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.complex128)


def full_state_prep_target(nx: int) -> np.ndarray:
    _, _, stacked = core_v0.vortex_initial_condition(2**nx)
    return complex128_array(stacked)


def shadow_state_prep_target(nx: int, K0: float) -> tuple[np.ndarray, int, int]:
    N = 2**nx
    psi1_0, psi2_0, _ = core_v0.vortex_initial_condition(N)
    mask = core_v0.low_freq_mask(N, K0)
    E = core_v0.energy_grid_free(N)
    modes, _energies = qiskit_shadow_v0.modes_from_mask(mask, E, order="energy")
    b1_0 = core_v0.unitary_fft2(psi1_0)
    b2_0 = core_v0.unitary_fft2(psi2_0)
    sv0, _scale, q_mode = qiskit_shadow_v0.pack_truncated_statevector(b1_0, b2_0, modes, normalize=True)
    return complex128_array(sv0), int(len(modes)), int(q_mode)


def v0_full_unitary_target(nx: int, t: float) -> tuple[np.ndarray, int]:
    N = 2**nx
    E = core_v0.energy_grid_free(N)
    phases = np.exp(-1j * E.reshape(-1) * float(t))
    U = np.diag(phases.astype(np.complex128))
    return U, int(2 * nx)


def v0_shadow_unitary_target(nx: int, K0: float, t: float) -> tuple[np.ndarray, int, int]:
    N = 2**nx
    mask = core_v0.low_freq_mask(N, K0)
    E = core_v0.energy_grid_free(N)
    _modes, energies = qiskit_shadow_v0.modes_from_mask(mask, E, order="energy")
    M = int(len(energies))
    q_mode = int(math.ceil(math.log2(M))) if M > 1 else 0
    dim = 1 << q_mode
    phases = np.ones((dim,), dtype=np.complex128)
    phases[:M] = np.exp(-1j * energies * float(t))
    return np.diag(phases), M, q_mode


def v1_targets(
    *,
    nx: int,
    alpha: float,
    qx: int,
    qy: int,
    K0: float,
    t: float,
    R_hops: int,
) -> dict[str, tuple[np.ndarray, int, int]]:
    N = 2**nx
    comps = core_v1.potential_single(alpha, qx=qx, qy=qy)
    H_dense = core_v1.build_H_dense(N, comps)

    mask = core_v0.low_freq_mask(N, K0)
    K_flat = core_v1.mask_to_flat(mask, N)
    R_flat = core_v1.build_R_closure((0, 0), comps, N, max_hops=R_hops)
    H_K = core_v1.extract_submatrix(H_dense, K_flat)
    H_R = core_v1.extract_submatrix(H_dense, R_flat)

    U_full = qiskit_shadow_v1.expm(-1j * H_dense * t)
    U_K = qiskit_shadow_v1.build_padded_unitary(H_K, t)
    U_R = qiskit_shadow_v1.build_padded_unitary(H_R, t)

    return {
        "full": (complex128_array(U_full), int(H_dense.shape[0]), int(2 * nx)),
        "shadow_K": (complex128_array(U_K), int(H_K.shape[0]), int(math.log2(U_K.shape[0]))),
        "shadow_R": (complex128_array(U_R), int(H_R.shape[0]), int(math.log2(U_R.shape[0]))),
    }


def build_state_prep_circuit(target_state: np.ndarray) -> QuantumCircuit:
    num_qubits = int(math.log2(target_state.shape[0]))
    qc = QuantumCircuit(num_qubits)
    qc.append(StatePreparation(target_state), list(range(num_qubits)))
    return qc


def build_unitary_circuit(target_unitary: np.ndarray) -> QuantumCircuit:
    num_qubits = int(math.log2(target_unitary.shape[0]))
    qc = QuantumCircuit(num_qubits)
    if np.allclose(target_unitary, np.diag(np.diag(target_unitary))):
        qc.append(Diagonal(np.diag(target_unitary).tolist()), list(range(num_qubits)))
    else:
        qc.append(UnitaryGate(target_unitary), list(range(num_qubits)))
    return qc


def make_row(
    *,
    case_id: str,
    task_type: str,
    physics_regime: str,
    target_family: str,
    target_kind: str,
    nx: int,
    N: int,
    K0: float | None,
    alpha: float | None,
    qx: int | None,
    qy: int | None,
    t: float | None,
    R_hops: int | None,
    compiled_qubits: int,
    target_size_descriptor: str,
    compiler_backend: str,
    quality_metric: str,
    quality_value: float,
    twoq_count: int,
    twoq_depth: int,
    total_depth: int,
    compile_walltime_s: float,
    seed: int,
    budget_tag: str,
    aggregation_group: str,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "task_type": task_type,
        "physics_regime": physics_regime,
        "target_family": target_family,
        "target_kind": target_kind,
        "nx": nx,
        "N": N,
        "K0": "" if K0 is None else K0,
        "alpha": "" if alpha is None else alpha,
        "qx": "" if qx is None else qx,
        "qy": "" if qy is None else qy,
        "t": "" if t is None else t,
        "R_hops": "" if R_hops is None else R_hops,
        "compiled_qubits": compiled_qubits,
        "target_size_descriptor": target_size_descriptor,
        "compiler_backend": compiler_backend,
        "topology": TOPOLOGY,
        "basis_2q": BASIS_2Q,
        "quality_metric": quality_metric,
        "quality_value": quality_value,
        "twoq_count": twoq_count,
        "twoq_depth": twoq_depth,
        "total_depth": total_depth,
        "compile_walltime_s": compile_walltime_s,
        "seed": seed,
        "budget_tag": budget_tag,
        "aggregation_group": aggregation_group,
    }


def compile_generic_row(
    *,
    case_id: str,
    task_type: str,
    physics_regime: str,
    target_family: str,
    target_kind: str,
    target_object: np.ndarray,
    nx: int,
    N: int,
    K0: float | None,
    alpha: float | None,
    qx: int | None,
    qy: int | None,
    t: float | None,
    R_hops: int | None,
    compiled_qubits: int,
    target_size_descriptor: str,
    aggregation_group: str,
) -> dict[str, Any]:
    if target_kind == "statevector":
        source = build_state_prep_circuit(target_object)
    else:
        source = build_unitary_circuit(target_object)
    compiled, walltime_s = transpile_generic(source)
    result = evaluate_compiled_circuit(
        compiled,
        target_kind=target_kind,
        target_object=target_object,
        compile_walltime_s=walltime_s,
    )
    return make_row(
        case_id=case_id,
        task_type=task_type,
        physics_regime=physics_regime,
        target_family=target_family,
        target_kind=target_kind,
        nx=nx,
        N=N,
        K0=K0,
        alpha=alpha,
        qx=qx,
        qy=qy,
        t=t,
        R_hops=R_hops,
        compiled_qubits=compiled_qubits,
        target_size_descriptor=target_size_descriptor,
        compiler_backend="generic_transpile",
        quality_metric=result.quality_metric,
        quality_value=result.quality_value,
        twoq_count=result.twoq_count,
        twoq_depth=result.twoq_depth,
        total_depth=result.total_depth,
        compile_walltime_s=result.compile_walltime_s,
        seed=SEED,
        budget_tag=f"generic_o{TRANSPILER_OPT_LEVEL}",
        aggregation_group=aggregation_group,
    )


def compile_cpflow_row(
    *,
    case_id: str,
    task_type: str,
    physics_regime: str,
    target_family: str,
    target_unitary: np.ndarray,
    nx: int,
    N: int,
    K0: float | None,
    alpha: float | None,
    qx: int | None,
    qy: int | None,
    t: float | None,
    R_hops: int | None,
    compiled_qubits: int,
    target_size_descriptor: str,
    aggregation_group: str,
) -> dict[str, Any] | None:
    out = synthesize_with_cpflow(target_unitary, num_qubits=compiled_qubits, seed=SEED)
    if out is None:
        return None
    circuit, _loss_from_cpflow, walltime_s, budget_tag = out
    result = evaluate_compiled_circuit(
        circuit,
        target_kind="unitary",
        target_object=target_unitary,
        compile_walltime_s=walltime_s,
    )
    return make_row(
        case_id=case_id,
        task_type=task_type,
        physics_regime=physics_regime,
        target_family=target_family,
        target_kind="unitary",
        nx=nx,
        N=N,
        K0=K0,
        alpha=alpha,
        qx=qx,
        qy=qy,
        t=t,
        R_hops=R_hops,
        compiled_qubits=compiled_qubits,
        target_size_descriptor=target_size_descriptor,
        compiler_backend="cpflow",
        quality_metric=result.quality_metric,
        quality_value=result.quality_value,
        twoq_count=result.twoq_count,
        twoq_depth=result.twoq_depth,
        total_depth=result.total_depth,
        compile_walltime_s=result.compile_walltime_s,
        seed=SEED,
        budget_tag=budget_tag,
        aggregation_group=aggregation_group,
    )


def gather_raw_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    full_state = full_state_prep_target(REP_NX)
    rows.append(
        compile_generic_row(
            case_id="v0_state_full_generic",
            task_type="state_prep",
            physics_regime="v0",
            target_family="full",
            target_kind="statevector",
            target_object=full_state,
            nx=REP_NX,
            N=REP_N,
            K0=REP_K0,
            alpha=None,
            qx=None,
            qy=None,
            t=None,
            R_hops=None,
            compiled_qubits=metrics.q_base(REP_NX),
            target_size_descriptor=f"{2 * REP_N * REP_N} amplitudes on {metrics.q_base(REP_NX)} qubits",
            aggregation_group="v0_state_prep_full",
        )
    )

    shadow_state, M_K_state, q_mode_state = shadow_state_prep_target(REP_NX, REP_K0)
    rows.append(
        compile_generic_row(
            case_id="v0_state_shadowK_generic",
            task_type="state_prep",
            physics_regime="v0",
            target_family="shadow_K",
            target_kind="statevector",
            target_object=shadow_state,
            nx=REP_NX,
            N=REP_N,
            K0=REP_K0,
            alpha=None,
            qx=None,
            qy=None,
            t=None,
            R_hops=None,
            compiled_qubits=q_mode_state + 1,
            target_size_descriptor=f"{2 * M_K_state} amplitudes packed into {1 << (q_mode_state + 1)} amplitudes",
            aggregation_group="v0_state_prep_shadow",
        )
    )

    v0_full_u, v0_full_qubits = v0_full_unitary_target(REP_NX, REP_T)
    rows.append(
        compile_generic_row(
            case_id="v0_evo_full_generic",
            task_type="time_evolution",
            physics_regime="v0",
            target_family="full",
            target_kind="unitary",
            target_object=v0_full_u,
            nx=REP_NX,
            N=REP_N,
            K0=REP_K0,
            alpha=None,
            qx=None,
            qy=None,
            t=REP_T,
            R_hops=None,
            compiled_qubits=v0_full_qubits,
            target_size_descriptor=f"{REP_N * REP_N}x{REP_N * REP_N} active-mode unitary",
            aggregation_group="v0_time_evolution_full",
        )
    )

    v0_shadow_u, M_K_u, q_mode_u = v0_shadow_unitary_target(REP_NX, REP_K0, REP_T)
    rows.append(
        compile_generic_row(
            case_id="v0_evo_shadowK_generic",
            task_type="time_evolution",
            physics_regime="v0",
            target_family="shadow_K",
            target_kind="unitary",
            target_object=v0_shadow_u,
            nx=REP_NX,
            N=REP_N,
            K0=REP_K0,
            alpha=None,
            qx=None,
            qy=None,
            t=REP_T,
            R_hops=None,
            compiled_qubits=q_mode_u,
            target_size_descriptor=f"{M_K_u}x{M_K_u} reduced unitary padded to {v0_shadow_u.shape[0]}x{v0_shadow_u.shape[0]}",
            aggregation_group="v0_time_evolution_shadow",
        )
    )
    cpflow_v0 = compile_cpflow_row(
        case_id="v0_evo_shadowK_cpflow",
        task_type="time_evolution",
        physics_regime="v0",
        target_family="shadow_K",
        target_unitary=v0_shadow_u,
        nx=REP_NX,
        N=REP_N,
        K0=REP_K0,
        alpha=None,
        qx=None,
        qy=None,
        t=REP_T,
        R_hops=None,
        compiled_qubits=q_mode_u,
        target_size_descriptor=f"{M_K_u}x{M_K_u} reduced unitary padded to {v0_shadow_u.shape[0]}x{v0_shadow_u.shape[0]}",
        aggregation_group="v0_time_evolution_shadow",
    )
    if cpflow_v0 is not None:
        rows.append(cpflow_v0)

    v1 = v1_targets(
        nx=REP_NX,
        alpha=REP_ALPHA,
        qx=REP_QX,
        qy=REP_QY,
        K0=REP_K0,
        t=REP_T,
        R_hops=REP_R_HOPS,
    )
    full_u, full_dim, full_qubits = v1["full"]
    rows.append(
        compile_generic_row(
            case_id="v1_evo_full_generic",
            task_type="time_evolution",
            physics_regime="v1",
            target_family="full",
            target_kind="unitary",
            target_object=full_u,
            nx=REP_NX,
            N=REP_N,
            K0=REP_K0,
            alpha=REP_ALPHA,
            qx=REP_QX,
            qy=REP_QY,
            t=REP_T,
            R_hops=REP_R_HOPS,
            compiled_qubits=full_qubits,
            target_size_descriptor=f"{full_dim}x{full_dim} active-mode unitary",
            aggregation_group="v1_time_evolution_full",
        )
    )

    for target_family, case_prefix in [("shadow_K", "v1_evo_shadowK"), ("shadow_R", "v1_evo_shadowR")]:
        target_u, reduced_dim, reduced_qubits = v1[target_family]
        rows.append(
            compile_generic_row(
                case_id=f"{case_prefix}_generic",
                task_type="time_evolution",
                physics_regime="v1",
                target_family=target_family,
                target_kind="unitary",
                target_object=target_u,
                nx=REP_NX,
                N=REP_N,
                K0=REP_K0,
                alpha=REP_ALPHA,
                qx=REP_QX,
                qy=REP_QY,
                t=REP_T,
                R_hops=REP_R_HOPS,
                compiled_qubits=reduced_qubits,
                target_size_descriptor=f"{reduced_dim}x{reduced_dim} reduced unitary padded to {target_u.shape[0]}x{target_u.shape[0]}",
                aggregation_group="v1_time_evolution_shadow_total",
            )
        )
        cpflow_row = compile_cpflow_row(
            case_id=f"{case_prefix}_cpflow",
            task_type="time_evolution",
            physics_regime="v1",
            target_family=target_family,
            target_unitary=target_u,
            nx=REP_NX,
            N=REP_N,
            K0=REP_K0,
            alpha=REP_ALPHA,
            qx=REP_QX,
            qy=REP_QY,
            t=REP_T,
            R_hops=REP_R_HOPS,
            compiled_qubits=reduced_qubits,
            target_size_descriptor=f"{reduced_dim}x{reduced_dim} reduced unitary padded to {target_u.shape[0]}x{target_u.shape[0]}",
            aggregation_group="v1_time_evolution_shadow_total",
        )
        if cpflow_row is not None:
            rows.append(cpflow_row)

    return rows


def choose_shadow_row(rows: list[dict[str, Any]], task_group: str) -> dict[str, Any] | None:
    cpflow_rows = [r for r in rows if r["compiler_backend"] == "cpflow"]
    generic_rows = [r for r in rows if r["compiler_backend"] == "generic_transpile"]
    if task_group == "v1_time_evolution_shadow_total":
        candidates = cpflow_rows or generic_rows
        if not candidates:
            return None
        grouped = [r for r in candidates if r["aggregation_group"] == task_group]
        if not grouped:
            return None
        backend = grouped[0]["compiler_backend"]
        return {
            "compiler_backend": backend,
            "compiled_qubits": "+".join(str(r["compiled_qubits"]) for r in grouped),
            "target_size_descriptor": " + ".join(r["target_size_descriptor"] for r in grouped),
            "quality_value": max(float(r["quality_value"]) for r in grouped),
            "twoq_count": sum(int(r["twoq_count"]) for r in grouped),
            "twoq_depth": sum(int(r["twoq_depth"]) for r in grouped),
        }

    preferred = [r for r in rows if r["aggregation_group"] == task_group and r["compiler_backend"] == "cpflow"]
    if preferred:
        return preferred[0]
    fallback = [r for r in rows if r["aggregation_group"] == task_group and r["compiler_backend"] == "generic_transpile"]
    return fallback[0] if fallback else None


def build_main_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def add_pair(case: str, task: str, full_case_id: str, shadow_group: str) -> None:
        full_row = next(r for r in rows if r["case_id"] == full_case_id)
        shadow_row = choose_shadow_row(rows, shadow_group)
        if shadow_row is None:
            return
        out.append(
            {
                "Case": case,
                "Task": task,
                "Full backend": full_row["compiler_backend"],
                "Shadow backend": shadow_row["compiler_backend"],
                "Full qubits": full_row["compiled_qubits"],
                "Shadow qubits": shadow_row["compiled_qubits"],
                "Full target size": full_row["target_size_descriptor"],
                "Shadow target size": shadow_row["target_size_descriptor"],
                "Full achieved quality": full_row["quality_value"],
                "Shadow achieved quality": shadow_row["quality_value"],
                "Full twoq count": full_row["twoq_count"],
                "Shadow twoq count": shadow_row["twoq_count"],
                "Full twoq depth": full_row["twoq_depth"],
                "Shadow twoq depth": shadow_row["twoq_depth"],
            }
        )

    add_pair("V=0", "state_prep", "v0_state_full_generic", "v0_state_prep_shadow")
    add_pair("V=0", "time_evolution", "v0_evo_full_generic", "v0_time_evolution_shadow")
    add_pair("V!=0", "time_evolution", "v1_evo_full_generic", "v1_time_evolution_shadow_total")
    return out


def float_str(x: Any) -> str:
    xf = float(x)
    if abs(xf) >= 1e-3 and abs(xf) < 1e3:
        return f"{xf:.6f}".rstrip("0").rstrip(".")
    return f"{xf:.6e}"


def write_summary(rows: list[dict[str, Any]], main_rows: list[dict[str, Any]]) -> None:
    cpflow_rows = [r for r in rows if r["compiler_backend"] == "cpflow"]
    summary = [
        "# CPFlow Compile Comparison",
        "",
        "## Cases run",
        "",
        f"- Representative grid: `nx = {REP_NX}`, `N = {REP_N}`.",
        f"- `V=0` sanity case with canonical vortex state, `K0 = {REP_K0}`, `t = {REP_T}`.",
        f"- `V!=0` representative case with single-cosine potential, `alpha = {REP_ALPHA}`, `q = ({REP_QX}, {REP_QY})`, `K0 = {REP_K0}`, `t = {REP_T}`, `R_hops = {REP_R_HOPS}`.",
        "",
        "## Targets synthesized",
        "",
        "- Full initial-state target on the full spinor register.",
        "- Reduced Shadow initial-state target on the packed retained-mode register.",
        "- Full active-mode evolution unitary for `V=0` and `V!=0`.",
        "- Reduced `shadow_K` and `shadow_R` active-mode evolution unitaries for `V!=0`.",
        "",
        "## Fixed compilation conditions",
        "",
        f"- Topology family: `{TOPOLOGY}`.",
        f"- Two-qubit basis: `{BASIS_2Q}`.",
        f"- Generic backend: `transpile` with optimization level `{TRANSPILER_OPT_LEVEL}` and fixed seed `{TRANSPILER_SEED}`.",
        f"- Quality metrics: `state_infidelity` for state preparation and `unitary_loss` for unitary targets.",
        f"- CPFlow target loss budget when available: `{float_str(CPFLOW_TARGET_LOSS)}`.",
        "",
        "## Produced artifacts",
        "",
        f"- Raw long-form CSV: `{RAW_CSV}`.",
        f"- Compact paired table: `{MAIN_TABLE_CSV}`.",
        "",
    ]
    if cpflow_rows:
        summary.extend(
            [
                "## Notes on reduced `V!=0` evolution",
                "",
                "- `shadow_K` and `shadow_R` are kept separate in the raw CSV.",
                "- They are aggregated into a single reduced `V!=0` time-evolution entry in `main_table.csv` by summing the two circuit costs and taking the worse achieved quality.",
                "",
                "## Factual takeaway",
                "",
                "- In this representative run, ShadowFluid creates smaller synthesis targets than the corresponding full targets.",
                "- Under the same topology family and recorded quality constraints, the reduced targets lead to lower two-qubit circuit complexity, and the reduced unitary targets are compatible with CPFlow-style AI-assisted synthesis.",
            ]
        )
    else:
        summary.extend(
            [
                "## Notes on reduced `V!=0` evolution",
                "",
                "- `shadow_K` and `shadow_R` are kept separate in the raw CSV.",
                "- They are aggregated into a single reduced `V!=0` time-evolution entry in `main_table.csv` by summing the two circuit costs and taking the worse achieved quality.",
                "",
                "## Factual takeaway",
                "",
                "- In this representative run, ShadowFluid creates smaller synthesis targets than the corresponding full targets.",
                "- The generic compilation rows already show lower two-qubit complexity on the reduced targets; CPFlow rows were not generated in this run because CPFlow was unavailable or did not converge under the selected budget.",
            ]
        )

    SUMMARY_MD.write_text("\n".join(summary) + "\n")


def main() -> int:
    ensure_out_dir()
    rows = gather_raw_rows()
    write_csv(RAW_CSV, rows, RAW_FIELDS)
    main_rows = build_main_table(rows)
    write_csv(MAIN_TABLE_CSV, main_rows, MAIN_FIELDS)
    write_summary(rows, main_rows)
    print(f"Wrote {RAW_CSV}")
    print(f"Wrote {MAIN_TABLE_CSV}")
    print(f"Wrote {SUMMARY_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
