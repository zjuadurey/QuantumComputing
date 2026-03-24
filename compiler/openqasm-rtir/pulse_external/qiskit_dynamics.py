"""Qiskit Dynamics as supporting evidence for v0.5.

This module intentionally plays a secondary role:
- FullContract remains the primary verifier of lowering correctness.
- Qiskit Dynamics is used as an independent, community-recognized witness
  that certain lowering faults produce externally observable deviations.

The current implementation focuses on single-frame phase-sensitive cases,
which are the cleanest bridge from our PulseEvent schedules to a trusted
external component.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from pulse_ir.ir import Config, PulseStmt
from pulse_lowering.lower_to_schedule import lower_to_schedule
from pulse_lowering.schedule import PulseEvent

try:
    from qiskit.quantum_info import Statevector
    from qiskit_dynamics import Solver
    from qiskit_dynamics.signals import DiscreteSignal
except ImportError as exc:  # pragma: no cover - exercised only when optional dep missing
    Solver = None
    Statevector = None
    DiscreteSignal = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


AmplitudeResolver = Callable[[str], float] | dict[str, float] | None


@dataclass(frozen=True)
class DynamicsCorroborationResult:
    """Outcome of comparing two schedules in Qiskit Dynamics."""

    frame: str
    fidelity: float
    correct_state: np.ndarray
    candidate_state: np.ndarray
    correct_events: list[PulseEvent]
    candidate_events: list[PulseEvent]


def _require_qiskit_dynamics() -> None:
    if _IMPORT_ERROR is not None:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "qiskit-dynamics is required for v0.5 external corroboration. "
            "Install it in the active environment before using this module."
        ) from _IMPORT_ERROR


def _resolve_amplitude(payload: str, amplitude_map: AmplitudeResolver) -> float:
    if callable(amplitude_map):
        return float(amplitude_map(payload))
    if isinstance(amplitude_map, dict):
        return float(amplitude_map.get(payload, 1.0))
    return 1.0


def _normalize(state: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(state)
    if norm == 0:
        raise ValueError("Dynamics produced the zero vector, cannot normalize.")
    return state / norm


def _coerce_frames(
    events: list[PulseEvent],
    frames: list[str] | tuple[str, ...] | set[str] | frozenset[str] | None,
) -> set[str]:
    if frames is None:
        return {ev.frame for ev in events}
    return set(frames)


def _build_xy_samples(
    events: list[PulseEvent],
    frames: set[str],
    amplitude_map: AmplitudeResolver,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    if dt <= 0:
        raise ValueError("dt must be positive.")

    relevant_events = [ev for ev in events if ev.frame in frames]
    if not relevant_events:
        return np.zeros(1, dtype=float), np.zeros(1, dtype=float), dt

    play_events = [ev for ev in relevant_events if ev.kind == "play"]
    if not play_events:
        total_time = max(ev.end for ev in relevant_events)
        samples = max(1, int(total_time / dt))
        return np.zeros(samples, dtype=float), np.zeros(samples, dtype=float), total_time

    final_end = max(ev.end for ev in relevant_events)
    samples = int(final_end / dt)
    if abs(samples * dt - final_end) > 1e-9:
        raise ValueError(
            f"Event end {final_end} is not aligned to dt={dt}; "
            "use a dt that divides the schedule grid."
        )

    x_samples = np.zeros(samples, dtype=float)
    y_samples = np.zeros(samples, dtype=float)

    for ev in play_events:
        start = int(ev.start / dt)
        end = int(ev.end / dt)
        amp = _resolve_amplitude(ev.payload, amplitude_map)
        x_samples[start:end] += amp * np.cos(ev.phase_before)
        y_samples[start:end] += amp * np.sin(ev.phase_before)

    return x_samples, y_samples, final_end


def simulate_schedule(
    events: list[PulseEvent],
    *,
    frames: list[str] | tuple[str, ...] | set[str] | frozenset[str] | None = None,
    amplitude_map: AmplitudeResolver = None,
    dt: float = 1.0,
    drive_scale: float = 0.5,
    static_drift: float = 0.0,
) -> np.ndarray:
    """Simulate the play content of a lowered schedule on a single qubit.

    All selected frames are projected onto one effective qubit using two
    orthogonal drive channels (`ux`, `uy`). Frame phase determines how each
    play contributes to those two channels. Delays affect the total evolution
    time through `static_drift`, which makes timing-sensitive bugs externally
    visible.
    """
    _require_qiskit_dynamics()

    selected_frames = _coerce_frames(events, frames)
    x_samples, y_samples, total_time = _build_xy_samples(
        events,
        selected_frames,
        amplitude_map,
        dt,
    )

    x_op = drive_scale * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    y_op = drive_scale * np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    static_h = static_drift * np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    solver = Solver(
        static_hamiltonian=static_h,
        hamiltonian_operators=[x_op, y_op],
        hamiltonian_channels=["ux", "uy"],
        channel_carrier_freqs={"ux": 0.0, "uy": 0.0},
        dt=dt,
    )

    result = solver.solve(
        t_span=[0.0, total_time],
        y0=Statevector([1.0, 0.0]),
        signals=[
            DiscreteSignal(dt, x_samples),
            DiscreteSignal(dt, y_samples),
        ],
    )
    return _normalize(np.asarray(result.y[-1], dtype=complex))


def simulate_single_frame_schedule(
    events: list[PulseEvent],
    frame: str,
    *,
    amplitude_map: AmplitudeResolver = None,
    dt: float = 1.0,
    drive_scale: float = 0.5,
) -> np.ndarray:
    """Simulate a single-frame play schedule with Qiskit Dynamics.

    The schedule is translated to two orthogonal drive channels (`ux`, `uy`)
    using the frame phase snapshot carried by each `PulseEvent`.
    """
    return simulate_schedule(
        events,
        frames=[frame],
        amplitude_map=amplitude_map,
        dt=dt,
        drive_scale=drive_scale,
    )

    

def compare_schedule_lowerings(
    program: list[PulseStmt],
    config: Config,
    *,
    scope: str = "schedule",
    frames: list[str] | tuple[str, ...] | set[str] | frozenset[str] | None = None,
    lower_correct: Callable[[list[PulseStmt], Config], list[PulseEvent]] = lower_to_schedule,
    lower_candidate: Callable[[list[PulseStmt], Config], list[PulseEvent]],
    amplitude_map: AmplitudeResolver = None,
    dt: float = 1.0,
    drive_scale: float = 0.5,
    static_drift: float = 0.0,
) -> DynamicsCorroborationResult:
    """Compare two lowerings using a schedule-level Qiskit Dynamics witness."""
    correct_events = lower_correct(program, config)
    candidate_events = lower_candidate(program, config)

    correct_state = simulate_schedule(
        correct_events,
        frames=frames,
        amplitude_map=amplitude_map,
        dt=dt,
        drive_scale=drive_scale,
        static_drift=static_drift,
    )
    candidate_state = simulate_schedule(
        candidate_events,
        frames=frames,
        amplitude_map=amplitude_map,
        dt=dt,
        drive_scale=drive_scale,
        static_drift=static_drift,
    )

    fidelity = abs(np.vdot(correct_state, candidate_state)) ** 2
    return DynamicsCorroborationResult(
        frame=scope,
        fidelity=float(fidelity),
        correct_state=correct_state,
        candidate_state=candidate_state,
        correct_events=correct_events,
        candidate_events=candidate_events,
    )


def compare_single_frame_lowerings(
    program: list[PulseStmt],
    config: Config,
    *,
    frame: str,
    lower_correct: Callable[[list[PulseStmt], Config], list[PulseEvent]] = lower_to_schedule,
    lower_candidate: Callable[[list[PulseStmt], Config], list[PulseEvent]],
    amplitude_map: AmplitudeResolver = None,
    dt: float = 1.0,
    drive_scale: float = 0.5,
) -> DynamicsCorroborationResult:
    """Compare correct vs candidate lowering through Qiskit Dynamics."""
    return compare_schedule_lowerings(
        program,
        config,
        scope=frame,
        frames=[frame],
        lower_correct=lower_correct,
        lower_candidate=lower_candidate,
        amplitude_map=amplitude_map,
        dt=dt,
        drive_scale=drive_scale,
    )
