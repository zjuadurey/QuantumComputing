"""Frame Consistency checker — correspondence property.

v0.4: Port-aware source-vs-compiled correspondence.

The checker independently computes expected (time, phase) from the SOURCE
PROGRAM using port-aware timing (max(frame_time, port_time)), then compares
against the provided compiled state.

It does NOT trust the compiled state's own time — it computes time
independently. This ensures that a compiled output with extra delays
or missing stalls will be caught.

Does NOT import ref_semantics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from pulse_ir.ir import (
    Config,
    FrameState,
    Play,
    Acquire,
    ShiftPhase,
    Delay,
    IfBit,
    PulseStmt,
)

TWO_PI = 2.0 * math.pi
PHASE_TOL = 1e-9  # floating-point tolerance


@dataclass(frozen=True)
class FrameMismatch:
    frame: str
    expected_time: int
    actual_time: int
    time_diff: int
    expected_phase: float
    actual_phase: float
    phase_diff: float


@dataclass(frozen=True)
class FrameConsistencyDiagnostics:
    mismatches: list[FrameMismatch] = field(default_factory=list)
    max_abs_time_diff: int = 0
    max_abs_phase_diff: float = 0.0
    num_time_mismatches: int = 0
    num_phase_mismatches: int = 0


def _compute_expected(
    program: list[PulseStmt],
    config: Config,
) -> dict[str, tuple[int, float]]:
    """Independently compute expected (time, phase) per frame from program AST.

    v0.4: Uses port-aware timing (max(frame_time, port_time)).
    Returns {frame: (expected_time, expected_phase)}.
    """
    time: dict[str, int] = {f: 0 for f in config.frames}
    phase: dict[str, float] = {f: config.init_phase[f] for f in config.frames}
    port_time: dict[str, int] = {p: 0 for p in config.ports}

    def walk(stmt: PulseStmt) -> None:
        match stmt:
            case Play(frame, waveform):
                d = waveform.duration
                p = config.port_of[frame]
                freq = config.init_freq[frame]
                start = max(time[frame], port_time[p])
                end = start + d
                total_advance = end - time[frame]
                phase[frame] += TWO_PI * freq * total_advance
                time[frame] = end
                port_time[p] = end

            case Acquire(frame, duration, _):
                p = config.port_of[frame]
                freq = config.init_freq[frame]
                start = max(time[frame], port_time[p])
                end = start + duration
                total_advance = end - time[frame]
                phase[frame] += TWO_PI * freq * total_advance
                time[frame] = end
                port_time[p] = end

            case ShiftPhase(frame, angle):
                phase[frame] += angle

            case Delay(duration, frame):
                time[frame] += duration
                phase[frame] += TWO_PI * config.init_freq[frame] * duration

            case IfBit(_, body):
                walk(body)

    for stmt in program:
        walk(stmt)

    return {f: (time[f], phase[f]) for f in config.frames}


def check_frame_consistency(
    state: FrameState,
    program: list[PulseStmt],
    config: Config,
) -> tuple[bool, list[str]]:
    """Check that each frame's time and phase match the source program semantics.

    This is a correspondence check: independently computes expected time and
    phase from the program AST (with port-aware timing), then compares against
    the provided state.

    Does NOT call ref_semantics — independently computes from the AST.
    Returns (ok, errors).
    """
    diagnostics = diagnose_frame_consistency(state, program, config)
    errors = format_frame_consistency_errors(diagnostics)
    return (len(errors) == 0, errors)


def diagnose_frame_consistency(
    state: FrameState,
    program: list[PulseStmt],
    config: Config,
) -> FrameConsistencyDiagnostics:
    """Return structured source-vs-compiled correspondence diagnostics."""
    expected = _compute_expected(program, config)
    mismatches: list[FrameMismatch] = []
    max_abs_time_diff = 0
    max_abs_phase_diff = 0.0
    num_time_mismatches = 0
    num_phase_mismatches = 0

    for f in config.frames:
        exp_time, exp_phase = expected[f]
        act_time = state.time[f]
        act_phase = state.phase[f]
        time_diff = act_time - exp_time
        phase_diff = act_phase - exp_phase

        if time_diff != 0 or abs(phase_diff) > PHASE_TOL:
            mismatches.append(FrameMismatch(
                frame=f,
                expected_time=exp_time,
                actual_time=act_time,
                time_diff=time_diff,
                expected_phase=exp_phase,
                actual_phase=act_phase,
                phase_diff=phase_diff,
            ))

        if time_diff != 0:
            num_time_mismatches += 1
            max_abs_time_diff = max(max_abs_time_diff, abs(time_diff))

        if abs(phase_diff) > PHASE_TOL:
            num_phase_mismatches += 1
            max_abs_phase_diff = max(max_abs_phase_diff, abs(phase_diff))

    return FrameConsistencyDiagnostics(
        mismatches=mismatches,
        max_abs_time_diff=max_abs_time_diff,
        max_abs_phase_diff=max_abs_phase_diff,
        num_time_mismatches=num_time_mismatches,
        num_phase_mismatches=num_phase_mismatches,
    )


def format_frame_consistency_errors(
    diagnostics: FrameConsistencyDiagnostics,
) -> list[str]:
    """Render human-readable messages from structured diagnostics."""
    errors: list[str] = []
    for mismatch in diagnostics.mismatches:
        if mismatch.time_diff != 0:
            errors.append(
                f"frame {mismatch.frame}: expected time {mismatch.expected_time} "
                f"but got {mismatch.actual_time} (diff={mismatch.time_diff})"
            )
        if abs(mismatch.phase_diff) > PHASE_TOL:
            errors.append(
                f"frame {mismatch.frame}: expected phase {mismatch.expected_phase:.6f} "
                f"but got {mismatch.actual_phase:.6f} "
                f"(diff={mismatch.phase_diff:.2e})"
            )
    return errors
