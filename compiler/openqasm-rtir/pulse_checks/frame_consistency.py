"""Frame Consistency checker — correspondence property.

Property (formal_definitions_v0.md §2.3, lines 209-218):
    FrameConsist_compiled(P, P') ≡ ∀f ∈ frames:
        phase_ref(f, P) = phase_compiled(f, P')

This is a CORRESPONDENCE check: the checker independently computes the
expected time and phase from the SOURCE PROGRAM, then compares against
the provided state (which may come from a compiler/lowering).

It does NOT trust the state's own elapsed time — it computes elapsed time
independently from the program AST. This ensures that a compiled output
which inserts extra delays (changing time) but keeps phase self-consistent
will still be caught.
"""

from __future__ import annotations

import math

from pulse_ir.ir import (
    Config,
    FrameState,
    Play,
    Acquire,
    ShiftPhase,
    Delay,
    IfBit,
    PulseStmt,
    Waveform,
)

TWO_PI = 2.0 * math.pi
PHASE_TOL = 1e-9  # floating-point tolerance


def _compute_expected(
    program: list[PulseStmt],
    config: Config,
) -> dict[str, tuple[int, float]]:
    """Independently compute expected (time, phase) per frame from program AST.

    Walks the program and applies the same time/phase rules as the formal
    definitions, but without sharing code with ref_semantics.
    Returns {frame: (expected_time, expected_phase)}.
    """
    time: dict[str, int] = {f: 0 for f in config.frames}
    phase: dict[str, float] = {f: config.init_phase[f] for f in config.frames}

    def walk(stmt: PulseStmt) -> None:
        match stmt:
            case Play(frame, waveform):
                d = waveform.duration
                time[frame] += d
                phase[frame] += TWO_PI * config.init_freq[frame] * d

            case Acquire(frame, duration, _):
                time[frame] += duration
                phase[frame] += TWO_PI * config.init_freq[frame] * duration

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
    phase from the program AST, then compares against the provided state.
    Catches both phase drift AND time drift (e.g., extra delays inserted by
    a compiler that keep phase self-consistent but diverge from source).

    Does NOT call ref_semantics — independently computes from the AST.
    Returns (ok, errors).
    """
    errors: list[str] = []
    expected = _compute_expected(program, config)

    for f in config.frames:
        exp_time, exp_phase = expected[f]
        act_time = state.time[f]
        act_phase = state.phase[f]

        if act_time != exp_time:
            errors.append(
                f"frame {f}: expected time {exp_time} "
                f"but got {act_time} (diff={act_time - exp_time})"
            )

        if abs(act_phase - exp_phase) > PHASE_TOL:
            errors.append(
                f"frame {f}: expected phase {exp_phase:.6f} "
                f"but got {act_phase:.6f} (diff={act_phase - exp_phase:.2e})"
            )

    return (len(errors) == 0, errors)
