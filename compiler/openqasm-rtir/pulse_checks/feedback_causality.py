"""Feedback Causality checker.

Property (formal_definitions_v0.md §2.2):
    FeedbackCausal(σ, P) ≡ ∀ IfBit(c, body) in P:
                              t_use ≥ σ.cbit_ready(c)

where t_use is the frame time when body would execute.

This checker walks the program AST and checks timing against the state.
It does NOT call ref_semantics — it independently tracks frame times.
"""

from __future__ import annotations

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


def _body_frame(stmt: PulseStmt) -> str | None:
    """Extract the frame that a statement operates on."""
    match stmt:
        case Play(frame, _):
            return frame
        case Acquire(frame, _, _):
            return frame
        case ShiftPhase(frame, _):
            return frame
        case Delay(_, frame):
            return frame
        case IfBit(_, body):
            return _body_frame(body)
    return None


def check_feedback_causality(
    program: list[PulseStmt],
    config: Config,
) -> tuple[bool, list[str]]:
    """Check that every IfBit uses a cbit whose acquire has already completed.

    Independently tracks frame times (does NOT reuse ref_semantics.run).
    Returns (ok, errors).
    """
    errors: list[str] = []

    # Independent time tracking — mirrors the time component of step rules
    frame_time: dict[str, int] = {f: 0 for f in config.frames}
    cbit_ready: dict[str, int] = {}

    def walk(stmt: PulseStmt) -> None:
        match stmt:
            case Play(frame, waveform):
                frame_time[frame] += waveform.duration

            case Acquire(frame, duration, cbit):
                t = frame_time[frame]
                frame_time[frame] = t + duration
                cbit_ready[cbit] = t + duration

            case ShiftPhase():
                pass  # zero duration

            case Delay(duration, frame):
                frame_time[frame] += duration

            case IfBit(cbit, body):
                f_body = _body_frame(body)
                if f_body is not None:
                    t_use = frame_time[f_body]
                    t_ready = cbit_ready.get(cbit)
                    if t_ready is None:
                        errors.append(
                            f"IfBit({cbit}): cbit was never acquired"
                        )
                    elif t_use < t_ready:
                        errors.append(
                            f"IfBit({cbit}): body starts at t={t_use} "
                            f"but cbit not ready until t={t_ready}"
                        )
                # Advance time as if body executes (conservative)
                walk(body)

    for stmt in program:
        walk(stmt)

    return (len(errors) == 0, errors)
