"""Feedback Causality checker.

Property (formal_definitions_v0.md §2.2):
    FeedbackCausal(σ, P) ≡ ∀ IfBit(c, body) in P:
                              t_use ≥ σ.cbit_ready(c)

where t_use is the frame time when body would execute.

Two modes:
  1. Source-level: independently tracks frame times from the program AST.
  2. Compiled (schedule-level): checks the lowered PulseEvent schedule directly.
     Events tagged with conditional_on are compared against cbit_ready derived
     from acquire events in the schedule. This enables end-to-end verification:
     source program → lowering → schedule → checker.

Neither mode imports ref_semantics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pulse_ir.ir import (
    Config,
    Play,
    Acquire,
    ShiftPhase,
    Delay,
    IfBit,
    PulseStmt,
)

if TYPE_CHECKING:
    from pulse_lowering.schedule import PulseEvent


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
    compiled_events: list[PulseEvent] | None = None,
) -> tuple[bool, list[str]]:
    """Check that every IfBit uses a cbit whose acquire has already completed.

    If compiled_events is None (source-level mode):
        Independently tracks frame times from the program AST.
    If compiled_events is provided (compiled mode):
        Checks the schedule directly: for each event with conditional_on set,
        verifies event.start >= cbit_ready derived from acquire events.

    Returns (ok, errors).
    """
    errors: list[str] = []

    if compiled_events is not None:
        _check_compiled_events(compiled_events, errors)
    else:
        frame_time: dict[str, int] = {f: 0 for f in config.frames}
        cbit_ready: dict[str, int] = {}
        _check_source(program, frame_time, cbit_ready, errors)

    return (len(errors) == 0, errors)


def _check_source(
    program: list[PulseStmt],
    frame_time: dict[str, int],
    cbit_ready: dict[str, int],
    errors: list[str],
) -> None:
    """Source-level check: independently track timing from program AST."""
    for stmt in program:
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
                _check_source([body], frame_time, cbit_ready, errors)


def _check_compiled_events(
    events: list[PulseEvent],
    errors: list[str],
) -> None:
    """Compiled mode: check feedback causality directly on the schedule.

    1. Scan acquire events to build cbit_ready map.
    2. For each event with non-empty conditional_on, check start >= cbit_ready
       for ALL cbits in the dependency set (handles nested IfBit).
    """
    # Build cbit_ready from acquire events
    cbit_ready: dict[str, int] = {}
    for ev in events:
        if ev.kind == "acquire" and ev.cbit is not None:
            cbit_ready[ev.cbit] = ev.end

    # Check conditional events
    for ev in events:
        if ev.conditional_on:
            for cbit in ev.conditional_on:
                t_ready = cbit_ready.get(cbit)
                if t_ready is None:
                    errors.append(
                        f"Event {ev.event_id} ({ev.kind} on {ev.frame}): "
                        f"conditional on {cbit} but cbit was never acquired"
                    )
                elif ev.start < t_ready:
                    errors.append(
                        f"Event {ev.event_id} ({ev.kind} on {ev.frame}): "
                        f"starts at t={ev.start} but {cbit} not ready "
                        f"until t={t_ready}"
                    )
