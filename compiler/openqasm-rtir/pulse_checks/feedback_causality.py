"""Feedback Causality checker.

Property (formal_definitions_v0.md §2.2):
    FeedbackCausal(σ, P) ≡ ∀ IfBit(c, body) in P:
                              t_use ≥ σ.cbit_ready(c)

where t_use is the frame time when body would execute.

Two modes:
  1. Source-level: independently tracks frame times from the program AST.
     v0.4 note: body start is port-aware for Play/Acquire.
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


def _body_start(
    stmt: PulseStmt,
    frame_time: dict[str, int],
    port_time: dict[str, int],
    config: Config,
) -> int | None:
    """Compute the real start time of a guarded body event at encounter."""
    match stmt:
        case Play(frame, _):
            p = config.port_of[frame]
            return max(frame_time[frame], port_time[p])
        case Acquire(frame, _, _):
            p = config.port_of[frame]
            return max(frame_time[frame], port_time[p])
        case ShiftPhase(frame, _):
            return frame_time[frame]
        case Delay(_, frame):
            return frame_time[frame]
        case IfBit(_, body):
            return _body_start(body, frame_time, port_time, config)
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
        port_time: dict[str, int] = {p: 0 for p in config.ports}
        cbit_ready: dict[str, int] = {}
        _check_source(program, frame_time, port_time, cbit_ready, config, errors)

    return (len(errors) == 0, errors)


def _check_source(
    program: list[PulseStmt],
    frame_time: dict[str, int],
    port_time: dict[str, int],
    cbit_ready: dict[str, int],
    config: Config,
    errors: list[str],
) -> None:
    """Source-level check: independently track timing from program AST."""
    for stmt in program:
        match stmt:
            case Play(frame, waveform):
                p = config.port_of[frame]
                start = max(frame_time[frame], port_time[p])
                end = start + waveform.duration
                frame_time[frame] = end
                port_time[p] = end

            case Acquire(frame, duration, cbit):
                p = config.port_of[frame]
                start = max(frame_time[frame], port_time[p])
                end = start + duration
                frame_time[frame] = end
                port_time[p] = end
                cbit_ready[cbit] = end

            case ShiftPhase():
                pass  # zero duration

            case Delay(duration, frame):
                frame_time[frame] += duration

            case IfBit(cbit, body):
                f_body = _body_frame(body)
                t_use = _body_start(body, frame_time, port_time, config)
                if f_body is not None and t_use is not None:
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
                _check_source([body], frame_time, port_time, cbit_ready, config, errors)


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
