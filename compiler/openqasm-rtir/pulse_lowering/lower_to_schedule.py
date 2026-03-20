"""Lower PulseStmt program → explicit timed PulseEvent schedule.

Minimal sequential lowering: walks the program in order, tracks per-frame
time and phase, emits one PulseEvent per statement.

No reordering, no optimization. Purpose: produce a concrete schedule
that can be checked against source program semantics.

This module does NOT import ref_semantics — it is an independent
compilation path whose output is verified by the checkers.
"""

from __future__ import annotations

import math

from pulse_ir.ir import (
    Config,
    Play,
    Acquire,
    ShiftPhase,
    Delay,
    IfBit,
    PulseStmt,
)
from pulse_lowering.schedule import PulseEvent

TWO_PI = 2.0 * math.pi


def lower_to_schedule(
    program: list[PulseStmt],
    config: Config,
) -> list[PulseEvent]:
    """Lower a PulseStmt program to an explicit timed schedule."""
    time: dict[str, int] = {f: 0 for f in config.frames}
    phase: dict[str, float] = {f: config.init_phase[f] for f in config.frames}
    cbit_ready: dict[str, int] = {}

    events: list[PulseEvent] = []
    next_id = 0
    # Track IfBit context: accumulate all active cbits for nested conditionals
    active_cbits: frozenset[str] = frozenset()

    def emit(stmt: PulseStmt) -> None:
        nonlocal next_id, active_cbits

        match stmt:
            case Play(frame, waveform):
                t = time[frame]
                d = waveform.duration
                p = config.port_of[frame]
                ph_before = phase[frame]
                time[frame] = t + d
                phase[frame] += TWO_PI * config.init_freq[frame] * d
                events.append(PulseEvent(
                    event_id=next_id, kind="play", frame=frame, port=p,
                    start=t, end=t + d,
                    phase_before=ph_before, phase_after=phase[frame],
                    payload=waveform.name,
                    conditional_on=active_cbits,
                ))
                next_id += 1

            case Acquire(frame, duration, cbit):
                t = time[frame]
                p = config.port_of[frame]
                ph_before = phase[frame]
                time[frame] = t + duration
                phase[frame] += TWO_PI * config.init_freq[frame] * duration
                cbit_ready[cbit] = t + duration
                events.append(PulseEvent(
                    event_id=next_id, kind="acquire", frame=frame, port=p,
                    start=t, end=t + duration,
                    phase_before=ph_before, phase_after=phase[frame],
                    cbit=cbit,
                    conditional_on=active_cbits,
                ))
                next_id += 1

            case ShiftPhase(frame, angle):
                ph_before = phase[frame]
                phase[frame] += angle
                events.append(PulseEvent(
                    event_id=next_id, kind="shift_phase", frame=frame,
                    port=None,
                    start=time[frame], end=time[frame],
                    phase_before=ph_before, phase_after=phase[frame],
                    payload=f"{angle:.6f}",
                    conditional_on=active_cbits,
                ))
                next_id += 1

            case Delay(duration, frame):
                t = time[frame]
                ph_before = phase[frame]
                time[frame] = t + duration
                phase[frame] += TWO_PI * config.init_freq[frame] * duration
                events.append(PulseEvent(
                    event_id=next_id, kind="delay", frame=frame,
                    port=None, start=t, end=t + duration,
                    phase_before=ph_before, phase_after=phase[frame],
                    conditional_on=active_cbits,
                ))
                next_id += 1

            case IfBit(cbit, body):
                # Add this cbit to the active set (accumulate for nesting)
                prev_cbits = active_cbits
                active_cbits = active_cbits | {cbit}
                emit(body)
                active_cbits = prev_cbits

    for stmt in program:
        emit(stmt)

    return events
