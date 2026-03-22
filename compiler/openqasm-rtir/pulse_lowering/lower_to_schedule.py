"""Lower PulseStmt program → explicit timed PulseEvent schedule.

v0.4: Port-aware sequential lowering. Play/Acquire start at
max(frame_time, port_time). Phase evolves during implicit stall.

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
    port_time: dict[str, int] = {p: 0 for p in config.ports}
    cbit_ready: dict[str, int] = {}

    events: list[PulseEvent] = []
    next_id = 0
    active_cbits: frozenset[str] = frozenset()

    def emit(stmt: PulseStmt) -> None:
        nonlocal next_id, active_cbits

        match stmt:
            case Play(frame, waveform):
                d = waveform.duration
                p = config.port_of[frame]
                freq = config.init_freq[frame]
                # Port-aware: wait for port to be free
                start = max(time[frame], port_time[p])
                end = start + d
                ph_before = phase[frame]
                # Phase evolves for entire advance (stall + operation)
                total_advance = end - time[frame]
                phase[frame] += TWO_PI * freq * total_advance
                time[frame] = end
                port_time[p] = end
                events.append(PulseEvent(
                    event_id=next_id, kind="play", frame=frame, port=p,
                    start=start, end=end,
                    phase_before=ph_before, phase_after=phase[frame],
                    payload=waveform.name,
                    conditional_on=active_cbits,
                ))
                next_id += 1

            case Acquire(frame, duration, cbit):
                p = config.port_of[frame]
                freq = config.init_freq[frame]
                start = max(time[frame], port_time[p])
                end = start + duration
                ph_before = phase[frame]
                total_advance = end - time[frame]
                phase[frame] += TWO_PI * freq * total_advance
                time[frame] = end
                port_time[p] = end
                cbit_ready[cbit] = end
                events.append(PulseEvent(
                    event_id=next_id, kind="acquire", frame=frame, port=p,
                    start=start, end=end,
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
                freq = config.init_freq[frame]
                t = time[frame]
                ph_before = phase[frame]
                time[frame] = t + duration
                phase[frame] += TWO_PI * freq * duration
                events.append(PulseEvent(
                    event_id=next_id, kind="delay", frame=frame,
                    port=None, start=t, end=t + duration,
                    phase_before=ph_before, phase_after=phase[frame],
                    conditional_on=active_cbits,
                ))
                next_id += 1

            case IfBit(cbit, body):
                prev_cbits = active_cbits
                active_cbits = active_cbits | {cbit}
                emit(body)
                active_cbits = prev_cbits

    for stmt in program:
        emit(stmt)

    return events
