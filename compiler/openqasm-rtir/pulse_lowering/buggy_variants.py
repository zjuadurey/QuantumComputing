"""Buggy lowering variants — simulate compiler bugs.

Each function takes the same (program, config) signature as lower_to_schedule
but introduces a specific, documented bug. Used to demonstrate that
checkers catch real compilation errors.

These do NOT import ref_semantics.
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
from pulse_lowering.lower_to_schedule import lower_to_schedule

TWO_PI = 2.0 * math.pi


def lower_buggy_drop_phase(
    program: list[PulseStmt],
    config: Config,
) -> list[PulseEvent]:
    """Bug: silently drops all ShiftPhase instructions.

    Simulates a compiler that loses virtual-Z gates.
    Caught by: FrameConsist (phase drift).
    """
    filtered = [s for s in program if not isinstance(s, ShiftPhase)]
    return lower_to_schedule(filtered, config)


def lower_buggy_extra_delay(
    program: list[PulseStmt],
    config: Config,
    extra_dt: int = 50,
) -> list[PulseEvent]:
    """Bug: inserts an extra delay on every frame before the program.

    Simulates a compiler that shifts all timing.
    Caught by: FrameConsist (time drift).
    """
    extra: list[PulseStmt] = [Delay(extra_dt, f) for f in config.frames]
    return lower_to_schedule(extra + list(program), config)


def lower_buggy_reorder_ports(
    program: list[PulseStmt],
    config: Config,
) -> list[PulseEvent]:
    """Bug: emits all Play/Acquire events at time 0 (ignores sequencing).

    Simulates a compiler that flattens everything to t=0.
    Caught by: PortExcl (overlapping intervals on shared ports).
    """
    events: list[PulseEvent] = []
    eid = 0

    for stmt in program:
        match stmt:
            case Play(frame, waveform):
                p = config.port_of[frame]
                ph = config.init_phase[frame]
                d = waveform.duration
                events.append(PulseEvent(
                    event_id=eid, kind="play", frame=frame, port=p,
                    start=0, end=d,
                    phase_before=ph,
                    phase_after=ph + TWO_PI * config.init_freq[frame] * d,
                    payload=waveform.name,
                ))
                eid += 1

            case Acquire(frame, duration, cbit):
                p = config.port_of[frame]
                ph = config.init_phase[frame]
                events.append(PulseEvent(
                    event_id=eid, kind="acquire", frame=frame, port=p,
                    start=0, end=duration,
                    phase_before=ph,
                    phase_after=ph + TWO_PI * config.init_freq[frame] * duration,
                    cbit=cbit,
                ))
                eid += 1

            case ShiftPhase(frame, angle):
                ph = config.init_phase[frame]
                events.append(PulseEvent(
                    event_id=eid, kind="shift_phase", frame=frame,
                    port=None, start=0, end=0,
                    phase_before=ph, phase_after=ph + angle,
                ))
                eid += 1

            case Delay(duration, frame):
                ph = config.init_phase[frame]
                events.append(PulseEvent(
                    event_id=eid, kind="delay", frame=frame,
                    port=None, start=0, end=duration,
                    phase_before=ph,
                    phase_after=ph + TWO_PI * config.init_freq[frame] * duration,
                ))
                eid += 1

            case IfBit(_, body):
                sub = lower_buggy_reorder_ports([body], config)
                for ev in sub:
                    events.append(PulseEvent(
                        event_id=eid, kind=ev.kind, frame=ev.frame,
                        port=ev.port, start=ev.start, end=ev.end,
                        phase_before=ev.phase_before,
                        phase_after=ev.phase_after,
                        cbit=ev.cbit, payload=ev.payload,
                    ))
                    eid += 1

    return events


def lower_buggy_early_feedback(
    program: list[PulseStmt],
    config: Config,
) -> list[PulseEvent]:
    """Bug: moves IfBit before the preceding Delay (reorder).

    Simulates a compiler that reorders operations for "optimization"
    without respecting measurement-feedback dependency.

    Caught by: FeedbackCausal (conditional event starts before cbit_ready).
    Not caught by: PortExcl, FrameConsist (same total time/phase).
    """
    reordered: list[PulseStmt] = list(program)
    for i in range(len(reordered) - 1):
        if isinstance(reordered[i], Delay) and isinstance(reordered[i + 1], IfBit):
            reordered[i], reordered[i + 1] = reordered[i + 1], reordered[i]
    return lower_to_schedule(reordered, config)


def lower_buggy_ignore_shared_port(
    program: list[PulseStmt],
    config: Config,
) -> list[PulseEvent]:
    """Bug: ignores port_time — uses only frame-local time for scheduling.

    Simulates a compiler that doesn't model shared-port serialization.
    Two frames sharing a port will produce overlapping events.

    Caught by: PortExcl (overlapping intervals on shared port).
    Also caught by: FrameConsist (wrong time due to missing stall).
    """
    time: dict[str, int] = {f: 0 for f in config.frames}
    phase: dict[str, float] = {f: config.init_phase[f] for f in config.frames}

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
                # BUG: uses only frame time, ignores port_time
                start = time[frame]
                end = start + d
                ph_before = phase[frame]
                phase[frame] += TWO_PI * freq * d
                time[frame] = end
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
                start = time[frame]
                end = start + duration
                ph_before = phase[frame]
                phase[frame] += TWO_PI * freq * duration
                time[frame] = end
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
