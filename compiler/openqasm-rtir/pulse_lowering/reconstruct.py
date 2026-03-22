"""Reconstruct a FrameState from a PulseEvent schedule.

This bridges lowering output → checker input.
The checkers expect a FrameState; this function builds one from
the explicit timed events produced by lower_to_schedule.

Independent from both ref_semantics and lower_to_schedule logic —
it reads the event list as opaque data.
"""

from __future__ import annotations

from pulse_ir.ir import Config, FrameState
from pulse_lowering.schedule import PulseEvent


def reconstruct_state(
    events: list[PulseEvent],
    config: Config,
) -> FrameState:
    """Build a FrameState from a list of PulseEvents.

    Extracts final time, phase, port_time, occupancy, and cbit_ready from events.
    """
    state = FrameState.initial(config)

    for ev in events:
        f = ev.frame
        if f in config.frames:
            # Take the latest time and phase seen for each frame
            if ev.end > state.time.get(f, 0):
                state.time[f] = ev.end
            state.phase[f] = ev.phase_after

        # Port occupancy and port_time: only for play and acquire
        if ev.port is not None and ev.duration > 0:
            state.occupancy[ev.port].append((ev.start, ev.end))
            if ev.end > state.port_time.get(ev.port, 0):
                state.port_time[ev.port] = ev.end

        # cbit readiness: from acquire events
        if ev.kind == "acquire" and ev.cbit is not None:
            state.cbit[ev.cbit] = None
            state.cbit_ready[ev.cbit] = ev.end

    return state
