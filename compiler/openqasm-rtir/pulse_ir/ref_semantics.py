"""Reference semantics (oracle) for pulse-level programs.

Implements the v0.4 step rules with port-aware timing:
- Play/Acquire: start = max(time[f], port_time[p]), phase evolves during stall
- ShiftPhase: zero duration, no port change
- Delay: advances frame time, no port change
- IfBit: always-taken, no auto-wait for cbit_ready

This is the independent oracle — checkers must NOT reuse this code path.
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
)

TWO_PI = 2.0 * math.pi


def step(state: FrameState, stmt: PulseStmt, config: Config) -> FrameState:
    """Execute a single statement, returning a new state.

    Mutates nothing — returns a fresh FrameState.
    """
    s = state.copy()

    match stmt:
        case Play(frame, waveform):
            d = waveform.duration
            p = config.port_of[frame]
            freq = config.init_freq[frame]
            # Port-aware: wait for port to be free
            start = max(s.time[frame], s.port_time[p])
            end = start + d
            # Phase evolves for entire advance (stall + operation)
            total_advance = end - s.time[frame]
            s.phase[frame] += TWO_PI * freq * total_advance
            s.time[frame] = end
            s.port_time[p] = end
            s.occupancy[p].append((start, end))

        case Acquire(frame, duration, cbit):
            p = config.port_of[frame]
            freq = config.init_freq[frame]
            start = max(s.time[frame], s.port_time[p])
            end = start + duration
            total_advance = end - s.time[frame]
            s.phase[frame] += TWO_PI * freq * total_advance
            s.time[frame] = end
            s.port_time[p] = end
            s.occupancy[p].append((start, end))
            s.cbit[cbit] = None
            s.cbit_ready[cbit] = end

        case ShiftPhase(frame, angle):
            # Zero duration, no time advance, no port activity
            s.phase[frame] += angle

        case Delay(duration, frame):
            freq = config.init_freq[frame]
            s.time[frame] += duration
            s.phase[frame] += TWO_PI * freq * duration
            # No port occupancy — delay is silence

        case IfBit(cbit, body):
            # Always-taken: execute body unconditionally for conservative trace.
            # Does NOT auto-wait for cbit_ready.
            s = step(s, body, config)

        case _:
            raise ValueError(f"Unknown statement type: {type(stmt)}")

    return s


def run(program: list[PulseStmt], config: Config) -> FrameState:
    """Execute a full program sequentially, returning the final state.

    This is the reference semantics: σₙ = step(...step(step(σ₀, s₁), s₂)..., sₙ)
    """
    state = FrameState.initial(config)
    for stmt in program:
        state = step(state, stmt, config)
    return state
