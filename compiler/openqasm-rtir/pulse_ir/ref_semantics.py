"""Reference semantics (oracle) for pulse-level programs.

Implements the step rules from formal_definitions_v0.md §1.2.
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
            t = s.time[frame]
            d = waveform.duration
            p = config.port_of[frame]
            s.time[frame] = t + d
            s.phase[frame] += TWO_PI * config.init_freq[frame] * d
            s.occupancy[p].append((t, t + d))

        case Acquire(frame, duration, cbit):
            t = s.time[frame]
            p = config.port_of[frame]
            s.time[frame] = t + duration
            s.phase[frame] += TWO_PI * config.init_freq[frame] * duration
            s.occupancy[p].append((t, t + duration))
            # Value is non-deterministic at hardware level; we leave it as None
            s.cbit[cbit] = None
            s.cbit_ready[cbit] = t + duration

        case ShiftPhase(frame, angle):
            # Zero duration, no time advance, no port activity
            s.phase[frame] += angle

        case Delay(duration, frame):
            s.time[frame] += duration
            s.phase[frame] += TWO_PI * config.init_freq[frame] * duration
            # No port occupancy — delay is silence

        case IfBit(cbit, body):
            # Feedback: execute body only if cbit is set
            # For oracle: we execute body unconditionally to track timing,
            # but record the dependency for causality checking.
            # In a real system, execution depends on cbit value.
            # Here we always execute to produce a conservative trace.
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
