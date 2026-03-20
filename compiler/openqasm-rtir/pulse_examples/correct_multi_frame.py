"""Correct example: two frames on different ports, with phase shift.

Frame d0 on p0, frame d1 on p1. Independent plays + ShiftPhase.
Expected: all three checks PASS (no port sharing, no feedback).
"""

import math
from pulse_ir.ir import Config, Waveform, Play, ShiftPhase, Delay

config = Config(
    frames=frozenset(["d0", "d1"]),
    ports=frozenset(["p0", "p1"]),
    port_of={"d0": "p0", "d1": "p1"},
    init_freq={"d0": 5.0e-3, "d1": 5.1e-3},
    init_phase={"d0": 0.0, "d1": 0.0},
)

program = [
    Play("d0", Waveform("gaussian_160", duration=160)),
    ShiftPhase("d0", math.pi / 2),
    Play("d0", Waveform("gaussian_160", duration=160)),
    # d1 operates independently
    Play("d1", Waveform("drag_200", duration=200)),
    Delay(duration=100, frame="d1"),
    Play("d1", Waveform("drag_200", duration=200)),
]
