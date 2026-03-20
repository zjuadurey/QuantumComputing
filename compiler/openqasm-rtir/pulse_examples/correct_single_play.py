"""Correct example: single frame, single play.

One frame (d0) on one port (p0). Play a 160dt Gaussian pulse.
Expected: all three checks PASS.
"""

from pulse_ir.ir import Config, Waveform, Play

config = Config(
    frames=frozenset(["d0"]),
    ports=frozenset(["p0"]),
    port_of={"d0": "p0"},
    init_freq={"d0": 5.0e-3},   # 5 MHz in GHz-scale units (arbitrary for model)
    init_phase={"d0": 0.0},
)

program = [
    Play("d0", Waveform("gaussian_160", duration=160)),
]
