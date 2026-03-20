"""Violation example: frame consistency — tampered phase in state.

Run a correct program through the oracle, then manually corrupt the phase.
The checker should detect that phase(f) ≠ expected.
Expected: frame_consistency FAILS.

NOTE: In real use, this scenario arises when a *compiler* produces a schedule
whose phase tracking diverges from the reference semantics.
Here we simulate it by post-hoc tampering.
"""

import math
from pulse_ir.ir import Config, Waveform, Play, ShiftPhase

config = Config(
    frames=frozenset(["d0"]),
    ports=frozenset(["p0"]),
    port_of={"d0": "p0"},
    init_freq={"d0": 5.0e-3},
    init_phase={"d0": 0.0},
)

program = [
    Play("d0", Waveform("gaussian_160", duration=160)),
    ShiftPhase("d0", math.pi / 2),
    Play("d0", Waveform("gaussian_160", duration=160)),
]

# The phase corruption is applied in the test after running the oracle.
# See tests/test_pulse.py for usage.
phase_corruption = 0.1  # add this to oracle phase to simulate compiler bug
