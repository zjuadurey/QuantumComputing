"""Correct example: two frames share a port — serialized by port_time.

v0.4: With port-aware semantics, Play(d0) at [0,160) and Play(d1) at
[160,360) because d1 waits for port p0 to be free.
Expected: all three checks PASS (no overlap on p0).
"""

from pulse_ir.ir import Config, Waveform, Play

config = Config(
    frames=frozenset(["d0", "d1"]),
    ports=frozenset(["p0"]),
    port_of={"d0": "p0", "d1": "p0"},  # both on same port
    init_freq={"d0": 5.0e-3, "d1": 5.1e-3},
    init_phase={"d0": 0.0, "d1": 0.0},
)

program = [
    Play("d0", Waveform("gaussian_160", duration=160)),
    Play("d1", Waveform("drag_200", duration=200)),
    # v0.4: d0 occupies p0 at [0,160), d1 waits and occupies p0 at [160,360)
    # No conflict — port_time serializes access.
]
