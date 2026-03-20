"""Violation example: two frames share a port with overlapping play.

Frame d0 and d1 BOTH map to port p0.
d0 plays [0, 160), d1 plays [0, 200) — overlap on p0.
Expected: port_exclusivity FAILS.
"""

from pulse_ir.ir import Config, Waveform, Play

config = Config(
    frames=frozenset(["d0", "d1"]),
    ports=frozenset(["p0"]),
    port_of={"d0": "p0", "d1": "p0"},  # both on same port!
    init_freq={"d0": 5.0e-3, "d1": 5.1e-3},
    init_phase={"d0": 0.0, "d1": 0.0},
)

program = [
    Play("d0", Waveform("gaussian_160", duration=160)),
    Play("d1", Waveform("drag_200", duration=200)),
    # d0 occupies p0 at [0,160), d1 occupies p0 at [0,200) — CONFLICT
]
