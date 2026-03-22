"""Shared port example for lowering bug demonstration.

v0.4: With port-aware semantics, this program is CORRECT — the oracle
serializes shared-port access. This example is used with
lower_buggy_ignore_shared_port() to demonstrate that PortExcl catches
lowering bugs that ignore port serialization.
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
    # Correct lowering: d0=[0,160), d1=[160,360) — serialized by port_time
    # Buggy lowering (ignore_shared_port): d0=[0,160), d1=[0,200) — OVERLAP
]
