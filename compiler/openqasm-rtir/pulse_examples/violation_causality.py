"""Violation example: conditional play before acquire finishes.

Acquire on m0 takes 1000 dt, but IfBit fires on d0 at t=0
(no delay to wait for result).
Expected: feedback_causality FAILS.
"""

from pulse_ir.ir import Config, Waveform, Play, Acquire, IfBit

config = Config(
    frames=frozenset(["d0", "m0"]),
    ports=frozenset(["p_drive", "p_meas"]),
    port_of={"d0": "p_drive", "m0": "p_meas"},
    init_freq={"d0": 5.0e-3, "m0": 7.0e-3},
    init_phase={"d0": 0.0, "m0": 0.0},
)

program = [
    Acquire("m0", duration=1000, cbit="c0"),
    # NO delay on d0 — d0.time is still 0
    IfBit("c0", Play("d0", Waveform("x_pulse", duration=160))),
    # t_use=0 < cbit_ready=1000 — VIOLATION
]
