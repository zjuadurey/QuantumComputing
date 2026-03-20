"""Correct example: acquire then conditional play (feedback).

Frame d0 (drive) and m0 (measure) on separate ports.
Acquire on m0, then IfBit conditional play on d0.
Delay on d0 ensures causality: d0 waits until acquire finishes.
Expected: all three checks PASS.
"""

from pulse_ir.ir import Config, Waveform, Play, Acquire, Delay, IfBit

config = Config(
    frames=frozenset(["d0", "m0"]),
    ports=frozenset(["p_drive", "p_meas"]),
    port_of={"d0": "p_drive", "m0": "p_meas"},
    init_freq={"d0": 5.0e-3, "m0": 7.0e-3},
    init_phase={"d0": 0.0, "m0": 0.0},
)

program = [
    # Acquire on measure frame: 1000 dt
    Acquire("m0", duration=1000, cbit="c0"),
    # Delay on drive frame to wait for acquire to finish
    Delay(duration=1000, frame="d0"),
    # Now conditional play — d0.time=1000 >= cbit_ready(c0)=1000, OK
    IfBit("c0", Play("d0", Waveform("x_pulse", duration=160))),
]
