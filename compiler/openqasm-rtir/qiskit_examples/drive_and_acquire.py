"""Drive-plus-acquire Qiskit pulse example in its own file.

Source:
- Qiskit Dynamics tutorial
  "Simulating backends at the pulse-level with DynamicsBackend"
  https://qiskit-community.github.io/qiskit-dynamics/tutorials/dynamics_backend.html

Adaptation notes:
- derived from the tutorial's pulse schedule example in the
  ``pulse.build()`` loop
- intentionally removes ``shift_frequency(...)`` because the current
  core IR models fixed-frequency frames only
- stays within the currently supported subset:
  ``play``, ``acquire``, ``delay``, and ``shift_phase``
"""

from __future__ import annotations

import warnings


def build_schedule():
    from qiskit import pulse
    from qiskit.pulse import AcquireChannel, DriveChannel, MemorySlot
    from qiskit.pulse.library import Constant

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pulse.build(name="drive_and_acquire") as sched:
            d0 = DriveChannel(0)
            a0 = AcquireChannel(0)
            pulse.play(Constant(160, 0.2), d0)
            pulse.acquire(1000, a0, MemorySlot(0))
            pulse.delay(48, d0)
            pulse.shift_phase(1.0, d0)
    return sched
