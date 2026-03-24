"""Single-drive Qiskit pulse example in its own file.

Source:
- Qiskit Dynamics tutorial
  "Simulating Qiskit Pulse Schedules with Qiskit Dynamics"
  https://qiskit-community.github.io/qiskit-dynamics/tutorials/qiskit_pulse.html

Adaptation notes:
- derived from the tutorial's ``sx-sy schedule`` pattern
- keeps only the fixed-frequency subset needed by this artifact
- stays within the currently supported subset:
  ``play``, ``delay``, and ``shift_phase``
"""

from __future__ import annotations

import warnings


def build_schedule():
    from qiskit import pulse
    from qiskit.pulse import DriveChannel
    from qiskit.pulse.library import Constant

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pulse.build(name="single_drive_with_virtual_z") as sched:
            d0 = DriveChannel(0)
            pulse.play(Constant(160, 0.2), d0)
            pulse.delay(32, d0)
            pulse.shift_phase(0.5, d0)
            pulse.play(Constant(80, 0.15), d0)
    return sched
