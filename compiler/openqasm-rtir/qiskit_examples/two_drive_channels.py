"""Two-drive-channel Qiskit pulse example in its own file.

Source:
- IBM Quantum / Qiskit pulse guide
  https://qiskit.qotlabs.org/docs/guides/pulse
- Qiskit Dynamics tutorial
  "Simulating Qiskit Pulse Schedules with Qiskit Dynamics"
  https://qiskit-community.github.io/qiskit-dynamics/tutorials/qiskit_pulse.html

Adaptation notes:
- uses the official pulse-builder constructs shown in those sources
- specialized to two independent drive channels so the adapter can be
  tested on multiple frame ids without introducing unsupported features
"""

from __future__ import annotations

import warnings


def build_schedule():
    from qiskit import pulse
    from qiskit.pulse import DriveChannel
    from qiskit.pulse.library import Constant

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pulse.build(name="two_drive_channels") as sched:
            d0 = DriveChannel(0)
            d1 = DriveChannel(1)
            pulse.play(Constant(64, 0.2), d0)
            pulse.play(Constant(96, 0.1), d1)
            pulse.shift_phase(0.25, d1)
    return sched
