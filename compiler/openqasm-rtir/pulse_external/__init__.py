"""External corroboration helpers for v0.5.

These modules are intentionally secondary evidence layers: they do not define
the source semantics or replace the contract checkers.
"""

from pulse_external.qiskit_dynamics import (
    DynamicsCorroborationResult,
    compare_schedule_lowerings,
    compare_single_frame_lowerings,
    simulate_schedule,
    simulate_single_frame_schedule,
)

__all__ = [
    "DynamicsCorroborationResult",
    "compare_schedule_lowerings",
    "compare_single_frame_lowerings",
    "simulate_schedule",
    "simulate_single_frame_schedule",
]
