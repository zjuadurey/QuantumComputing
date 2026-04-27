"""P3 workload definitions.

P3 keeps workloads deliberately small: each workload is a named set of knobs for
the existing Stim repetition-code backend plus a lightweight runtime service
model. The goal is breadth across execution shapes, not a new compiler.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Workload:
    workload_id: str
    circuit_id: str
    distance: int
    num_rounds: int
    data_error_rate: float
    measurement_error_rate: float
    idle_error_rate: float
    detector_load_scale: float
    idle_window_scale: float
    deadline_scale: float


WORKLOADS = {
    "memory_x": Workload(
        workload_id="memory_x",
        circuit_id="stim_repetition_memory_x",
        distance=5,
        num_rounds=8,
        data_error_rate=0.030,
        measurement_error_rate=0.020,
        idle_error_rate=0.0020,
        detector_load_scale=1.00,
        idle_window_scale=1.00,
        deadline_scale=1.00,
    ),
    "memory_z": Workload(
        workload_id="memory_z",
        circuit_id="stim_repetition_memory_z",
        distance=7,
        num_rounds=8,
        data_error_rate=0.025,
        measurement_error_rate=0.025,
        idle_error_rate=0.0020,
        detector_load_scale=1.08,
        idle_window_scale=0.95,
        deadline_scale=1.05,
    ),
    "burst_sensitive_memory": Workload(
        workload_id="burst_sensitive_memory",
        circuit_id="stim_repetition_burst_sensitive",
        distance=5,
        num_rounds=12,
        data_error_rate=0.035,
        measurement_error_rate=0.025,
        idle_error_rate=0.0025,
        detector_load_scale=1.35,
        idle_window_scale=1.05,
        deadline_scale=0.90,
    ),
    "high_detector_load_memory": Workload(
        workload_id="high_detector_load_memory",
        circuit_id="stim_repetition_high_detector_load",
        distance=7,
        num_rounds=12,
        data_error_rate=0.035,
        measurement_error_rate=0.030,
        idle_error_rate=0.0020,
        detector_load_scale=1.55,
        idle_window_scale=1.00,
        deadline_scale=0.85,
    ),
    "idle_heavy_memory": Workload(
        workload_id="idle_heavy_memory",
        circuit_id="stim_repetition_idle_heavy",
        distance=5,
        num_rounds=10,
        data_error_rate=0.025,
        measurement_error_rate=0.020,
        idle_error_rate=0.0040,
        detector_load_scale=1.05,
        idle_window_scale=1.45,
        deadline_scale=0.95,
    ),
}


def get_workload(workload_id: str) -> Workload:
    try:
        return WORKLOADS[workload_id]
    except KeyError as exc:
        choices = ", ".join(sorted(WORKLOADS))
        raise ValueError(f"unknown workload {workload_id!r}; choose from: {choices}") from exc


def parse_workload_ids(value: str) -> list[str]:
    ids = [item.strip() for item in value.split(",") if item.strip()]
    for workload_id in ids:
        get_workload(workload_id)
    return ids

