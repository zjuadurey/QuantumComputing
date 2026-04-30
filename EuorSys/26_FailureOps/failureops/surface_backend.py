"""Small rotated surface-code pilot backend for P5c."""

from __future__ import annotations

import json

from failureops.runtime_service import compute_runtime_service_fields, get_stress_config, recompute_p3_failure_behavior
from failureops.workloads import get_workload


def generate_surface_code_runs(
    *,
    workload_id: str,
    stress_level: str,
    num_shots: int,
    seed: int,
    run_id: str,
    basis: str = "x",
) -> list[dict[str, object]]:
    from failureops.qec_backend import sample_and_decode
    import stim

    workload = get_workload(workload_id)
    stress = get_stress_config(stress_level)
    circuit = stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        distance=workload.distance,
        rounds=workload.num_rounds,
        after_clifford_depolarization=workload.idle_error_rate,
        before_round_data_depolarization=workload.data_error_rate,
        before_measure_flip_probability=workload.measurement_error_rate,
        after_reset_flip_probability=workload.measurement_error_rate * 0.5,
    )
    detector_samples, observable_samples, predictions = sample_and_decode(
        circuit=circuit,
        num_shots=num_shots,
        seed=seed,
    )

    rows = []
    for shot_id in range(num_shots):
        shot_seed = seed + shot_id
        detector_indices = [
            index for index, value in enumerate(detector_samples[shot_id]) if bool(value)
        ]
        row: dict[str, object] = {
            "run_id": run_id,
            "workload_id": workload.workload_id,
            "circuit_id": workload.circuit_id,
            "backend": "stim_surface_code+pymatching+p3_runtime_service",
            "code_family": "rotated_surface_code_memory",
            "distance": workload.distance,
            "num_rounds": workload.num_rounds,
            "shot_id": shot_id,
            "seed": shot_seed,
            "stress_level": stress_level,
            "data_error_rate": workload.data_error_rate,
            "measurement_error_rate": workload.measurement_error_rate,
            "idle_error_rate": workload.idle_error_rate,
            "decoder_timeout_base_rate": stress["decoder_timeout_base_rate"],
            "detector_load_scale": workload.detector_load_scale,
            "idle_window_scale": workload.idle_window_scale,
            "deadline_scale": workload.deadline_scale,
            "decoder_workers": stress["decoder_workers"],
            "decoder_service_rate": stress["decoder_service_rate"],
            "decoder_queue_depth": stress["decoder_queue_depth"],
            "decoder_deadline": stress["decoder_deadline"] * workload.deadline_scale,
            "detector_count": circuit.num_detectors,
            "detector_event_count": len(detector_indices),
            "detector_events": json.dumps(detector_indices, separators=(",", ":")),
            "observable_flip": bool(observable_samples[shot_id][0]),
            "decoder_prediction": bool(predictions[shot_id][0]),
        }
        row.update(compute_runtime_service_fields(row))
        rows.append(recompute_p3_failure_behavior(row))
    return rows

