"""P3 lightweight decoder runtime service model.

This module is intentionally a systems proxy rather than a full scheduler. It
turns each detector-event record into a small decoder-service trace with worker
capacity, queueing, deadline miss, timeout, and idle-exposure fields. P3
interventions replay the same shot material and change one runtime or noise
factor at a time.
"""

from __future__ import annotations

import json
import random
from copy import deepcopy

from failureops.io_utils import fmt_float, parse_bool, parse_float, parse_int
from failureops.workloads import Workload, get_workload

P3_STRESS_CONFIGS = {
    "low": {
        "decoder_timeout_base_rate": 0.004,
        "decoder_workers": 4,
        "decoder_service_rate": 5.5,
        "decoder_queue_depth": 14,
        "decoder_deadline": 0.90,
    },
    "medium": {
        "decoder_timeout_base_rate": 0.020,
        "decoder_workers": 2,
        "decoder_service_rate": 3.5,
        "decoder_queue_depth": 7,
        "decoder_deadline": 0.58,
    },
    "high": {
        "decoder_timeout_base_rate": 0.040,
        "decoder_workers": 1,
        "decoder_service_rate": 2.4,
        "decoder_queue_depth": 4,
        "decoder_deadline": 0.42,
    },
}

P3_NOISE_INTERVENTIONS = [
    "remove_data_noise",
    "weaken_data_noise_50pct",
    "remove_measurement_noise",
    "weaken_measurement_noise_50pct",
    "remove_idle_noise",
    "weaken_idle_noise_50pct",
]

P3_RUNTIME_INTERVENTIONS = [
    "remove_decoder_timeout",
    "increase_decoder_workers_2x",
    "increase_decoder_service_rate_2x",
    "remove_decoder_queueing",
    "relax_decoder_deadline_2x",
    "reduce_idle_exposure_50pct",
    "prioritize_high_weight_syndromes",
]

P3_INTERVENTIONS = [*P3_NOISE_INTERVENTIONS, *P3_RUNTIME_INTERVENTIONS]


def generate_p3_runs(
    *,
    workload_id: str,
    stress_level: str,
    num_shots: int,
    seed: int,
    run_id: str,
) -> list[dict[str, object]]:
    from failureops.qec_backend import build_repetition_memory_circuit, sample_and_decode

    workload = get_workload(workload_id)
    stress = get_stress_config(stress_level)
    circuit = build_repetition_memory_circuit(
        distance=workload.distance,
        num_rounds=workload.num_rounds,
        data_error_rate=workload.data_error_rate,
        measurement_error_rate=workload.measurement_error_rate,
        idle_error_rate=workload.idle_error_rate,
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
            "backend": "stim+pymatching+p3_runtime_service",
            "code_family": "repetition_code_memory",
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


def get_stress_config(stress_level: str) -> dict[str, float]:
    try:
        return P3_STRESS_CONFIGS[stress_level]
    except KeyError as exc:
        choices = ", ".join(sorted(P3_STRESS_CONFIGS))
        raise ValueError(f"unknown stress level {stress_level!r}; choose from: {choices}") from exc


def compute_runtime_service_fields(
    record: dict[str, object],
    *,
    worker_scale: float = 1.0,
    service_rate_scale: float = 1.0,
    queue_scale: float = 1.0,
    deadline_scale: float = 1.0,
    exposure_scale: float = 1.0,
    force_no_timeout: bool = False,
    remove_queueing: bool = False,
    prioritize_high_weight: bool = False,
) -> dict[str, object]:
    seed = parse_int(record["seed"])
    rng = random.Random(seed + 3_000_003)
    detector_event_count = parse_int(record["detector_event_count"])
    distance = parse_int(record["distance"])
    num_rounds = parse_int(record["num_rounds"])
    idle_error_rate = parse_float(record["idle_error_rate"])
    detector_load_scale = parse_float(record["detector_load_scale"], 1.0)
    idle_window_scale = parse_float(record["idle_window_scale"], 1.0)
    decoder_workers = max(1.0, parse_float(record["decoder_workers"], 1.0) * worker_scale)
    decoder_service_rate = max(
        0.25,
        parse_float(record["decoder_service_rate"], 1.0) * service_rate_scale,
    )
    decoder_queue_depth = parse_float(record["decoder_queue_depth"], 1.0)
    decoder_deadline = parse_float(record["decoder_deadline"], 0.5) * deadline_scale
    timeout_base_rate = parse_float(record["decoder_timeout_base_rate"], 0.0)

    scaled_load = detector_event_count * detector_load_scale
    priority_factor = 0.68 if prioritize_high_weight and scaled_load >= distance else 1.0
    effective_service_capacity = decoder_workers * decoder_service_rate
    offered_load = scaled_load * priority_factor
    queue_pressure = max(0.0, offered_load - effective_service_capacity)
    queue_overflow = queue_pressure > decoder_queue_depth
    decoder_backlog = 0.0 if remove_queueing else min(decoder_queue_depth, queue_pressure) * queue_scale

    jitter = rng.uniform(0.0, 0.025)
    arrival_time = 0.02 * (seed % 17)
    queue_wait = 0.0 if remove_queueing else decoder_backlog / max(0.25, effective_service_capacity)
    service_time = (0.06 + offered_load / max(0.25, effective_service_capacity)) * 0.18
    start_time = arrival_time + queue_wait
    finish_time = start_time + service_time + jitter
    latency = finish_time - arrival_time
    deadline_miss = latency > decoder_deadline or queue_overflow

    normalized_load = offered_load / max(1.0, distance * num_rounds)
    timeout_probability = min(
        0.95,
        timeout_base_rate
        + 0.24 * normalized_load
        + 0.06 * decoder_backlog
        + (0.18 if deadline_miss else 0.0),
    )
    decoder_timeout = False if force_no_timeout else rng.random() < timeout_probability

    runtime_stall_rounds = max(0.0, latency - decoder_deadline) * 2.5 + decoder_backlog * 0.18
    idle_exposure = exposure_scale * idle_window_scale * (
        0.08
        + 0.75 * latency
        + 0.10 * decoder_backlog
        + 0.06 * runtime_stall_rounds
        + idle_error_rate * num_rounds * 4.5
    )
    idle_rng = random.Random(seed + 4_000_003)
    idle_flip_probability = min(0.70, max(0.0, idle_exposure - 0.48) * 0.38)
    runtime_idle_flip = idle_rng.random() < idle_flip_probability

    return {
        "decoder_arrival_time": fmt_float(arrival_time),
        "decoder_start_time": fmt_float(start_time),
        "decoder_finish_time": fmt_float(finish_time),
        "decoder_latency": fmt_float(latency),
        "decoder_deadline_miss": deadline_miss,
        "decoder_queue_overflow": queue_overflow,
        "decoder_timeout": decoder_timeout,
        "decoder_backlog": fmt_float(decoder_backlog),
        "runtime_stall_rounds": fmt_float(runtime_stall_rounds),
        "idle_exposure": fmt_float(idle_exposure),
        "runtime_idle_flip": runtime_idle_flip,
    }


def apply_p3_intervention(record: dict[str, object], intervention: str) -> dict[str, object]:
    if intervention not in P3_INTERVENTIONS:
        raise ValueError(f"unknown P3 intervention: {intervention}")
    out = normalize_p3_record_types(record)
    if intervention in P3_NOISE_INTERVENTIONS:
        out = apply_paired_noise_intervention(out, intervention)
    else:
        kwargs = runtime_intervention_kwargs(intervention)
        out.update(compute_runtime_service_fields(out, **kwargs))
    return recompute_p3_failure_behavior(out)


def apply_paired_noise_intervention(
    record: dict[str, object],
    intervention: str,
) -> dict[str, object]:
    out = deepcopy(record)
    strength = {
        "remove_data_noise": 0.55,
        "weaken_data_noise_50pct": 0.28,
        "remove_measurement_noise": 0.50,
        "weaken_measurement_noise_50pct": 0.25,
        "remove_idle_noise": 0.30,
        "weaken_idle_noise_50pct": 0.15,
    }[intervention]
    detector_events = parse_detector_events(out.get("detector_events", "[]"))
    seed = parse_int(out["seed"])
    filtered = thin_detector_events(detector_events, seed=seed, intervention=intervention, remove_fraction=strength)
    out["detector_events"] = json.dumps(filtered, separators=(",", ":"))
    out["detector_event_count"] = len(filtered)

    if intervention.endswith("data_noise") or intervention.endswith("data_noise_50pct"):
        out["data_error_rate"] = fmt_float(parse_float(out["data_error_rate"]) * (1.0 - strength))
    elif intervention.endswith("measurement_noise") or intervention.endswith("measurement_noise_50pct"):
        out["measurement_error_rate"] = fmt_float(parse_float(out["measurement_error_rate"]) * (1.0 - strength))
    else:
        out["idle_error_rate"] = fmt_float(parse_float(out["idle_error_rate"]) * (1.0 - strength))

    rng = random.Random(seed + stable_intervention_offset(intervention))
    if parse_bool(out["qec_decoder_failure"]) and rng.random() < strength:
        out["decoder_prediction"] = out["observable_flip"]

    out.update(compute_runtime_service_fields(out))
    return out


def runtime_intervention_kwargs(intervention: str) -> dict[str, object]:
    if intervention == "remove_decoder_timeout":
        return {"force_no_timeout": True}
    if intervention == "increase_decoder_workers_2x":
        return {"worker_scale": 2.0}
    if intervention == "increase_decoder_service_rate_2x":
        return {"service_rate_scale": 2.0}
    if intervention == "remove_decoder_queueing":
        return {"remove_queueing": True, "queue_scale": 0.0}
    if intervention == "relax_decoder_deadline_2x":
        return {"deadline_scale": 2.0}
    if intervention == "reduce_idle_exposure_50pct":
        return {"exposure_scale": 0.5}
    if intervention == "prioritize_high_weight_syndromes":
        return {"prioritize_high_weight": True}
    raise ValueError(f"unknown runtime intervention: {intervention}")


def compare_p3_records(
    baseline: dict[str, object],
    intervened: dict[str, object],
    intervention: str,
) -> dict[str, object]:
    baseline_failure = parse_bool(baseline["logical_failure"])
    intervened_failure = parse_bool(intervened["logical_failure"])
    return {
        "run_id": baseline["run_id"],
        "workload_id": baseline["workload_id"],
        "stress_level": baseline["stress_level"],
        "shot_id": baseline["shot_id"],
        "seed": baseline["seed"],
        "intervention": intervention,
        "intervention_class": classify_p3_intervention(intervention),
        "baseline_logical_failure": baseline_failure,
        "intervened_logical_failure": intervened_failure,
        "baseline_failure_round": baseline["failure_round"],
        "intervened_failure_round": intervened["failure_round"],
        "baseline_failure_mode": baseline["failure_mode"],
        "intervened_failure_mode": intervened["failure_mode"],
        "baseline_failure_pattern": baseline["failure_pattern"],
        "intervened_failure_pattern": intervened["failure_pattern"],
        "rescued_failure": baseline_failure and not intervened_failure,
        "new_failure": (not baseline_failure) and intervened_failure,
    }


def classify_p3_intervention(intervention: str) -> str:
    if intervention in P3_NOISE_INTERVENTIONS:
        return "noise"
    if intervention in P3_RUNTIME_INTERVENTIONS:
        return "runtime"
    raise ValueError(f"unknown P3 intervention: {intervention}")


def recompute_p3_failure_behavior(record: dict[str, object]) -> dict[str, object]:
    out = deepcopy(record)
    observable_flip = parse_bool(out["observable_flip"])
    decoder_prediction = parse_bool(out["decoder_prediction"])
    decoder_timeout = parse_bool(out["decoder_timeout"])
    runtime_idle_flip = parse_bool(out["runtime_idle_flip"])
    detector_event_count = parse_int(out["detector_event_count"])

    effective_observable = observable_flip ^ runtime_idle_flip
    applied_prediction = False if decoder_timeout else decoder_prediction
    logical_failure = applied_prediction != effective_observable
    out["logical_failure"] = logical_failure
    out["qec_decoder_failure"] = decoder_prediction != observable_flip

    if not logical_failure:
        out["failure_round"] = ""
        out["failure_region"] = "none"
        out["failure_operation"] = ""
        out["failure_mode"] = "none"
        out["failure_pattern"] = "no_failure"
        return out

    detector_indices = parse_detector_events(out.get("detector_events", "[]"))
    num_rounds = parse_int(out["num_rounds"], 1)
    distance = parse_int(out["distance"], 1)
    detector_per_round = max(1, distance - 1)
    first_detector = detector_indices[0] if detector_indices else 0
    failure_round = min(num_rounds, first_detector // detector_per_round)
    out["failure_round"] = failure_round
    out["failure_region"] = f"detector_{first_detector % detector_per_round}" if detector_indices else "logical"
    out["failure_operation"] = failure_round

    if decoder_timeout:
        out["failure_mode"] = "deadline_timeout"
        out["failure_pattern"] = "deadline_timeout"
    elif parse_bool(out["decoder_deadline_miss"]):
        out["failure_mode"] = "deadline_miss"
        out["failure_pattern"] = "deadline_miss"
    elif parse_float(out["decoder_backlog"]) > max(1.0, 0.3 * distance):
        out["failure_mode"] = "queue_pressure"
        out["failure_pattern"] = "queue_correlated"
    elif runtime_idle_flip:
        out["failure_mode"] = "runtime_idle_flip"
        out["failure_pattern"] = "idle_correlated"
    elif parse_bool(out["qec_decoder_failure"]):
        out["failure_mode"] = "decoder_mismatch"
        out["failure_pattern"] = "syndrome_decoding_mismatch"
    elif detector_event_count >= max(2, distance):
        out["failure_mode"] = "high_syndrome_weight"
        out["failure_pattern"] = "syndrome_burst"
    else:
        out["failure_mode"] = "logical_observable_flip"
        out["failure_pattern"] = "logical_observable"
    return out


def normalize_p3_record_types(row: dict[str, object]) -> dict[str, object]:
    out = deepcopy(row)
    for field in (
        "distance",
        "num_rounds",
        "shot_id",
        "seed",
        "detector_count",
        "detector_event_count",
        "decoder_workers",
        "decoder_queue_depth",
    ):
        if field in out:
            out[field] = parse_int(out[field])
    for field in (
        "data_error_rate",
        "measurement_error_rate",
        "idle_error_rate",
        "decoder_timeout_base_rate",
        "detector_load_scale",
        "idle_window_scale",
        "deadline_scale",
        "decoder_service_rate",
        "decoder_deadline",
        "decoder_arrival_time",
        "decoder_start_time",
        "decoder_finish_time",
        "decoder_latency",
        "decoder_backlog",
        "runtime_stall_rounds",
        "idle_exposure",
    ):
        if field in out:
            out[field] = parse_float(out[field])
    for field in (
        "decoder_deadline_miss",
        "decoder_queue_overflow",
        "observable_flip",
        "decoder_prediction",
        "qec_decoder_failure",
        "decoder_timeout",
        "runtime_idle_flip",
        "logical_failure",
    ):
        if field in out:
            out[field] = parse_bool(out[field])
    return out


def thin_detector_events(
    detector_events: list[int],
    *,
    seed: int,
    intervention: str,
    remove_fraction: float,
) -> list[int]:
    rng = random.Random(seed + stable_intervention_offset(intervention))
    return [event for event in detector_events if rng.random() >= remove_fraction]


def stable_intervention_offset(intervention: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(intervention))


def parse_detector_events(value: object) -> list[int]:
    if value in (None, ""):
        return []
    return [int(item) for item in json.loads(str(value))]
