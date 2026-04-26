"""Lightweight Stim/PyMatching backend for P1.

P1 uses a generated repetition-code memory circuit from Stim and decodes detector
events with PyMatching. Runtime fields are intentionally simple post-processing
signals so FailureOps can compare QEC noise factors with system factors without
building a full runtime simulator.
"""

from __future__ import annotations

import json
import os
import random
from copy import deepcopy
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/failureops-matplotlib")

import pymatching
import stim

from failureops.io_utils import fmt_float, parse_bool, parse_float, parse_int

RUNTIME_IDLE_THRESHOLD = 0.55


def generate_qec_runs(
    *,
    num_shots: int,
    distance: int,
    num_rounds: int,
    data_error_rate: float,
    measurement_error_rate: float,
    idle_error_rate: float,
    decoder_timeout_base_rate: float,
    seed: int,
    decoder_capacity: float = 4.0,
    synchronization_slack: float = 0.45,
    run_id: str = "p1",
    circuit_id: str = "stim_repetition_memory",
) -> list[dict[str, object]]:
    circuit = build_repetition_memory_circuit(
        distance=distance,
        num_rounds=num_rounds,
        data_error_rate=data_error_rate,
        measurement_error_rate=measurement_error_rate,
        idle_error_rate=idle_error_rate,
    )
    detector_samples, observable_samples, predictions = sample_and_decode(
        circuit=circuit,
        num_shots=num_shots,
        seed=seed,
    )

    rows = []
    for shot_id in range(num_shots):
        shot_seed = seed + shot_id
        detector_bits = [bool(value) for value in detector_samples[shot_id]]
        observable_flip = bool(observable_samples[shot_id][0])
        decoder_prediction = bool(predictions[shot_id][0])
        detector_indices = [index for index, value in enumerate(detector_bits) if value]
        runtime = compute_runtime_fields(
            detector_event_count=len(detector_indices),
            distance=distance,
            num_rounds=num_rounds,
            idle_error_rate=idle_error_rate,
            decoder_timeout_base_rate=decoder_timeout_base_rate,
            decoder_capacity=decoder_capacity,
            synchronization_slack=synchronization_slack,
            seed=shot_seed,
        )
        row: dict[str, object] = {
            "run_id": run_id,
            "circuit_id": circuit_id,
            "backend": "stim+pymatching",
            "code_family": "repetition_code_memory",
            "distance": distance,
            "num_rounds": num_rounds,
            "shot_id": shot_id,
            "seed": shot_seed,
            "data_error_rate": data_error_rate,
            "measurement_error_rate": measurement_error_rate,
            "idle_error_rate": idle_error_rate,
            "decoder_timeout_base_rate": decoder_timeout_base_rate,
            "decoder_capacity": fmt_float(decoder_capacity),
            "detector_count": circuit.num_detectors,
            "detector_event_count": len(detector_indices),
            "detector_events": json.dumps(detector_indices, separators=(",", ":")),
            "observable_flip": observable_flip,
            "decoder_prediction": decoder_prediction,
            "qec_decoder_failure": decoder_prediction != observable_flip,
            **runtime,
        }
        rows.append(recompute_qec_failure_behavior(row))
    return rows


def build_repetition_memory_circuit(
    *,
    distance: int,
    num_rounds: int,
    data_error_rate: float,
    measurement_error_rate: float,
    idle_error_rate: float,
) -> stim.Circuit:
    return stim.Circuit.generated(
        "repetition_code:memory",
        distance=distance,
        rounds=num_rounds,
        before_round_data_depolarization=data_error_rate,
        before_measure_flip_probability=measurement_error_rate,
        after_clifford_depolarization=idle_error_rate,
    )


def sample_and_decode(
    *,
    circuit: stim.Circuit,
    num_shots: int,
    seed: int,
) -> tuple[Any, Any, Any]:
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    sampler = circuit.compile_detector_sampler(seed=seed)
    detector_samples, observable_samples = sampler.sample(num_shots, separate_observables=True)
    predictions = matching.decode_batch(detector_samples)
    return detector_samples, observable_samples, predictions


def compute_runtime_fields(
    *,
    detector_event_count: int,
    distance: int,
    num_rounds: int,
    idle_error_rate: float,
    decoder_timeout_base_rate: float,
    decoder_capacity: float = 4.0,
    synchronization_slack: float = 0.45,
    seed: int,
    delay_scale: float = 1.0,
    exposure_scale: float = 1.0,
    capacity_scale: float = 1.0,
    backlog_scale: float = 1.0,
    slack_bonus: float = 0.0,
    timeout_scale: float = 1.0,
    force_no_timeout: bool = False,
) -> dict[str, object]:
    rng = random.Random(seed + 1_000_003)
    normalized_syndrome_weight = detector_event_count / max(1, distance * num_rounds)
    effective_capacity = max(0.25, decoder_capacity * capacity_scale)
    decoder_backlog = max(0.0, detector_event_count - effective_capacity) * backlog_scale
    effective_slack = min(1.0, synchronization_slack + slack_bonus)
    delay = delay_scale * (
        0.08
        + 0.035 * detector_event_count
        + 0.11 * decoder_backlog
        + rng.uniform(0.0, 0.035)
    )
    timeout_probability = min(
        0.95,
        decoder_timeout_base_rate
        + timeout_scale * (0.9 * normalized_syndrome_weight + 0.10 * decoder_backlog),
    )
    decoder_timeout = False if force_no_timeout else rng.random() < timeout_probability
    if decoder_timeout:
        delay += delay_scale * (0.30 + 0.035 * detector_event_count + 0.08 * decoder_backlog)

    idle_exposure = exposure_scale * (
        0.10
        + delay * 0.85
        + decoder_backlog * 0.06
        + max(0.0, 0.55 - effective_slack) * 0.45
        + idle_error_rate * num_rounds * 4.0
    )
    idle_rng = random.Random(seed + 2_000_003)
    idle_flip_probability = min(0.65, max(0.0, idle_exposure - RUNTIME_IDLE_THRESHOLD) * 0.35)
    runtime_idle_flip = idle_rng.random() < idle_flip_probability
    return {
        "decoder_timeout": decoder_timeout,
        "decoder_delay": fmt_float(delay),
        "decoder_backlog": fmt_float(decoder_backlog),
        "synchronization_slack": fmt_float(effective_slack),
        "idle_exposure": fmt_float(idle_exposure),
        "runtime_idle_flip": runtime_idle_flip,
    }


def recompute_qec_failure_behavior(record: dict[str, object]) -> dict[str, object]:
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

    detector_indices = _parse_detector_events(out.get("detector_events", "[]"))
    num_rounds = parse_int(out["num_rounds"], 1)
    distance = parse_int(out["distance"], 1)
    detector_per_round = max(1, distance - 1)
    first_detector = detector_indices[0] if detector_indices else 0
    failure_round = min(num_rounds, first_detector // detector_per_round)
    failure_region = f"detector_{first_detector % detector_per_round}" if detector_indices else "logical"

    out["failure_round"] = failure_round
    out["failure_region"] = failure_region
    out["failure_operation"] = failure_round

    if decoder_timeout:
        out["failure_mode"] = "timeout_sensitive"
        out["failure_pattern"] = "timeout_correlated"
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


def normalize_qec_record_types(row: dict[str, object]) -> dict[str, object]:
    out = deepcopy(row)
    for field in (
        "distance",
        "num_rounds",
        "shot_id",
        "seed",
        "detector_count",
        "detector_event_count",
    ):
        if field in out:
            out[field] = parse_int(out[field])
    for field in (
        "data_error_rate",
        "measurement_error_rate",
        "idle_error_rate",
        "decoder_timeout_base_rate",
        "decoder_capacity",
        "decoder_backlog",
        "synchronization_slack",
        "decoder_delay",
        "idle_exposure",
    ):
        if field in out:
            out[field] = parse_float(out[field])
    for field in (
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


def apply_runtime_intervention(record: dict[str, object], intervention: str) -> dict[str, object]:
    out = normalize_qec_record_types(record)
    detector_event_count = parse_int(out["detector_event_count"])
    distance = parse_int(out["distance"])
    num_rounds = parse_int(out["num_rounds"])
    idle_error_rate = parse_float(out["idle_error_rate"])
    decoder_timeout_base_rate = parse_float(out["decoder_timeout_base_rate"])
    decoder_capacity = parse_float(out.get("decoder_capacity", 4.0))
    synchronization_slack = parse_float(out.get("synchronization_slack", 0.45))
    seed = parse_int(out["seed"])

    delay_scale = 1.0
    exposure_scale = 1.0
    capacity_scale = 1.0
    backlog_scale = 1.0
    slack_bonus = 0.0
    timeout_scale = 1.0
    force_no_timeout = False
    if intervention == "remove_decoder_timeout":
        force_no_timeout = True
    elif intervention == "reduce_decoder_delay_50pct":
        delay_scale = 0.5
    elif intervention == "reduce_idle_exposure_50pct":
        exposure_scale = 0.5
    elif intervention == "increase_decoder_capacity_2x":
        capacity_scale = 2.0
    elif intervention == "eliminate_decoder_backlog":
        backlog_scale = 0.0
    elif intervention == "increase_synchronization_slack":
        slack_bonus = 0.35
    elif intervention == "relax_timeout_policy":
        timeout_scale = 0.35
    else:
        raise ValueError(f"unknown runtime intervention: {intervention}")

    runtime = compute_runtime_fields(
        detector_event_count=detector_event_count,
        distance=distance,
        num_rounds=num_rounds,
        idle_error_rate=idle_error_rate,
        decoder_timeout_base_rate=decoder_timeout_base_rate,
        decoder_capacity=decoder_capacity,
        synchronization_slack=synchronization_slack,
        seed=seed,
        delay_scale=delay_scale,
        exposure_scale=exposure_scale,
        capacity_scale=capacity_scale,
        backlog_scale=backlog_scale,
        slack_bonus=slack_bonus,
        timeout_scale=timeout_scale,
        force_no_timeout=force_no_timeout,
    )
    out.update(runtime)
    return recompute_qec_failure_behavior(out)


def _parse_detector_events(value: object) -> list[int]:
    if value in (None, ""):
        return []
    return [int(item) for item in json.loads(str(value))]
