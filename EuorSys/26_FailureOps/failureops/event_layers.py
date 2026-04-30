"""Layered event records for P5b noise-side paired replay."""

from __future__ import annotations

import json
import random
from copy import deepcopy

from failureops.io_utils import fmt_float, parse_bool, parse_float, parse_int
from failureops.pairing import stable_hash
from failureops.runtime_service import (
    compute_runtime_service_fields,
    parse_detector_events,
    recompute_p3_failure_behavior,
    stable_intervention_offset,
)

LAYER_NAMES = ("data_events", "measurement_events", "idle_events", "runtime_events")

NOISE_LAYER_MAP = {
    "remove_data_noise": ("data_events", 0.55),
    "weaken_data_noise_50pct": ("data_events", 0.28),
    "remove_measurement_noise": ("measurement_events", 0.50),
    "weaken_measurement_noise_50pct": ("measurement_events", 0.25),
    "remove_idle_noise": ("idle_events", 0.30),
    "weaken_idle_noise_50pct": ("idle_events", 0.15),
}


def attach_event_layers(record: dict[str, object]) -> dict[str, object]:
    out = dict(record)
    if not out.get("event_layers"):
        out["event_layers"] = serialize_event_layers(build_event_layers(out))
    out["event_layer_hash"] = event_layer_hash(out)
    return out


def build_event_layers(record: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    distance = parse_int(record.get("distance", 1), 1)
    num_rounds = parse_int(record.get("num_rounds", 1), 1)
    detector_per_round = max(1, distance - 1)
    layers: dict[str, list[dict[str, object]]] = {name: [] for name in LAYER_NAMES}
    layer_cycle = ("data_events", "measurement_events", "idle_events")
    for position, detector_id in enumerate(parse_detector_events(record.get("detector_events", "[]"))):
        layer_name = layer_cycle[detector_id % len(layer_cycle)]
        layers[layer_name].append(
            {
                "event_id": f"{layer_name}:{position}:{detector_id}",
                "detector_id": detector_id,
                "round_id": min(num_rounds, detector_id // detector_per_round),
                "region_id": f"detector_{detector_id % detector_per_round}",
                "weight": fmt_float(1.0),
            }
        )
    if parse_bool(record.get("decoder_timeout", False)):
        layers["runtime_events"].append(runtime_event(record, "decoder_timeout"))
    if parse_bool(record.get("decoder_deadline_miss", False)):
        layers["runtime_events"].append(runtime_event(record, "deadline_miss"))
    if parse_float(record.get("decoder_backlog", 0.0)) > 0.0:
        layers["runtime_events"].append(runtime_event(record, "decoder_backlog"))
    if parse_bool(record.get("runtime_idle_flip", False)):
        layers["runtime_events"].append(runtime_event(record, "runtime_idle_flip"))
    return layers


def runtime_event(record: dict[str, object], event_type: str) -> dict[str, object]:
    return {
        "event_id": f"runtime:{event_type}:{record.get('seed')}:{record.get('shot_id')}",
        "event_type": event_type,
        "latency": fmt_float(parse_float(record.get("decoder_latency", 0.0))),
        "backlog": fmt_float(parse_float(record.get("decoder_backlog", 0.0))),
        "idle_exposure": fmt_float(parse_float(record.get("idle_exposure", 0.0))),
    }


def apply_layered_noise_intervention(record: dict[str, object], intervention: str) -> dict[str, object]:
    if intervention not in NOISE_LAYER_MAP:
        raise ValueError(f"not a layered noise intervention: {intervention}")
    out = attach_event_layers(deepcopy(record))
    layer_name, remove_fraction = NOISE_LAYER_MAP[intervention]
    layers = parse_event_layers(out.get("event_layers", "{}"))
    seed = parse_int(out["seed"])
    rng = random.Random(seed + stable_intervention_offset(intervention))
    layers[layer_name] = [
        event for event in layers.get(layer_name, [])
        if rng.random() >= remove_fraction
    ]
    out["event_layers"] = serialize_event_layers(layers)
    out["event_layer_hash"] = event_layer_hash(out)
    detector_events = detector_events_from_layers(layers)
    out["detector_events"] = json.dumps(detector_events, separators=(",", ":"))
    out["detector_event_count"] = len(detector_events)
    scale_noise_rate(out, intervention, remove_fraction)
    if parse_bool(out.get("qec_decoder_failure", False)) and rng.random() < remove_fraction:
        out["decoder_prediction"] = out["observable_flip"]
    out.update(compute_runtime_service_fields(out))
    return attach_event_layers(recompute_p3_failure_behavior(out))


def detector_events_from_layers(layers: dict[str, list[dict[str, object]]]) -> list[int]:
    detector_ids = []
    for layer_name in ("data_events", "measurement_events", "idle_events"):
        detector_ids.extend(parse_int(event["detector_id"]) for event in layers.get(layer_name, []))
    return sorted(dict.fromkeys(detector_ids))


def scale_noise_rate(record: dict[str, object], intervention: str, remove_fraction: float) -> None:
    if "data_noise" in intervention:
        field = "data_error_rate"
    elif "measurement_noise" in intervention:
        field = "measurement_error_rate"
    else:
        field = "idle_error_rate"
    record[field] = fmt_float(parse_float(record[field]) * (1.0 - remove_fraction))


def parse_event_layers(value: object) -> dict[str, list[dict[str, object]]]:
    if value in (None, ""):
        return {name: [] for name in LAYER_NAMES}
    layers = json.loads(str(value))
    return {name: list(layers.get(name, [])) for name in LAYER_NAMES}


def serialize_event_layers(layers: dict[str, list[dict[str, object]]]) -> str:
    normalized = {name: layers.get(name, []) for name in LAYER_NAMES}
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def event_layer_hash(record: dict[str, object]) -> str:
    return stable_hash(parse_event_layers(record.get("event_layers", "{}")))

