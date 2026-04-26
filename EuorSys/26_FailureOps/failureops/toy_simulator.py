"""Toy logical execution model for P0.

This is intentionally not a physical QEC simulator. It is a transparent,
deterministic proxy that makes the FailureOps loop executable:
baseline record -> intervention -> recomputed failure behavior -> sensitivity.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any

from failureops.io_utils import fmt_float, parse_bool, parse_float, parse_int

REGIONS = ("R0", "R1", "R2")
FAILURE_THRESHOLD = 4.4

EVENT_WEIGHTS = {
    "data_error": 1.0,
    "measurement_error": 0.75,
    "idle_error": 0.65,
    "decoder_timeout": 1.7,
    "decoder_delay": 0.55,
    "idle_exposure": 0.35,
}

PATTERN_BY_COMPONENT = {
    "data": "data_dominated",
    "measurement": "measurement_dominated",
    "idle": "idle_correlated",
    "timeout": "timeout_correlated",
    "delay": "delay_correlated",
}

MODE_BY_COMPONENT = {
    "data": "logical_X",
    "measurement": "logical_Z",
    "idle": "logical_Z",
    "timeout": "timeout_sensitive",
    "delay": "delay_sensitive",
}


def encode_events(events: list[dict[str, Any]]) -> str:
    return json.dumps(events, sort_keys=True, separators=(",", ":"))


def decode_events(value: object) -> list[dict[str, Any]]:
    if value in (None, ""):
        return []
    return json.loads(str(value))


def generate_runs(
    *,
    num_shots: int,
    num_rounds: int,
    num_operations: int,
    data_error_rate: float,
    measurement_error_rate: float,
    idle_error_rate: float,
    decoder_timeout_rate: float,
    seed: int,
    run_id: str = "p0",
    circuit_id: str = "toy_logical_workload",
) -> list[dict[str, object]]:
    return [
        generate_one_run(
            run_id=run_id,
            circuit_id=circuit_id,
            shot_id=shot_id,
            seed=seed + shot_id,
            num_rounds=num_rounds,
            num_operations=num_operations,
            data_error_rate=data_error_rate,
            measurement_error_rate=measurement_error_rate,
            idle_error_rate=idle_error_rate,
            decoder_timeout_rate=decoder_timeout_rate,
        )
        for shot_id in range(num_shots)
    ]


def generate_one_run(
    *,
    run_id: str,
    circuit_id: str,
    shot_id: int,
    seed: int,
    num_rounds: int,
    num_operations: int,
    data_error_rate: float,
    measurement_error_rate: float,
    idle_error_rate: float,
    decoder_timeout_rate: float,
) -> dict[str, object]:
    rng = random.Random(seed)
    events: list[dict[str, Any]] = []

    for round_id in range(num_rounds):
        for operation_id in range(num_operations):
            if rng.random() < data_error_rate:
                events.append(_make_event(rng, "data_error", round_id, operation_id))
            if rng.random() < measurement_error_rate:
                events.append(_make_event(rng, "measurement_error", round_id, operation_id))
            if rng.random() < idle_error_rate:
                events.append(_make_event(rng, "idle_error", round_id, operation_id))

    decoder_timeout = rng.random() < decoder_timeout_rate
    decoder_delay = rng.gammavariate(1.8, 0.28)
    decoder_delay += 0.03 * len(events)
    if decoder_timeout:
        decoder_delay += rng.uniform(1.1, 2.0)

    idle_error_count = sum(1 for event in events if event["event_type"] == "idle_error")
    idle_exposure = 0.35 + 0.55 * idle_error_count + 0.65 * decoder_delay
    idle_exposure += rng.uniform(0.0, 0.45)

    events.append(
        {
            "event_type": "decoder_delay",
            "round_id": rng.randrange(num_rounds),
            "operation_id": rng.randrange(num_operations),
            "region_id": "runtime",
            "weight": round(decoder_delay, 6),
        }
    )
    events.append(
        {
            "event_type": "idle_exposure",
            "round_id": rng.randrange(num_rounds),
            "operation_id": rng.randrange(num_operations),
            "region_id": "runtime",
            "weight": round(idle_exposure, 6),
        }
    )
    if decoder_timeout:
        events.append(
            {
                "event_type": "decoder_timeout",
                "round_id": rng.randrange(num_rounds),
                "operation_id": rng.randrange(num_operations),
                "region_id": "runtime",
                "weight": 1.0,
            }
        )

    record: dict[str, object] = {
        "run_id": run_id,
        "circuit_id": circuit_id,
        "shot_id": shot_id,
        "seed": seed,
        "num_rounds": num_rounds,
        "num_operations": num_operations,
        "data_error_rate": data_error_rate,
        "measurement_error_rate": measurement_error_rate,
        "idle_error_rate": idle_error_rate,
        "decoder_timeout_rate": decoder_timeout_rate,
        "decoder_timeout": decoder_timeout,
        "decoder_delay": fmt_float(decoder_delay),
        "idle_exposure": fmt_float(idle_exposure),
        "error_events": encode_events(events),
    }
    return recompute_failure_behavior(record)


def recompute_failure_behavior(record: dict[str, object]) -> dict[str, object]:
    """Recompute all derived failure fields after generation or intervention."""
    out = deepcopy(record)
    events = decode_events(out.get("error_events", "[]"))

    out["data_error_count"] = sum(1 for event in events if event["event_type"] == "data_error")
    out["measurement_error_count"] = sum(
        1 for event in events if event["event_type"] == "measurement_error"
    )
    out["idle_error_count"] = sum(1 for event in events if event["event_type"] == "idle_error")
    out["decoder_timeout"] = any(event["event_type"] == "decoder_timeout" for event in events)

    delay_weight = sum(event_weight(event) for event in events if event["event_type"] == "decoder_delay")
    exposure_weight = sum(
        event_weight(event) for event in events if event["event_type"] == "idle_exposure"
    )
    out["decoder_delay"] = fmt_float(delay_weight)
    out["idle_exposure"] = fmt_float(exposure_weight)
    out["error_events"] = encode_events(events)

    score, components, event_scores = failure_score(events)
    logical_failure = score >= FAILURE_THRESHOLD
    out["logical_failure"] = logical_failure

    if not logical_failure:
        out["failure_round"] = ""
        out["failure_region"] = "none"
        out["failure_operation"] = ""
        out["failure_mode"] = "none"
        out["failure_pattern"] = "no_failure"
        return out

    top_event, top_component = max(event_scores, key=lambda item: item[0])[1:]
    out["failure_round"] = top_event.get("round_id", "")
    out["failure_region"] = top_event.get("region_id", "unknown")
    out["failure_operation"] = top_event.get("operation_id", "")
    out["failure_mode"] = _failure_mode(components)
    out["failure_pattern"] = _failure_pattern(events, components, top_component)
    return out


def failure_score(
    events: list[dict[str, Any]],
) -> tuple[float, dict[str, float], list[tuple[float, dict[str, Any], str]]]:
    components: dict[str, float] = defaultdict(float)
    event_scores: list[tuple[float, dict[str, Any], str]] = []
    for event in events:
        event_type = event.get("event_type", "")
        score = EVENT_WEIGHTS.get(event_type, 0.0) * event_weight(event)
        component = component_name(event_type)
        components[component] += score
        event_scores.append((score, event, component))
    return sum(components.values()), dict(components), event_scores


def event_weight(event: dict[str, Any]) -> float:
    return parse_float(event.get("weight", 1.0), 1.0)


def component_name(event_type: str) -> str:
    if event_type == "data_error":
        return "data"
    if event_type == "measurement_error":
        return "measurement"
    if event_type in {"idle_error", "idle_exposure"}:
        return "idle"
    if event_type == "decoder_timeout":
        return "timeout"
    if event_type == "decoder_delay":
        return "delay"
    return "other"


def normalize_record_types(row: dict[str, object]) -> dict[str, object]:
    out = deepcopy(row)
    for field in (
        "shot_id",
        "seed",
        "num_rounds",
        "num_operations",
        "data_error_count",
        "measurement_error_count",
        "idle_error_count",
    ):
        if field in out:
            out[field] = parse_int(out[field])
    for field in (
        "data_error_rate",
        "measurement_error_rate",
        "idle_error_rate",
        "decoder_timeout_rate",
        "decoder_delay",
        "idle_exposure",
    ):
        if field in out:
            out[field] = parse_float(out[field])
    if "decoder_timeout" in out:
        out["decoder_timeout"] = parse_bool(out["decoder_timeout"])
    if "logical_failure" in out:
        out["logical_failure"] = parse_bool(out["logical_failure"])
    return out


def _make_event(
    rng: random.Random,
    event_type: str,
    round_id: int,
    operation_id: int,
) -> dict[str, Any]:
    return {
        "event_type": event_type,
        "round_id": round_id,
        "operation_id": operation_id,
        "region_id": rng.choice(REGIONS),
        "weight": round(rng.uniform(0.75, 1.25), 6),
    }


def _failure_mode(components: dict[str, float]) -> str:
    total = sum(components.values())
    dominant_component, dominant_score = max(components.items(), key=lambda item: item[1])
    if total > 0 and dominant_score / total < 0.4:
        return "mixed"
    return MODE_BY_COMPONENT.get(dominant_component, "mixed")


def _failure_pattern(
    events: list[dict[str, Any]],
    components: dict[str, float],
    top_component: str,
) -> str:
    region_counts = Counter(
        event.get("region_id", "unknown")
        for event in events
        if event.get("region_id") not in {"runtime", None}
    )
    operation_counts = Counter(
        event.get("operation_id", "unknown")
        for event in events
        if event.get("operation_id") not in {"", None}
    )
    if region_counts and region_counts.most_common(1)[0][1] >= 4:
        return "region_hotspot"
    if operation_counts and operation_counts.most_common(1)[0][1] >= 4:
        return "operation_hotspot"

    total = sum(components.values())
    dominant_component, dominant_score = max(components.items(), key=lambda item: item[1])
    if total > 0 and dominant_score / total >= 0.38:
        return PATTERN_BY_COMPONENT.get(dominant_component, "mixed")
    return PATTERN_BY_COMPONENT.get(top_component, "mixed")
