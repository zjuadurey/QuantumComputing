"""Paired counterfactual interventions for P0."""

from __future__ import annotations

from copy import deepcopy
from typing import Callable

from failureops.io_utils import parse_bool
from failureops.toy_simulator import decode_events, encode_events, event_weight, recompute_failure_behavior

INTERVENTIONS = [
    "remove_data_errors",
    "remove_measurement_errors",
    "remove_idle_errors",
    "weaken_data_errors_50pct",
    "weaken_measurement_errors_50pct",
    "weaken_idle_errors_50pct",
    "remove_decoder_timeout",
    "reduce_decoder_delay_50pct",
    "reduce_idle_exposure_50pct",
]


def apply_intervention(record: dict[str, object], intervention: str) -> dict[str, object]:
    if intervention not in _HANDLERS:
        raise ValueError(f"unknown intervention: {intervention}")

    out = deepcopy(record)
    events = decode_events(out.get("error_events", "[]"))
    events = _HANDLERS[intervention](events)
    out["error_events"] = encode_events(events)
    return recompute_failure_behavior(out)


def compare_intervention(
    baseline: dict[str, object],
    intervention: str,
) -> dict[str, object]:
    intervened = apply_intervention(baseline, intervention)
    baseline_failure = parse_bool(baseline["logical_failure"])
    intervened_failure = parse_bool(intervened["logical_failure"])
    return {
        "shot_id": baseline["shot_id"],
        "seed": baseline["seed"],
        "intervention": intervention,
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


def _remove_event_type(event_type: str) -> Callable[[list[dict[str, object]]], list[dict[str, object]]]:
    return lambda events: [event for event in events if event.get("event_type") != event_type]


def _weaken_event_type(
    event_type: str,
    factor: float,
) -> Callable[[list[dict[str, object]]], list[dict[str, object]]]:
    def handler(events: list[dict[str, object]]) -> list[dict[str, object]]:
        out = deepcopy(events)
        for event in out:
            if event.get("event_type") == event_type:
                event["weight"] = round(event_weight(event) * factor, 6)
        return out

    return handler


_HANDLERS = {
    "remove_data_errors": _remove_event_type("data_error"),
    "remove_measurement_errors": _remove_event_type("measurement_error"),
    "remove_idle_errors": _remove_event_type("idle_error"),
    "weaken_data_errors_50pct": _weaken_event_type("data_error", 0.5),
    "weaken_measurement_errors_50pct": _weaken_event_type("measurement_error", 0.5),
    "weaken_idle_errors_50pct": _weaken_event_type("idle_error", 0.5),
    "remove_decoder_timeout": _remove_event_type("decoder_timeout"),
    "reduce_decoder_delay_50pct": _weaken_event_type("decoder_delay", 0.5),
    "reduce_idle_exposure_50pct": _weaken_event_type("idle_exposure", 0.5),
}

