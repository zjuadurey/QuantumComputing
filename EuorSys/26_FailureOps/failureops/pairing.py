"""Pairing hashes and validators for P4 paired counterfactual rows."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any

from failureops.intervention_registry import (
    InterventionSpec,
    get_intervention_spec,
    serialize_fields,
)
from failureops.io_utils import fmt_float, parse_bool

EVENT_HASH_FIELDS = (
    "workload_id",
    "stress_level",
    "seed",
    "shot_id",
    "detector_events",
    "observable_flip",
    "decoder_prediction",
)

RECORD_HASH_EXCLUDED_FIELDS = {
    "event_record_hash",
    "record_hash",
    "baseline_record_hash",
    "intervened_record_hash",
    "baseline_event_record_hash",
    "intervened_event_record_hash",
    "pairing_valid",
    "pairing_violations",
}


def build_pair_id(record: dict[str, object]) -> str:
    return "|".join(
        str(record.get(field, ""))
        for field in ("run_id", "workload_id", "stress_level", "seed", "shot_id")
    )


def event_record_hash(record: dict[str, object]) -> str:
    return stable_hash({field: canonical_value(record.get(field, "")) for field in EVENT_HASH_FIELDS})


def record_hash(record: dict[str, object]) -> str:
    return stable_hash(
        {
            key: canonical_value(value)
            for key, value in sorted(record.items())
            if key not in RECORD_HASH_EXCLUDED_FIELDS
        }
    )


def stable_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def canonical_value(value: Any) -> object:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return ""
    text = str(value)
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        loaded = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return text
    return loaded


def build_p4_intervention_row(
    baseline: dict[str, object],
    intervened: dict[str, object],
    intervention: str,
) -> dict[str, object]:
    spec = get_intervention_spec(intervention)
    validation = validate_pair(baseline, intervened, spec)
    baseline_failure = parse_bool(baseline["logical_failure"])
    intervened_failure = parse_bool(intervened["logical_failure"])
    return {
        "pair_id": build_pair_id(baseline),
        "run_id": baseline["run_id"],
        "workload_id": baseline["workload_id"],
        "stress_level": baseline["stress_level"],
        "shot_id": baseline["shot_id"],
        "seed": baseline["seed"],
        "intervention": intervention,
        "intervention_class": spec.intervention_class,
        "intervention_allowed_changes": serialize_fields(spec.allowed_changes),
        "intervention_required_invariants": serialize_fields(spec.required_invariants),
        "baseline_record_hash": record_hash(baseline),
        "intervened_record_hash": record_hash(intervened),
        "baseline_event_record_hash": event_record_hash(baseline),
        "intervened_event_record_hash": event_record_hash(intervened),
        "pairing_valid": validation["valid"],
        "pairing_violations": "|".join(validation["violations"]),
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


def validate_pair(
    baseline: dict[str, object],
    intervened: dict[str, object],
    spec: InterventionSpec,
) -> dict[str, object]:
    violations = []
    for field in spec.required_invariants:
        if canonical_value(baseline.get(field, "")) != canonical_value(intervened.get(field, "")):
            violations.append(f"{field}_changed")
    if build_pair_id(baseline) != build_pair_id(intervened):
        violations.append("pair_id_changed")
    if spec.preserve_event_record and event_record_hash(baseline) != event_record_hash(intervened):
        violations.append("event_record_changed_when_forbidden")

    allowed = set(spec.allowed_changes) | set(spec.required_invariants)
    all_fields = set(baseline) | set(intervened)
    for field in sorted(all_fields):
        if field in allowed or field in RECORD_HASH_EXCLUDED_FIELDS:
            continue
        if canonical_value(baseline.get(field, "")) != canonical_value(intervened.get(field, "")):
            violations.append(f"unexpected_change:{field}")
    return {"valid": not violations, "violations": violations}


def validate_intervention_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["intervention"]), str(row.get("intervention_class", "unknown")))].append(row)

    out = []
    for (intervention, intervention_class), group in groups.items():
        invalid = [row for row in group if not parse_bool(row.get("pairing_valid", False))]
        violations = Counter()
        for row in invalid:
            for violation in str(row.get("pairing_violations", "")).split("|"):
                if violation:
                    violations[violation] += 1
        out.append(
            {
                "intervention": intervention,
                "intervention_class": intervention_class,
                "num_pairs": len(group),
                "valid_pairs": len(group) - len(invalid),
                "invalid_pairs": len(invalid),
                "violation_rate": fmt_float(len(invalid) / len(group) if group else 0.0),
                "top_violations": json.dumps(dict(violations.most_common(5)), sort_keys=True, separators=(",", ":")),
            }
        )
    out.sort(key=lambda row: (int(row["invalid_pairs"]), row["intervention"]))
    return out
