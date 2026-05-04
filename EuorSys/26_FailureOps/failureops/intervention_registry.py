"""Intervention registry and pairing contracts for P4."""

from __future__ import annotations

from dataclasses import dataclass

from failureops.runtime_service import P3_INTERVENTIONS, P3_NOISE_INTERVENTIONS, P3_RUNTIME_INTERVENTIONS

REGISTRY_VERSION = "p4_registry_v1"

IDENTITY_INVARIANTS = (
    "run_id",
    "workload_id",
    "stress_level",
    "shot_id",
    "seed",
)

EVENT_INVARIANTS = (
    "detector_events",
    "event_layers",
    "observable_flip",
    "decoder_prediction",
)

BASE_ALLOWED_CHANGES = (
    "logical_failure",
    "qec_decoder_failure",
    "failure_round",
    "failure_region",
    "failure_operation",
    "failure_mode",
    "failure_pattern",
)

NOISE_ALLOWED_CHANGES = (
    "data_error_rate",
    "measurement_error_rate",
    "idle_error_rate",
    "detector_event_count",
    "detector_events",
    "event_layers",
    "event_layer_hash",
    "decoder_prediction",
    "qec_decoder_failure",
    "decoder_arrival_time",
    "decoder_start_time",
    "decoder_finish_time",
    "decoder_latency",
    "decoder_deadline_miss",
    "decoder_queue_overflow",
    "decoder_timeout",
    "decoder_backlog",
    "runtime_stall_rounds",
    "idle_exposure",
    "runtime_idle_flip",
    *BASE_ALLOWED_CHANGES,
)

RUNTIME_ALLOWED_CHANGES = (
    "decoder_arrival_time",
    "decoder_start_time",
    "decoder_finish_time",
    "decoder_latency",
    "decoder_deadline_miss",
    "decoder_queue_overflow",
    "decoder_timeout",
    "decoder_backlog",
    "runtime_stall_rounds",
    "idle_exposure",
    "runtime_idle_flip",
    *BASE_ALLOWED_CHANGES,
)

POLICY_ALLOWED_CHANGES = (
    "decoder_deadline_miss",
    "decoder_timeout",
    "runtime_stall_rounds",
    "idle_exposure",
    "runtime_idle_flip",
    *BASE_ALLOWED_CHANGES,
)

DECODER_ALLOWED_CHANGES = (
    "decoder_pathway",
    "decoder_prediction",
    "qec_decoder_failure",
    *BASE_ALLOWED_CHANGES,
)


@dataclass(frozen=True)
class InterventionSpec:
    name: str
    intervention_class: str
    allowed_changes: tuple[str, ...]
    required_invariants: tuple[str, ...]
    preserve_event_record: bool


INTERVENTION_REGISTRY: dict[str, InterventionSpec] = {
    "remove_data_noise": InterventionSpec(
        name="remove_data_noise",
        intervention_class="noise",
        allowed_changes=NOISE_ALLOWED_CHANGES,
        required_invariants=IDENTITY_INVARIANTS,
        preserve_event_record=False,
    ),
    "weaken_data_noise_50pct": InterventionSpec(
        name="weaken_data_noise_50pct",
        intervention_class="noise",
        allowed_changes=NOISE_ALLOWED_CHANGES,
        required_invariants=IDENTITY_INVARIANTS,
        preserve_event_record=False,
    ),
    "remove_measurement_noise": InterventionSpec(
        name="remove_measurement_noise",
        intervention_class="noise",
        allowed_changes=NOISE_ALLOWED_CHANGES,
        required_invariants=IDENTITY_INVARIANTS,
        preserve_event_record=False,
    ),
    "weaken_measurement_noise_50pct": InterventionSpec(
        name="weaken_measurement_noise_50pct",
        intervention_class="noise",
        allowed_changes=NOISE_ALLOWED_CHANGES,
        required_invariants=IDENTITY_INVARIANTS,
        preserve_event_record=False,
    ),
    "remove_idle_noise": InterventionSpec(
        name="remove_idle_noise",
        intervention_class="noise",
        allowed_changes=NOISE_ALLOWED_CHANGES,
        required_invariants=IDENTITY_INVARIANTS,
        preserve_event_record=False,
    ),
    "weaken_idle_noise_50pct": InterventionSpec(
        name="weaken_idle_noise_50pct",
        intervention_class="noise",
        allowed_changes=NOISE_ALLOWED_CHANGES,
        required_invariants=IDENTITY_INVARIANTS,
        preserve_event_record=False,
    ),
    "remove_decoder_timeout": InterventionSpec(
        name="remove_decoder_timeout",
        intervention_class="policy",
        allowed_changes=POLICY_ALLOWED_CHANGES,
        required_invariants=(*IDENTITY_INVARIANTS, *EVENT_INVARIANTS),
        preserve_event_record=True,
    ),
    "increase_decoder_workers_2x": InterventionSpec(
        name="increase_decoder_workers_2x",
        intervention_class="runtime",
        allowed_changes=RUNTIME_ALLOWED_CHANGES,
        required_invariants=(*IDENTITY_INVARIANTS, *EVENT_INVARIANTS),
        preserve_event_record=True,
    ),
    "increase_decoder_service_rate_2x": InterventionSpec(
        name="increase_decoder_service_rate_2x",
        intervention_class="runtime",
        allowed_changes=RUNTIME_ALLOWED_CHANGES,
        required_invariants=(*IDENTITY_INVARIANTS, *EVENT_INVARIANTS),
        preserve_event_record=True,
    ),
    "remove_decoder_queueing": InterventionSpec(
        name="remove_decoder_queueing",
        intervention_class="runtime",
        allowed_changes=RUNTIME_ALLOWED_CHANGES,
        required_invariants=(*IDENTITY_INVARIANTS, *EVENT_INVARIANTS),
        preserve_event_record=True,
    ),
    "relax_decoder_deadline_2x": InterventionSpec(
        name="relax_decoder_deadline_2x",
        intervention_class="policy",
        allowed_changes=POLICY_ALLOWED_CHANGES,
        required_invariants=(*IDENTITY_INVARIANTS, *EVENT_INVARIANTS),
        preserve_event_record=True,
    ),
    "reduce_idle_exposure_50pct": InterventionSpec(
        name="reduce_idle_exposure_50pct",
        intervention_class="runtime",
        allowed_changes=RUNTIME_ALLOWED_CHANGES,
        required_invariants=(*IDENTITY_INVARIANTS, *EVENT_INVARIANTS),
        preserve_event_record=True,
    ),
    "prioritize_high_weight_syndromes": InterventionSpec(
        name="prioritize_high_weight_syndromes",
        intervention_class="policy",
        allowed_changes=RUNTIME_ALLOWED_CHANGES,
        required_invariants=(*IDENTITY_INVARIANTS, *EVENT_INVARIANTS),
        preserve_event_record=True,
    ),
    "switch_decoder_pathway": InterventionSpec(
        name="switch_decoder_pathway",
        intervention_class="decoder",
        allowed_changes=DECODER_ALLOWED_CHANGES,
        required_invariants=(
            *IDENTITY_INVARIANTS,
            "detector_events",
            "event_layers",
            "observable_flip",
        ),
        preserve_event_record=False,
    ),
    "switch_decoder_prior": InterventionSpec(
        name="switch_decoder_prior",
        intervention_class="decoder",
        allowed_changes=DECODER_ALLOWED_CHANGES,
        required_invariants=(
            *IDENTITY_INVARIANTS,
            "detector_events",
            "event_layers",
            "observable_flip",
        ),
        preserve_event_record=False,
    ),
}


def get_intervention_spec(intervention: str) -> InterventionSpec:
    try:
        return INTERVENTION_REGISTRY[intervention]
    except KeyError as exc:
        choices = ", ".join(sorted(INTERVENTION_REGISTRY))
        raise ValueError(f"unknown intervention {intervention!r}; choose from: {choices}") from exc


def validate_registry_complete() -> None:
    missing = sorted(set(P3_INTERVENTIONS) - set(INTERVENTION_REGISTRY))
    if missing:
        raise ValueError(f"registry mismatch: missing={missing}")
    for intervention in P3_NOISE_INTERVENTIONS:
        if INTERVENTION_REGISTRY[intervention].intervention_class != "noise":
            raise ValueError(f"{intervention} must be registered as noise")
    for intervention in P3_RUNTIME_INTERVENTIONS:
        if INTERVENTION_REGISTRY[intervention].intervention_class not in {"runtime", "policy"}:
            raise ValueError(f"{intervention} must be registered as runtime or policy")


def serialize_fields(fields: tuple[str, ...]) -> str:
    return "|".join(fields)


validate_registry_complete()
