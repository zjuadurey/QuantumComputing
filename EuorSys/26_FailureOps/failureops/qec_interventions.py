"""P1 interventions for Stim/PyMatching-backed records."""

from __future__ import annotations

from failureops.io_utils import parse_bool, parse_float, parse_int
from failureops.qec_backend import apply_runtime_intervention, generate_qec_runs

QEC_INTERVENTIONS = [
    "remove_data_noise",
    "weaken_data_noise_50pct",
    "remove_measurement_noise",
    "weaken_measurement_noise_50pct",
    "remove_idle_noise",
    "weaken_idle_noise_50pct",
    "remove_decoder_timeout",
    "reduce_decoder_delay_50pct",
    "reduce_idle_exposure_50pct",
]

NOISE_INTERVENTIONS = {
    "remove_data_noise",
    "weaken_data_noise_50pct",
    "remove_measurement_noise",
    "weaken_measurement_noise_50pct",
    "remove_idle_noise",
    "weaken_idle_noise_50pct",
}

RUNTIME_INTERVENTIONS = {
    "remove_decoder_timeout",
    "reduce_decoder_delay_50pct",
    "reduce_idle_exposure_50pct",
}


def generate_noise_intervention_rows(
    *,
    baseline_rows: list[dict[str, object]],
    intervention: str,
) -> list[dict[str, object]]:
    if intervention not in NOISE_INTERVENTIONS:
        raise ValueError(f"not a noise intervention: {intervention}")
    if not baseline_rows:
        return []

    first = baseline_rows[0]
    num_shots = len(baseline_rows)
    data_error_rate = parse_float(first["data_error_rate"])
    measurement_error_rate = parse_float(first["measurement_error_rate"])
    idle_error_rate = parse_float(first["idle_error_rate"])

    if intervention == "remove_data_noise":
        data_error_rate = 0.0
    elif intervention == "weaken_data_noise_50pct":
        data_error_rate *= 0.5
    elif intervention == "remove_measurement_noise":
        measurement_error_rate = 0.0
    elif intervention == "weaken_measurement_noise_50pct":
        measurement_error_rate *= 0.5
    elif intervention == "remove_idle_noise":
        idle_error_rate = 0.0
    elif intervention == "weaken_idle_noise_50pct":
        idle_error_rate *= 0.5

    seed = parse_int(first["seed"])
    intervened_rows = generate_qec_runs(
        num_shots=num_shots,
        distance=parse_int(first["distance"]),
        num_rounds=parse_int(first["num_rounds"]),
        data_error_rate=data_error_rate,
        measurement_error_rate=measurement_error_rate,
        idle_error_rate=idle_error_rate,
        decoder_timeout_base_rate=parse_float(first["decoder_timeout_base_rate"]),
        seed=seed,
        decoder_capacity=parse_float(first.get("decoder_capacity", 4.0)),
        synchronization_slack=parse_float(first.get("synchronization_slack", 0.45)),
        run_id=str(first["run_id"]),
        circuit_id=str(first["circuit_id"]),
    )
    return [
        compare_qec_records(baseline, intervened, intervention)
        for baseline, intervened in zip(baseline_rows, intervened_rows)
    ]


def generate_runtime_intervention_rows(
    *,
    baseline_rows: list[dict[str, object]],
    intervention: str,
) -> list[dict[str, object]]:
    if intervention not in RUNTIME_INTERVENTIONS:
        raise ValueError(f"not a runtime intervention: {intervention}")
    return [
        compare_qec_records(baseline, apply_runtime_intervention(baseline, intervention), intervention)
        for baseline in baseline_rows
    ]


def compare_qec_records(
    baseline: dict[str, object],
    intervened: dict[str, object],
    intervention: str,
) -> dict[str, object]:
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
