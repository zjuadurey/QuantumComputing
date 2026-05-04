"""Adapter for the Google decoder-prior public QEC record corpus.

This dataset exposes the same detector-record shots under multiple DEM prior
variants and multiple decoder backends. FailureOps uses it as a paired
decoder-prior intervention corpus: keep the detector record and decoder backend
fixed, then switch only the DEM prior used to produce the predicted logical
observable.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from failureops.event_layers import event_layer_hash, serialize_event_layers
from failureops.google_rl_qec_adapter import (
    code_family_from_experiment,
    detector_indices,
    read_b8,
    read_circuit,
    recompute_real_qec_failure_behavior,
)
from failureops.intervention_registry import get_intervention_spec
from failureops.io_utils import parse_bool, parse_int
from failureops.pairing import build_pair_id, event_record_hash, record_hash, validate_pair

DECODER_PRIOR_INTERVENTION = "switch_decoder_prior"
DEFAULT_BASELINE_PRIOR = "dem_simple"
GENERIC_INTERVENED_PRIORS = ("dem_correlations",)
MATCHED_INTERVENED_PRIORS = {
    "correlated_matching": (
        "dem_rl_isolated_correlated_matching",
        "dem_rl_shared_correlated_matching",
    ),
    "harmony": (
        "dem_rl_isolated_harmony",
        "dem_rl_shared_harmony",
    ),
    "belief_matching": (
        "dem_rl_isolated_belief_matching",
        "dem_rl_shared_belief_matching",
    ),
}


def discover_google_decoder_prior_data_dirs(
    root: str | Path,
    *,
    decoder_backend: str | None = None,
    required_prior_variants: tuple[str, ...] | None = None,
) -> list[Path]:
    """Find leaf experiment directories in the decoder-prior corpus."""
    root = Path(root)
    dirs = []
    for metadata_path in root.rglob("metadata.json"):
        data_dir = metadata_path.parent
        if not all((data_dir / name).exists() for name in required_files()):
            continue
        available_priors = available_prior_variants(data_dir, decoder_backend=decoder_backend)
        if decoder_backend and decoder_backend not in available_decoder_backends(data_dir):
            continue
        if required_prior_variants is None:
            if len(available_priors) < 2:
                continue
        elif not all(prior in available_priors for prior in required_prior_variants):
            continue
        dirs.append(data_dir)
    return sorted(dirs)


def required_files() -> tuple[str, ...]:
    return (
        "circuit_ideal.stim",
        "detection_events.b8",
        "obs_flips_actual.b8",
        "metadata.json",
    )


def available_prior_variants(
    data_dir: str | Path,
    *,
    decoder_backend: str | None = None,
) -> tuple[str, ...]:
    data_dir = Path(data_dir)
    prediction_root = data_dir / "obs_flips_predicted"
    if not prediction_root.exists():
        return ()
    variants = []
    for prior_dir in sorted(prediction_root.iterdir()):
        if not prior_dir.is_dir():
            continue
        if decoder_backend is None and any(prior_dir.glob("*.b8")):
            variants.append(prior_dir.name)
        elif decoder_backend is not None and (prior_dir / f"{decoder_backend}.b8").exists():
            variants.append(prior_dir.name)
    return tuple(variants)


def available_decoder_backends(
    data_dir: str | Path,
    *,
    prior_variant: str | None = None,
) -> tuple[str, ...]:
    data_dir = Path(data_dir)
    prediction_root = data_dir / "obs_flips_predicted"
    if not prediction_root.exists():
        return ()
    backends = set()
    prior_dirs = []
    if prior_variant is None:
        prior_dirs = [path for path in prediction_root.iterdir() if path.is_dir()]
    else:
        prior_dir = prediction_root / prior_variant
        if prior_dir.is_dir():
            prior_dirs = [prior_dir]
    for prior_dir in prior_dirs:
        for prediction_path in prior_dir.glob("*.b8"):
            backends.add(prediction_path.stem)
    return tuple(sorted(backends))


def recommended_intervened_priors(decoder_backend: str) -> tuple[str, ...]:
    matched = MATCHED_INTERVENED_PRIORS.get(decoder_backend)
    if matched is None:
        choices = ", ".join(sorted(MATCHED_INTERVENED_PRIORS))
        raise ValueError(f"unknown decoder backend {decoder_backend!r}; choose from: {choices}")
    return (*GENERIC_INTERVENED_PRIORS, *matched)


def read_google_decoder_prior_metadata(data_dir: str | Path) -> dict[str, Any]:
    data_dir = Path(data_dir)
    with (data_dir / "metadata.json").open() as handle:
        metadata = json.load(handle)
    return {
        **metadata,
        **infer_google_decoder_prior_context(data_dir),
    }


def infer_google_decoder_prior_context(data_dir: Path) -> dict[str, str]:
    cycle_dir = data_dir.name
    basis = data_dir.parent.name
    placement_id = data_dir.parent.parent.name
    sample_id = data_dir.parent.parent.parent.name
    experiment_name = data_dir.parent.parent.parent.parent.name
    condition_id = join_identifier_tokens(sample_id, placement_id)
    return {
        "experiment_name": experiment_name,
        "condition_id": condition_id,
        "sample_id": sample_id,
        "placement_id": placement_id,
        "basis": basis,
        "cycle_dir": cycle_dir,
    }


def google_decoder_prior_workload_id(metadata: dict[str, Any]) -> str:
    return join_identifier_tokens(
        str(metadata["experiment_name"]),
        str(metadata["condition_id"]),
        str(metadata["basis"]),
        str(metadata["cycle_dir"]),
    )


def read_prior_predictions(
    data_dir: str | Path,
    *,
    prior_variant: str,
    decoder_backend: str,
    bits_per_shot: int,
    max_shots: int | None,
):
    data_dir = Path(data_dir)
    path = data_dir / "obs_flips_predicted" / prior_variant / f"{decoder_backend}.b8"
    if not path.exists():
        available_priors = ", ".join(available_prior_variants(data_dir, decoder_backend=decoder_backend))
        available_backends = ", ".join(available_decoder_backends(data_dir, prior_variant=prior_variant))
        raise FileNotFoundError(
            f"missing prior prediction file for prior {prior_variant!r} and backend {decoder_backend!r} "
            f"under {data_dir}; available priors for backend: {available_priors or 'none'}; "
            f"available backends for prior: {available_backends or 'none'}"
        )
    return read_b8(path, bits_per_shot=bits_per_shot, max_shots=max_shots)


def load_google_decoder_prior_records(
    data_dir: str | Path,
    *,
    baseline_prior: str,
    intervened_prior: str,
    decoder_backend: str,
    max_shots: int | None = None,
    run_id: str = "p10_google_decoder_priors",
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Load one decoder-prior condition into paired baseline/intervention rows."""
    data_dir = Path(data_dir)
    metadata = read_google_decoder_prior_metadata(data_dir)
    circuit = read_circuit(data_dir / "circuit_ideal.stim")
    detector_samples = read_b8(
        data_dir / "detection_events.b8",
        bits_per_shot=circuit.num_detectors,
        max_shots=max_shots,
    )
    actual_flips = read_b8(
        data_dir / "obs_flips_actual.b8",
        bits_per_shot=circuit.num_observables,
        max_shots=max_shots,
    )
    baseline_predictions = read_prior_predictions(
        data_dir,
        prior_variant=baseline_prior,
        decoder_backend=decoder_backend,
        bits_per_shot=circuit.num_observables,
        max_shots=max_shots,
    )
    intervened_predictions = read_prior_predictions(
        data_dir,
        prior_variant=intervened_prior,
        decoder_backend=decoder_backend,
        bits_per_shot=circuit.num_observables,
        max_shots=max_shots,
    )
    num_shots = len(detector_samples)
    if not (len(actual_flips) == len(baseline_predictions) == len(intervened_predictions) == num_shots):
        raise ValueError("detector, observable, and prior-prediction files have mismatched shot counts")

    baseline_rows = []
    intervention_rows = []
    for shot_id in range(num_shots):
        detector_events = detector_indices(detector_samples[shot_id])
        observable_flip = bool(actual_flips[shot_id][0])
        baseline = build_google_decoder_prior_baseline_row(
            data_dir=data_dir,
            metadata=metadata,
            run_id=run_id,
            shot_id=shot_id,
            detector_count=circuit.num_detectors,
            detector_events=detector_events,
            observable_flip=observable_flip,
            decoder_backend=decoder_backend,
            prior_variant=baseline_prior,
            decoder_prediction=bool(baseline_predictions[shot_id][0]),
        )
        intervened = build_google_decoder_prior_intervened_row(
            baseline,
            decoder_backend=decoder_backend,
            prior_variant=intervened_prior,
            decoder_prediction=bool(intervened_predictions[shot_id][0]),
        )
        baseline_rows.append(baseline)
        intervention_rows.append(build_decoder_prior_intervention_row(baseline, intervened))
    return baseline_rows, intervention_rows


def join_identifier_tokens(*tokens: str) -> str:
    return "_".join(token.replace("/", "_") for token in tokens if token)


def prior_prediction_label(decoder_backend: str, prior_variant: str) -> str:
    return f"{decoder_backend}:{prior_variant}"


def build_google_decoder_prior_baseline_row(
    *,
    data_dir: Path,
    metadata: dict[str, Any],
    run_id: str,
    shot_id: int,
    detector_count: int,
    detector_events: list[int],
    observable_flip: bool,
    decoder_backend: str,
    prior_variant: str,
    decoder_prediction: bool,
) -> dict[str, object]:
    experiment_name = str(metadata["experiment_name"])
    cycles = parse_int(metadata["rounds"])
    workload_id = google_decoder_prior_workload_id(metadata)
    event_layers = serialize_event_layers({})
    row: dict[str, object] = {
        "run_id": run_id,
        "workload_id": workload_id,
        "stress_level": "real_device",
        "experiment_name": experiment_name,
        "basis": str(metadata["basis"]),
        "cycles": cycles,
        "cycle_dir": str(metadata["cycle_dir"]),
        "source_data_dir": str(data_dir),
        "circuit_id": workload_id,
        "backend": "google_sycamore_real_device",
        "code_family": code_family_from_experiment(experiment_name),
        "shot_id": shot_id,
        "seed": 0,
        "detector_count": detector_count,
        "detector_event_count": len(detector_events),
        "detector_events": json.dumps(detector_events, separators=(",", ":")),
        "observable_flip": observable_flip,
        "decoder_pathway": prior_prediction_label(decoder_backend, prior_variant),
        "decoder_prediction": decoder_prediction,
        "event_layers": event_layers,
        "condition_id": str(metadata["condition_id"]),
        "sample_id": str(metadata["sample_id"]),
        "placement_id": str(metadata["placement_id"]),
    }
    row["event_layer_hash"] = event_layer_hash(row)
    row = recompute_real_qec_failure_behavior(row)
    row["event_record_hash"] = event_record_hash(row)
    row["record_hash"] = record_hash(row)
    return row


def build_google_decoder_prior_intervened_row(
    baseline: dict[str, object],
    *,
    decoder_backend: str,
    prior_variant: str,
    decoder_prediction: bool,
) -> dict[str, object]:
    row = deepcopy(baseline)
    for field in ("event_record_hash", "record_hash"):
        row.pop(field, None)
    row["decoder_pathway"] = prior_prediction_label(decoder_backend, prior_variant)
    row["decoder_prediction"] = decoder_prediction
    row = recompute_real_qec_failure_behavior(row)
    row["event_record_hash"] = event_record_hash(row)
    row["record_hash"] = record_hash(row)
    return row


def build_decoder_prior_intervention_row(
    baseline: dict[str, object],
    intervened: dict[str, object],
) -> dict[str, object]:
    spec = get_intervention_spec(DECODER_PRIOR_INTERVENTION)
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
        "intervention": DECODER_PRIOR_INTERVENTION,
        "intervention_class": spec.intervention_class,
        "intervention_allowed_changes": "|".join(spec.allowed_changes),
        "intervention_required_invariants": "|".join(spec.required_invariants),
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
        "baseline_decoder_pathway": baseline["decoder_pathway"],
        "intervened_decoder_pathway": intervened["decoder_pathway"],
    }
