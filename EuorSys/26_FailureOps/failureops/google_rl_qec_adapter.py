"""Adapter for Google public QEC real-device records used in P7.

P7 intentionally starts with real QEC shot records, not real runtime traces.
It imports detector events, actual observable flips, and decoder predictions
from the Google Quantum AI RL QEC dataset, then compares decoder pathways over
the same shot-level detector record.
"""

from __future__ import annotations

import json
import statistics
from copy import deepcopy
from pathlib import Path
from typing import Any

from failureops.event_layers import event_layer_hash, serialize_event_layers
from failureops.io_utils import fmt_float, parse_bool, parse_int
from failureops.paired_metrics import summarize_paired_effects
from failureops.pairing import build_pair_id, event_record_hash, record_hash, validate_pair
from failureops.intervention_registry import get_intervention_spec

DECODER_INTERVENTION = "switch_decoder_pathway"
DECODER_PATHWAYS = (
    "correlated_matching_decoder_with_si1000_prior",
    "tesseract_decoder_with_si1000_prior",
)
CONTROL_MODES = (
    "reinforcement_learning",
    "traditional_calibration",
    "traditional_calibration_and_rl_fine_tuning",
)
EXPERIMENT_CODE_PREFIXES = (
    "surface_code",
    "color_code",
    "repetition_code",
)
GENERIC_DATASET_PREFIXES = (
    "google_",
    "sycamore_",
)


def load_google_rl_qec_records(
    data_dir: str | Path,
    *,
    baseline_decoder_pathway: str,
    intervened_decoder_pathway: str,
    max_shots: int | None = None,
    run_id: str = "p7_google_rl_qec",
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Load one Google RL QEC experiment directory into P7 rows."""
    data_dir = Path(data_dir)
    metadata = read_metadata(data_dir)
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
    baseline_predictions = read_decoder_predictions(
        data_dir,
        baseline_decoder_pathway,
        bits_per_shot=circuit.num_observables,
        max_shots=max_shots,
    )
    intervened_predictions = read_decoder_predictions(
        data_dir,
        intervened_decoder_pathway,
        bits_per_shot=circuit.num_observables,
        max_shots=max_shots,
    )

    num_shots = len(detector_samples)
    if not (len(actual_flips) == len(baseline_predictions) == len(intervened_predictions) == num_shots):
        raise ValueError("detector, observable, and prediction files have mismatched shot counts")

    baseline_rows = []
    intervention_rows = []
    for shot_id in range(num_shots):
        detector_events = detector_indices(detector_samples[shot_id])
        observable_flip = bool(actual_flips[shot_id][0])
        baseline = build_baseline_row(
            data_dir=data_dir,
            metadata=metadata,
            run_id=run_id,
            shot_id=shot_id,
            detector_count=circuit.num_detectors,
            detector_events=detector_events,
            observable_flip=observable_flip,
            decoder_pathway=baseline_decoder_pathway,
            decoder_prediction=bool(baseline_predictions[shot_id][0]),
        )
        intervened = build_intervened_decoder_row(
            baseline,
            decoder_pathway=intervened_decoder_pathway,
            decoder_prediction=bool(intervened_predictions[shot_id][0]),
        )
        baseline_rows.append(baseline)
        intervention_rows.append(build_decoder_intervention_row(baseline, intervened))
    return baseline_rows, intervention_rows


def discover_google_rl_qec_data_dirs(
    root: str | Path,
    *,
    baseline_decoder_pathway: str | None = None,
    intervened_decoder_pathway: str | None = None,
    required_decoder_pathways: tuple[str, ...] | None = None,
) -> list[Path]:
    """Find leaf experiment directories in extracted Google real-record datasets."""
    root = Path(root)
    dirs = []
    if required_decoder_pathways is None:
        requested = tuple(
            pathway
            for pathway in (baseline_decoder_pathway, intervened_decoder_pathway)
            if pathway
        )
        required_decoder_pathways = requested or None
    for metadata_path in root.rglob("metadata.json"):
        data_dir = metadata_path.parent
        if not all((data_dir / name).exists() for name in required_files()):
            continue
        available = available_decoder_pathways(data_dir)
        if required_decoder_pathways is None:
            if len(available) < 2:
                continue
        elif not all(pathway in available for pathway in required_decoder_pathways):
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


def available_decoder_pathways(data_dir: str | Path) -> tuple[str, ...]:
    data_dir = Path(data_dir)
    decoding_root = data_dir / "decoding_results"
    if not decoding_root.exists():
        return ()
    pathways = {
        path.parent.name
        for path in decoding_root.glob("*/obs_flips_predicted.b8")
    }
    return tuple(sorted(pathways))


def summarize_google_rl_qec_condition(
    data_dir: str | Path,
    *,
    baseline_decoder_pathway: str,
    intervened_decoder_pathway: str,
    max_shots: int | None,
    run_id: str,
    num_bootstrap: int = 100,
    bootstrap_seed: int = 2026,
) -> dict[str, object]:
    baseline_rows, intervention_rows = load_google_rl_qec_records(
        data_dir,
        baseline_decoder_pathway=baseline_decoder_pathway,
        intervened_decoder_pathway=intervened_decoder_pathway,
        max_shots=max_shots,
        run_id=run_id,
    )
    effects = summarize_paired_effects(
        intervention_rows,
        num_bootstrap=num_bootstrap,
        bootstrap_seed=bootstrap_seed,
    )
    if len(effects) != 1:
        raise ValueError(f"expected exactly one paired-effect row for {data_dir}, got {len(effects)}")
    effect = effects[0]
    first = baseline_rows[0]
    valid_pairs = parse_int(effect["valid_pairs"])
    baseline_failures = parse_int(effect["baseline_failure_count"])
    intervened_failures = parse_int(effect["intervened_failure_count"])
    invalid_pairs = parse_int(effect["invalid_pairs"])
    return {
        "experiment_name": first["experiment_name"],
        "code_family": first["code_family"],
        "control_mode": control_mode_from_experiment(str(first["experiment_name"])),
        "basis": first["basis"],
        "cycle_dir": first["cycle_dir"],
        "cycles": first["cycles"],
        "workload_id": first["workload_id"],
        "baseline_decoder_pathway": baseline_decoder_pathway,
        "intervened_decoder_pathway": intervened_decoder_pathway,
        "num_pairs": effect["num_pairs"],
        "valid_pairs": valid_pairs,
        "invalid_pairs": invalid_pairs,
        "baseline_failure_count": baseline_failures,
        "intervened_failure_count": intervened_failures,
        "rescued_failure_count": effect["rescued_failure_count"],
        "induced_failure_count": effect["induced_failure_count"],
        "unchanged_failure_count": effect["unchanged_failure_count"],
        "unchanged_success_count": effect["unchanged_success_count"],
        "net_rescue_count": effect["net_rescue_count"],
        "baseline_lfr": fmt_float(baseline_failures / valid_pairs if valid_pairs else 0.0),
        "intervened_lfr": fmt_float(intervened_failures / valid_pairs if valid_pairs else 0.0),
        "paired_delta_lfr": effect["paired_delta_lfr"],
        "net_rescue_rate": effect["net_rescue_rate"],
        "rescue_rate_among_baseline_failures": effect["rescue_rate_among_baseline_failures"],
        "induction_rate_among_baseline_successes": effect["induction_rate_among_baseline_successes"],
        "pairing_violation_rate": fmt_float(invalid_pairs / parse_int(effect["num_pairs"]) if parse_int(effect["num_pairs"]) else 0.0),
        "source_data_dir": str(data_dir),
    }


def summarize_p7_sweep_groups(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for group_by in ("code_family", "control_mode", "basis", "cycles"):
        values = sorted({str(row[group_by]) for row in rows})
        for value in values:
            group = [row for row in rows if str(row[group_by]) == value]
            out.append(summarize_p7_sweep_group(group_by, value, group))
    out.append(summarize_p7_sweep_group("all", "all", rows))
    return out


def summarize_p7_sweep_group(
    group_by: str,
    group_value: str,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    deltas = [float(row["paired_delta_lfr"]) for row in rows]
    net_rates = [float(row["net_rescue_rate"]) for row in rows]
    baseline_lfrs = [float(row["baseline_lfr"]) for row in rows]
    intervened_lfrs = [float(row["intervened_lfr"]) for row in rows]
    strongest = min(rows, key=lambda row: float(row["paired_delta_lfr"])) if rows else {}
    return {
        "group_by": group_by,
        "group_value": group_value,
        "num_conditions": len(rows),
        "total_pairs": sum(parse_int(row["num_pairs"]) for row in rows),
        "mean_baseline_lfr": fmt_float(statistics.fmean(baseline_lfrs) if baseline_lfrs else 0.0),
        "mean_intervened_lfr": fmt_float(statistics.fmean(intervened_lfrs) if intervened_lfrs else 0.0),
        "mean_paired_delta_lfr": fmt_float(statistics.fmean(deltas) if deltas else 0.0),
        "min_paired_delta_lfr": fmt_float(min(deltas) if deltas else 0.0),
        "max_paired_delta_lfr": fmt_float(max(deltas) if deltas else 0.0),
        "mean_net_rescue_rate": fmt_float(statistics.fmean(net_rates) if net_rates else 0.0),
        "strongest_condition": strongest.get("workload_id", ""),
        "strongest_paired_delta_lfr": strongest.get("paired_delta_lfr", "0.000000"),
    }


def read_metadata(data_dir: Path) -> dict[str, Any]:
    with (data_dir / "metadata.json").open() as handle:
        metadata = json.load(handle)
    return {
        **metadata,
        **infer_google_experiment_context(data_dir),
    }


def infer_google_experiment_context(data_dir: Path) -> dict[str, str]:
    basis = data_dir.parent.name
    cycle_dir = data_dir.name
    prefix_tokens = experiment_prefix_tokens(data_dir)
    nearest = prefix_tokens[0] if len(prefix_tokens) > 0 else ""
    next_nearest = prefix_tokens[1] if len(prefix_tokens) > 1 else ""
    third_nearest = prefix_tokens[2] if len(prefix_tokens) > 2 else ""

    if nearest.startswith(EXPERIMENT_CODE_PREFIXES):
        experiment_name = nearest
        condition_id = ""
    elif nearest in CONTROL_MODES:
        experiment_name = join_identifier_tokens(next_nearest, nearest)
        condition_id = ""
    elif next_nearest in CONTROL_MODES:
        experiment_name = join_identifier_tokens(third_nearest, next_nearest)
        condition_id = nearest
    elif third_nearest.startswith(GENERIC_DATASET_PREFIXES):
        experiment_name = join_identifier_tokens(third_nearest, next_nearest)
        condition_id = nearest
    elif next_nearest.startswith(GENERIC_DATASET_PREFIXES):
        experiment_name = next_nearest
        condition_id = nearest
    else:
        experiment_name = nearest or "google_qec"
        condition_id = ""

    return {
        "experiment_name": experiment_name,
        "basis": basis,
        "cycle_dir": cycle_dir,
        "condition_id": condition_id,
    }


def experiment_prefix_tokens(data_dir: Path, *, max_depth: int = 3) -> list[str]:
    tokens = []
    current = data_dir.parent.parent
    while current.name and len(tokens) < max_depth:
        tokens.append(current.name)
        current = current.parent
    return tokens


def join_identifier_tokens(*tokens: str) -> str:
    return "_".join(token.replace("/", "_") for token in tokens if token)


def read_circuit(path: Path):
    import stim

    return stim.Circuit.from_file(path)


def read_b8(path: Path, *, bits_per_shot: int, max_shots: int | None = None):
    import stim

    samples = stim.read_shot_data_file(
        path=path,
        format="b8",
        num_measurements=bits_per_shot,
    )
    if max_shots is not None:
        samples = samples[:max_shots]
    return samples


def read_decoder_predictions(
    data_dir: Path,
    decoder_pathway: str,
    *,
    bits_per_shot: int,
    max_shots: int | None,
):
    path = data_dir / "decoding_results" / decoder_pathway / "obs_flips_predicted.b8"
    if not path.exists():
        available = ", ".join(available_decoder_pathways(data_dir))
        raise FileNotFoundError(
            f"missing decoder prediction file for pathway {decoder_pathway!r} under {data_dir}; "
            f"available pathways: {available or 'none'}"
        )
    return read_b8(
        path,
        bits_per_shot=bits_per_shot,
        max_shots=max_shots,
    )


def detector_indices(bits: Any) -> list[int]:
    return [index for index, value in enumerate(bits) if bool(value)]


def build_baseline_row(
    *,
    data_dir: Path,
    metadata: dict[str, Any],
    run_id: str,
    shot_id: int,
    detector_count: int,
    detector_events: list[int],
    observable_flip: bool,
    decoder_pathway: str,
    decoder_prediction: bool,
) -> dict[str, object]:
    experiment_name = str(metadata["experiment_name"])
    condition_id = str(metadata.get("condition_id", ""))
    basis = str(metadata["basis"])
    cycle_dir = str(metadata["cycle_dir"])
    cycles = parse_int(metadata["rounds"])
    workload_id = google_workload_id(metadata)
    event_layers = serialize_event_layers({})
    row: dict[str, object] = {
        "run_id": run_id,
        "workload_id": workload_id,
        "stress_level": "real_device",
        "experiment_name": experiment_name,
        "basis": basis,
        "cycles": cycles,
        "cycle_dir": cycle_dir,
        "source_data_dir": str(data_dir),
        "circuit_id": workload_id,
        "backend": "google_willow_real_device",
        "code_family": code_family_from_experiment(experiment_name),
        "shot_id": shot_id,
        "seed": 0,
        "detector_count": detector_count,
        "detector_event_count": len(detector_events),
        "detector_events": json.dumps(detector_events, separators=(",", ":")),
        "observable_flip": observable_flip,
        "decoder_pathway": decoder_pathway,
        "decoder_prediction": decoder_prediction,
        "event_layers": event_layers,
    }
    if condition_id:
        row["condition_id"] = condition_id
    row["event_layer_hash"] = event_layer_hash(row)
    row = recompute_real_qec_failure_behavior(row)
    row["event_record_hash"] = event_record_hash(row)
    row["record_hash"] = record_hash(row)
    return row


def build_intervened_decoder_row(
    baseline: dict[str, object],
    *,
    decoder_pathway: str,
    decoder_prediction: bool,
) -> dict[str, object]:
    row = deepcopy(baseline)
    for field in ("event_record_hash", "record_hash"):
        row.pop(field, None)
    row["decoder_pathway"] = decoder_pathway
    row["decoder_prediction"] = decoder_prediction
    row = recompute_real_qec_failure_behavior(row)
    row["event_record_hash"] = event_record_hash(row)
    row["record_hash"] = record_hash(row)
    return row


def recompute_real_qec_failure_behavior(record: dict[str, object]) -> dict[str, object]:
    row = deepcopy(record)
    observable_flip = parse_bool(row["observable_flip"])
    decoder_prediction = parse_bool(row["decoder_prediction"])
    logical_failure = observable_flip != decoder_prediction
    row["qec_decoder_failure"] = logical_failure
    row["logical_failure"] = logical_failure

    if not logical_failure:
        row["failure_round"] = ""
        row["failure_region"] = "none"
        row["failure_operation"] = ""
        row["failure_mode"] = "none"
        row["failure_pattern"] = "no_failure"
        return row

    detector_events = json.loads(str(row.get("detector_events", "[]")))
    first_detector = int(detector_events[0]) if detector_events else 0
    detector_count = parse_int(row.get("detector_count", 1), 1)
    cycles = parse_int(row.get("cycles", 1), 1)
    detectors_per_cycle = max(1, detector_count // max(1, cycles))
    failure_round = min(cycles, first_detector // detectors_per_cycle)
    row["failure_round"] = failure_round
    row["failure_region"] = f"detector_{first_detector % detectors_per_cycle}" if detector_events else "logical"
    row["failure_operation"] = failure_round
    row["failure_mode"] = "decoder_pathway_disagreement"
    row["failure_pattern"] = (
        "real_detector_burst"
        if len(detector_events) >= max(2, detectors_per_cycle)
        else "real_decoder_mismatch"
    )
    return row


def build_decoder_intervention_row(
    baseline: dict[str, object],
    intervened: dict[str, object],
) -> dict[str, object]:
    spec = get_intervention_spec(DECODER_INTERVENTION)
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
        "intervention": DECODER_INTERVENTION,
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


def code_family_from_experiment(experiment_name: str) -> str:
    if "surface_code" in experiment_name:
        return "surface_code_memory"
    if "color_code" in experiment_name:
        return "color_code_memory"
    if "repetition_code" in experiment_name:
        return "repetition_code_memory"
    return "unknown_qec_memory"


def control_mode_from_experiment(experiment_name: str) -> str:
    if experiment_name.endswith("traditional_calibration_and_rl_fine_tuning"):
        return "traditional_calibration_and_rl_fine_tuning"
    if experiment_name.endswith("reinforcement_learning"):
        return "reinforcement_learning"
    if experiment_name.endswith("traditional_calibration"):
        return "traditional_calibration"
    return "unknown"


def google_workload_id(metadata: dict[str, Any]) -> str:
    return join_identifier_tokens(
        str(metadata["experiment_name"]),
        str(metadata.get("condition_id", "")),
        str(metadata["basis"]),
        str(metadata["cycle_dir"]),
    )
