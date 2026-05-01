"""P7.5 real-data analyses for EuroSys-facing FailureOps evaluation."""

from __future__ import annotations

import math
import os
import random
import re
import statistics
from pathlib import Path

from failureops.google_rl_qec_adapter import (
    build_baseline_row,
    build_decoder_intervention_row,
    build_intervened_decoder_row,
    code_family_from_experiment,
    control_mode_from_experiment,
    detector_indices,
    read_b8,
    read_circuit,
    read_metadata,
)
from failureops.io_utils import fmt_float, parse_bool
from failureops.paired_metrics import summarize_paired_effects

os.environ.setdefault("MPLCONFIGDIR", "/tmp/failureops-matplotlib")

TRANSITION_CLASSES = (
    "rescued",
    "induced",
    "unchanged_failure",
    "unchanged_success",
)

PRIOR_VARIANTS = (
    "uniform_median_probability",
    "flatten_log_odds_50pct",
    "sharpen_log_odds_2x",
    "probability_floor_1e-3",
    "probability_ceiling_1e-2",
)


def summarize_rescue_induction_features(
    data_dir: str | Path,
    *,
    baseline_decoder_pathway: str,
    intervened_decoder_pathway: str,
    max_shots: int | None,
    run_id: str,
) -> list[dict[str, object]]:
    from failureops.google_rl_qec_adapter import load_google_rl_qec_records

    baseline_rows, intervention_rows = load_google_rl_qec_records(
        data_dir,
        baseline_decoder_pathway=baseline_decoder_pathway,
        intervened_decoder_pathway=intervened_decoder_pathway,
        max_shots=max_shots,
        run_id=run_id,
    )
    baseline_by_pair = {str(row["pair_id"]): baseline for row, baseline in zip(intervention_rows, baseline_rows)}
    groups: dict[str, list[dict[str, object]]] = {name: [] for name in TRANSITION_CLASSES}
    for row in intervention_rows:
        groups[transition_class(row)].append(baseline_by_pair[str(row["pair_id"])])

    first = baseline_rows[0]
    out = []
    for name in TRANSITION_CLASSES:
        out.append(feature_row(name, groups[name], first, str(data_dir)))
    return out


def transition_class(row: dict[str, object]) -> str:
    baseline = parse_bool(row["baseline_logical_failure"])
    intervened = parse_bool(row["intervened_logical_failure"])
    if baseline and not intervened:
        return "rescued"
    if (not baseline) and intervened:
        return "induced"
    if baseline and intervened:
        return "unchanged_failure"
    return "unchanged_success"


def feature_row(
    transition: str,
    rows: list[dict[str, object]],
    exemplar: dict[str, object],
    source_data_dir: str,
) -> dict[str, object]:
    features = [detector_features(row) for row in rows]
    return {
        "experiment_name": exemplar["experiment_name"],
        "code_family": exemplar["code_family"],
        "control_mode": control_mode_from_experiment(str(exemplar["experiment_name"])),
        "basis": exemplar["basis"],
        "cycles": exemplar["cycles"],
        "workload_id": exemplar["workload_id"],
        "transition_class": transition,
        "num_shots": len(rows),
        "mean_detector_event_count": fmt_float(mean(feature["count"] for feature in features)),
        "mean_early_detector_count": fmt_float(mean(feature["early"] for feature in features)),
        "mean_mid_detector_count": fmt_float(mean(feature["mid"] for feature in features)),
        "mean_late_detector_count": fmt_float(mean(feature["late"] for feature in features)),
        "burst_fraction": fmt_float(mean(feature["is_burst"] for feature in features)),
        "mean_first_detector_fraction": fmt_float(mean(feature["first_fraction"] for feature in features)),
        "source_data_dir": source_data_dir,
    }


def detector_features(row: dict[str, object]) -> dict[str, float]:
    import json

    detector_events = [int(item) for item in json.loads(str(row.get("detector_events", "[]")))]
    detector_count = max(1, int(row.get("detector_count", 1)))
    cycles = max(1, int(row.get("cycles", 1)))
    first = detector_events[0] if detector_events else 0
    early_limit = detector_count / 3.0
    late_limit = 2.0 * detector_count / 3.0
    burst_threshold = max(2, detector_count / cycles)
    return {
        "count": float(len(detector_events)),
        "early": float(sum(1 for event in detector_events if event < early_limit)),
        "mid": float(sum(1 for event in detector_events if early_limit <= event < late_limit)),
        "late": float(sum(1 for event in detector_events if event >= late_limit)),
        "is_burst": float(len(detector_events) >= burst_threshold),
        "first_fraction": first / detector_count,
    }


def summarize_paired_vs_unpaired_variance(
    data_dir: str | Path,
    *,
    baseline_decoder_pathway: str,
    intervened_decoder_pathway: str,
    max_shots: int | None,
    run_id: str,
    num_bootstrap: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    from failureops.google_rl_qec_adapter import load_google_rl_qec_records

    baseline_rows, intervention_rows = load_google_rl_qec_records(
        data_dir,
        baseline_decoder_pathway=baseline_decoder_pathway,
        intervened_decoder_pathway=intervened_decoder_pathway,
        max_shots=max_shots,
        run_id=run_id,
    )
    first = baseline_rows[0]
    paired_values, unpaired_values = bootstrap_delta_values(
        intervention_rows,
        num_bootstrap=num_bootstrap,
        seed=bootstrap_seed,
    )
    paired_delta = mean(
        int(parse_bool(row["intervened_logical_failure"]))
        - int(parse_bool(row["baseline_logical_failure"]))
        for row in intervention_rows
    )
    paired_std = sample_std(paired_values)
    unpaired_std = sample_std(unpaired_values)
    return {
        "experiment_name": first["experiment_name"],
        "code_family": first["code_family"],
        "control_mode": control_mode_from_experiment(str(first["experiment_name"])),
        "basis": first["basis"],
        "cycles": first["cycles"],
        "workload_id": first["workload_id"],
        "num_pairs": len(intervention_rows),
        "paired_delta_lfr": fmt_float(paired_delta),
        "paired_bootstrap_std": fmt_float(paired_std),
        "unpaired_bootstrap_std": fmt_float(unpaired_std),
        "std_ratio_unpaired_over_paired": fmt_float(unpaired_std / paired_std if paired_std else 0.0),
        "paired_ci_lower": fmt_float(percentile(paired_values, 0.025)),
        "paired_ci_upper": fmt_float(percentile(paired_values, 0.975)),
        "unpaired_ci_lower": fmt_float(percentile(unpaired_values, 0.025)),
        "unpaired_ci_upper": fmt_float(percentile(unpaired_values, 0.975)),
        "num_bootstrap": num_bootstrap,
        "source_data_dir": str(data_dir),
    }


def bootstrap_delta_values(
    rows: list[dict[str, object]],
    *,
    num_bootstrap: int,
    seed: int,
) -> tuple[list[float], list[float]]:
    rng = random.Random(seed)
    pair_diffs = [
        int(parse_bool(row["intervened_logical_failure"]))
        - int(parse_bool(row["baseline_logical_failure"]))
        for row in rows
    ]
    baseline = [int(parse_bool(row["baseline_logical_failure"])) for row in rows]
    intervened = [int(parse_bool(row["intervened_logical_failure"])) for row in rows]
    paired_values = []
    unpaired_values = []
    for _ in range(num_bootstrap):
        paired_values.append(mean(pair_diffs[rng.randrange(len(pair_diffs))] for _ in pair_diffs))
        baseline_sample = mean(baseline[rng.randrange(len(baseline))] for _ in baseline)
        intervened_sample = mean(intervened[rng.randrange(len(intervened))] for _ in intervened)
        unpaired_values.append(intervened_sample - baseline_sample)
    paired_values.sort()
    unpaired_values.sort()
    return paired_values, unpaired_values


def run_decoder_prior_interventions(
    data_dir: str | Path,
    *,
    max_shots: int | None,
    run_id: str,
    num_bootstrap: int,
    bootstrap_seed: int,
) -> list[dict[str, object]]:
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
    dem_path = data_dir / "decoding_results" / "correlated_matching_decoder_with_si1000_prior" / "error_model.dem"
    original_predictions = decode_with_dem_text(dem_path.read_text(), detector_samples)

    out = []
    for variant in PRIOR_VARIANTS:
        variant_predictions = decode_with_dem_text(transform_dem_text(dem_path.read_text(), variant), detector_samples)
        rows = []
        for shot_id in range(len(detector_samples)):
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
                decoder_pathway="pymatching_original_dem",
                decoder_prediction=bool(original_predictions[shot_id][0]),
            )
            intervened = build_intervened_decoder_row(
                baseline,
                decoder_pathway=f"pymatching_{variant}",
                decoder_prediction=bool(variant_predictions[shot_id][0]),
            )
            rows.append(build_decoder_intervention_row(baseline, intervened))
        effect = summarize_paired_effects(
            rows,
            num_bootstrap=num_bootstrap,
            bootstrap_seed=bootstrap_seed,
        )[0]
        out.append(
            {
                "source_data_dir": str(data_dir),
                "workload_id": rows[0]["workload_id"],
                "baseline_prior": "pymatching_original_dem",
                "intervened_prior": f"pymatching_{variant}",
                **effect,
            }
        )
    return out


def decode_with_dem_text(dem_text: str, detector_samples):
    import pymatching
    import stim

    dem = stim.DetectorErrorModel(dem_text)
    matching = pymatching.Matching.from_detector_error_model(dem)
    return matching.decode_batch(detector_samples)


def transform_dem_text(dem_text: str, variant: str) -> str:
    probabilities = [float(match.group(1)) for match in re.finditer(r"error\(([^)]+)\)", dem_text)]
    median_probability = statistics.median(probabilities) if probabilities else 0.001

    def replace(match: re.Match[str]) -> str:
        probability = float(match.group(1))
        transformed = transform_probability(probability, variant, median_probability)
        return f"error({transformed:.17g})"

    return re.sub(r"error\(([^)]+)\)", replace, dem_text)


def transform_probability(probability: float, variant: str, median_probability: float) -> float:
    probability = min(max(probability, 1e-12), 1.0 - 1e-12)
    if variant == "uniform_median_probability":
        return median_probability
    if variant == "flatten_log_odds_50pct":
        return probability_from_log_odds_weight(log_odds_weight(probability) * 0.5)
    if variant == "sharpen_log_odds_2x":
        return probability_from_log_odds_weight(log_odds_weight(probability) * 2.0)
    if variant == "probability_floor_1e-3":
        return max(probability, 1e-3)
    if variant == "probability_ceiling_1e-2":
        return min(probability, 1e-2)
    raise ValueError(f"unknown DEM prior variant: {variant}")


def log_odds_weight(probability: float) -> float:
    return math.log((1.0 - probability) / probability)


def probability_from_log_odds_weight(weight: float) -> float:
    return 1.0 / (1.0 + math.exp(weight))


def mean(values) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, round(quantile * (len(values) - 1))))
    return values[index]
