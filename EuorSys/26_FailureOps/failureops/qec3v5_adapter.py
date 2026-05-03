"""Adapter for the Google qec3v5 public surface-code dataset.

The qec3v5 archive is independent from the Google RL QEC dataset already used
by P7. It contains shot-level detector records, observable flips, and decoder
predictions for several decoder pathways. This module maps those records into
FailureOps paired-transition summaries without claiming runtime attribution.
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path
from typing import Any

from failureops.experiment_config import parse_simple_yaml, yaml
from failureops.google_rl_qec_adapter import detector_indices
from failureops.io_utils import fmt_float, parse_int

QEC3V5_RECORD_RE = re.compile(
    r"(?P<code>surface_code|repetition_code)_b(?P<basis>[XZ])_d(?P<distance>\d+)"
    r"_r(?P<rounds>\d+)_center_(?P<center_row>\d+)_(?P<center_col>\d+)"
)

DECODER_FILES = {
    "pymatching": "obs_flips_predicted_by_pymatching.01",
    "correlated_matching": "obs_flips_predicted_by_correlated_matching.01",
    "belief_matching": "obs_flips_predicted_by_belief_matching.01",
    "tensor_network_contraction": "obs_flips_predicted_by_tensor_network_contraction.01",
}


def discover_qec3v5_data_dirs(
    root: str | Path,
    *,
    code: str = "surface_code",
    distance: int | None = 5,
    center: str | None = "5_5",
    baseline_decoder: str = "pymatching",
    intervened_decoder: str = "correlated_matching",
) -> list[Path]:
    root = Path(root)
    dirs = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        metadata = parse_qec3v5_dir_name(path.name)
        if metadata is None:
            continue
        if metadata["code"] != code:
            continue
        if distance is not None and metadata["distance"] != distance:
            continue
        if center is not None and f"{metadata['center_row']}_{metadata['center_col']}" != center:
            continue
        required = [
            "properties.yml",
            "detection_events.b8",
            "obs_flips_actual.01",
            DECODER_FILES[baseline_decoder],
            DECODER_FILES[intervened_decoder],
        ]
        if all((path / name).exists() for name in required):
            dirs.append(path)
    return sorted(dirs, key=lambda path: (parse_qec3v5_dir_name(path.name) or {}).get("sort_key", path.name))


def summarize_qec3v5_condition(
    data_dir: str | Path,
    *,
    baseline_decoder: str,
    intervened_decoder: str,
    max_shots: int | None,
) -> dict[str, object]:
    data_dir = Path(data_dir)
    metadata = parse_qec3v5_dir_name(data_dir.name)
    if metadata is None:
        raise ValueError(f"not a qec3v5 experiment directory: {data_dir}")
    properties = read_properties(data_dir / "properties.yml")
    detector_count = parse_int(properties["circuit_detectors"])
    actual = read_01(data_dir / "obs_flips_actual.01", max_shots=max_shots)
    baseline_prediction = read_01(data_dir / DECODER_FILES[baseline_decoder], max_shots=max_shots)
    intervened_prediction = read_01(data_dir / DECODER_FILES[intervened_decoder], max_shots=max_shots)
    detector_event_counts = read_detector_event_counts(
        data_dir / "detection_events.b8",
        detector_count=detector_count,
        max_shots=max_shots,
    )
    num_pairs = min(len(actual), len(baseline_prediction), len(intervened_prediction), len(detector_event_counts))
    if num_pairs == 0:
        raise ValueError(f"no qec3v5 shots found in {data_dir}")

    baseline_failures = [
        baseline_prediction[index] != actual[index]
        for index in range(num_pairs)
    ]
    intervened_failures = [
        intervened_prediction[index] != actual[index]
        for index in range(num_pairs)
    ]
    rescued = sum(baseline and not intervened for baseline, intervened in zip(baseline_failures, intervened_failures))
    induced = sum((not baseline) and intervened for baseline, intervened in zip(baseline_failures, intervened_failures))
    unchanged_failure = sum(baseline and intervened for baseline, intervened in zip(baseline_failures, intervened_failures))
    unchanged_success = sum((not baseline) and (not intervened) for baseline, intervened in zip(baseline_failures, intervened_failures))
    baseline_count = sum(baseline_failures)
    intervened_count = sum(intervened_failures)
    workload_id = f"qec3v5_{metadata['code']}_{metadata['basis']}_d{metadata['distance']}_r{metadata['rounds']}_center_{metadata['center_row']}_{metadata['center_col']}"
    baseline_success_count = num_pairs - baseline_count
    return {
        "source_dataset": "google_qec3v5_2022",
        "dataset_family": "google_surface_code_scaling_2022",
        "experiment_name": data_dir.name,
        "code_family": f"{metadata['code']}_memory",
        "basis": metadata["basis"],
        "distance": metadata["distance"],
        "cycles": metadata["rounds"],
        "center": f"{metadata['center_row']}_{metadata['center_col']}",
        "workload_id": workload_id,
        "baseline_decoder_pathway": baseline_decoder,
        "intervened_decoder_pathway": intervened_decoder,
        "num_pairs": num_pairs,
        "baseline_failure_count": baseline_count,
        "intervened_failure_count": intervened_count,
        "rescued_failure_count": rescued,
        "induced_failure_count": induced,
        "unchanged_failure_count": unchanged_failure,
        "unchanged_success_count": unchanged_success,
        "net_rescue_count": rescued - induced,
        "baseline_lfr": fmt_float(baseline_count / num_pairs),
        "intervened_lfr": fmt_float(intervened_count / num_pairs),
        "paired_delta_lfr": fmt_float((induced - rescued) / num_pairs),
        "net_rescue_rate": fmt_float((rescued - induced) / num_pairs),
        "rescue_rate_among_baseline_failures": fmt_float(rescued / baseline_count if baseline_count else 0.0),
        "induction_rate_among_baseline_successes": fmt_float(induced / baseline_success_count if baseline_success_count else 0.0),
        "mean_detector_event_count": fmt_float(statistics.fmean(detector_event_counts[:num_pairs])),
        "source_data_dir": str(data_dir),
    }


def summarize_qec3v5_groups(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for group_by in ("basis", "cycles", "distance", "all"):
        values = ["all"] if group_by == "all" else sorted({str(row[group_by]) for row in rows})
        for value in values:
            group = rows if group_by == "all" else [row for row in rows if str(row[group_by]) == value]
            deltas = [float(row["paired_delta_lfr"]) for row in group]
            strongest = min(group, key=lambda row: float(row["paired_delta_lfr"])) if group else {}
            out.append(
                {
                    "source_dataset": "google_qec3v5_2022",
                    "group_by": group_by,
                    "group_value": value,
                    "num_conditions": len(group),
                    "total_pairs": sum(parse_int(row["num_pairs"]) for row in group),
                    "mean_baseline_lfr": fmt_float(statistics.fmean(float(row["baseline_lfr"]) for row in group) if group else 0.0),
                    "mean_intervened_lfr": fmt_float(statistics.fmean(float(row["intervened_lfr"]) for row in group) if group else 0.0),
                    "mean_paired_delta_lfr": fmt_float(statistics.fmean(deltas) if deltas else 0.0),
                    "min_paired_delta_lfr": fmt_float(min(deltas) if deltas else 0.0),
                    "max_paired_delta_lfr": fmt_float(max(deltas) if deltas else 0.0),
                    "strongest_condition": strongest.get("workload_id", ""),
                    "strongest_paired_delta_lfr": strongest.get("paired_delta_lfr", "0.000000"),
                }
            )
    return out


def parse_qec3v5_dir_name(name: str) -> dict[str, Any] | None:
    match = QEC3V5_RECORD_RE.fullmatch(name)
    if match is None:
        return None
    out: dict[str, Any] = match.groupdict()
    for field in ("distance", "rounds", "center_row", "center_col"):
        out[field] = int(out[field])
    out["sort_key"] = (out["code"], out["basis"], out["distance"], out["rounds"], out["center_row"], out["center_col"])
    return out


def read_properties(path: Path) -> dict[str, Any]:
    text = path.read_text()
    return yaml.safe_load(text) if yaml is not None else parse_simple_yaml(text)


def read_01(path: Path, *, max_shots: int | None) -> list[bool]:
    values = []
    with path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            values.append(stripped == "1")
            if max_shots is not None and len(values) >= max_shots:
                break
    return values


def read_detector_event_counts(
    path: Path,
    *,
    detector_count: int,
    max_shots: int | None,
) -> list[int]:
    import stim

    samples = stim.read_shot_data_file(path=path, format="b8", num_measurements=detector_count)
    if max_shots is not None:
        samples = samples[:max_shots]
    return [len(detector_indices(row)) for row in samples]
