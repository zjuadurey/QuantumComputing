"""Unified P5/P6 experiment runner and cross-mode summaries."""

from __future__ import annotations

import itertools
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from failureops.data_model import (
    P4_BASELINE_FIELDS,
    P4_INTERVENTION_FIELDS,
    P4_PAIRED_EFFECT_FIELDS,
    P4_PAIRING_VALIDATION_FIELDS,
)
from failureops.event_layers import NOISE_LAYER_MAP, apply_layered_noise_intervention, attach_event_layers
from failureops.experiment_config import parse_simple_yaml, yaml
from failureops.io_utils import fmt_float, write_csv_rows
from failureops.manifest import write_manifest
from failureops.paired_metrics import summarize_paired_effects
from failureops.pairing import build_p4_intervention_row, event_record_hash, record_hash, validate_intervention_rows
from failureops.runtime_service import P3_INTERVENTIONS, apply_p3_intervention, generate_p3_runs
from failureops.runtime_trace import apply_trace_to_baseline_rows, export_runtime_trace_rows, load_runtime_trace, write_trace_rows
from failureops.surface_backend import generate_surface_code_runs


@dataclass(frozen=True)
class ModeSpec:
    mode_id: str
    backend: str
    runtime_source: str
    event_model: str
    workloads: tuple[str, ...]
    stress_levels: tuple[str, ...]


def load_p6_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        raw = yaml.safe_load(handle) if yaml is not None else parse_simple_yaml(handle.read())
    config = {
        "experiment_id": raw.get("experiment_id", "p6_cross_mode"),
        "mode_specs": list(raw.get("mode_specs", [])),
        "num_seeds": int(raw.get("num_seeds", 2)),
        "num_shots_per_seed": int(raw.get("num_shots_per_seed", 100)),
        "seed_start": int(raw.get("seed_start", 42)),
        "seed_stride": int(raw.get("seed_stride", 10000)),
        "interventions": list(raw.get("interventions", P3_INTERVENTIONS)),
        "bootstrap": dict(raw.get("bootstrap", {})),
        "outputs": dict(raw.get("outputs", {})),
    }
    config["bootstrap"].setdefault("num_resamples", 500)
    config["bootstrap"].setdefault("seed", 2026)
    config["outputs"].setdefault("dir", "data/results/p6")
    config["outputs"].setdefault("mode_summary", "data/results/p6_mode_summary.csv")
    config["outputs"].setdefault("intervention_stability", "data/results/p6_intervention_stability.csv")
    config["outputs"].setdefault("rank_stability", "data/results/p6_rank_stability.csv")
    if not config["mode_specs"]:
        raise ValueError("P6 config must provide mode_specs")
    return config


def parse_mode_spec(value: str) -> ModeSpec:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 6:
        raise ValueError(
            "mode spec must be mode_id,backend,runtime_source,event_model,workloads,stress_levels"
        )
    mode_id, backend, runtime_source, event_model, workloads, stress_levels = parts
    if backend not in {"repetition", "surface"}:
        raise ValueError(f"unsupported backend for {mode_id}: {backend}")
    if runtime_source not in {"proxy", "trace"}:
        raise ValueError(f"unsupported runtime_source for {mode_id}: {runtime_source}")
    if event_model not in {"detector", "layered"}:
        raise ValueError(f"unsupported event_model for {mode_id}: {event_model}")
    return ModeSpec(
        mode_id=mode_id,
        backend=backend,
        runtime_source=runtime_source,
        event_model=event_model,
        workloads=tuple(item for item in workloads.split("|") if item),
        stress_levels=tuple(item for item in stress_levels.split("|") if item),
    )


def run_p6_config(config: dict[str, Any], *, command: list[str]) -> dict[str, list[dict[str, object]]]:
    modes = [parse_mode_spec(item) for item in config["mode_specs"]]
    output_dir = Path(config["outputs"]["dir"])
    mode_summaries = []
    all_effects = []
    mode_rankings: dict[str, list[str]] = {}
    for mode in modes:
        result = run_mode(config, mode, output_dir=output_dir, command=command)
        mode_summaries.append(result["summary"])
        all_effects.extend(result["effects_with_mode"])
        mode_rankings[mode.mode_id] = result["ranking"]

    intervention_stability = summarize_intervention_stability(all_effects)
    rank_stability = [summarize_mode_rank_stability(mode_rankings)]
    return {
        "mode_summary": mode_summaries,
        "intervention_stability": intervention_stability,
        "rank_stability": rank_stability,
    }


def run_mode(
    config: dict[str, Any],
    mode: ModeSpec,
    *,
    output_dir: Path,
    command: list[str],
) -> dict[str, Any]:
    baseline_rows = []
    for workload_id in mode.workloads:
        for stress_level in mode.stress_levels:
            for seed_index in range(config["num_seeds"]):
                seed = config["seed_start"] + seed_index * config["seed_stride"]
                run_id = f"{config['experiment_id']}_{mode.mode_id}_{workload_id}_{stress_level}_{seed_index}"
                if mode.backend == "surface":
                    run_rows = generate_surface_code_runs(
                        workload_id=workload_id,
                        stress_level=stress_level,
                        num_shots=config["num_shots_per_seed"],
                        seed=seed,
                        run_id=run_id,
                    )
                else:
                    run_rows = generate_p3_runs(
                        workload_id=workload_id,
                        stress_level=stress_level,
                        num_shots=config["num_shots_per_seed"],
                        seed=seed,
                        run_id=run_id,
                    )
                baseline_rows.extend(run_rows)

    trace_stats = {"matched_traces": 0, "missing_traces": 0}
    trace_path = output_dir / f"{mode.mode_id}_runtime_trace.csv"
    if mode.runtime_source == "trace":
        trace_rows = export_runtime_trace_rows(
            baseline_rows,
            trace_source=f"p6_proxy_export:{mode.mode_id}",
        )
        write_trace_rows(trace_path, trace_rows, [
            "trace_id",
            "trace_source",
            "run_id",
            "workload_id",
            "stress_level",
            "shot_id",
            "seed",
            "decoder_arrival_time",
            "decoder_start_time",
            "decoder_finish_time",
            "decoder_latency",
            "decoder_queue_wait",
            "decoder_service_time",
            "decoder_backlog",
            "decoder_timeout",
            "decoder_deadline_miss",
            "decoder_queue_overflow",
            "runtime_stall_rounds",
            "idle_exposure",
            "runtime_idle_flip",
        ])
        baseline_rows, trace_stats = apply_trace_to_baseline_rows(
            baseline_rows,
            load_runtime_trace(trace_path),
            missing_policy="error",
        )

    if mode.event_model == "layered":
        baseline_rows = [attach_event_layers(row) for row in baseline_rows]
    for row in baseline_rows:
        row["event_record_hash"] = event_record_hash(row)
        row["record_hash"] = record_hash(row)

    intervention_rows = []
    for baseline in baseline_rows:
        for intervention in config["interventions"]:
            if mode.event_model == "layered" and intervention in NOISE_LAYER_MAP:
                intervened = apply_layered_noise_intervention(baseline, intervention)
            else:
                intervened = apply_p3_intervention(baseline, intervention)
            intervention_rows.append(build_p4_intervention_row(baseline, intervened, intervention))

    validation_rows = validate_intervention_rows(intervention_rows)
    effect_rows = summarize_paired_effects(
        intervention_rows,
        num_bootstrap=int(config["bootstrap"]["num_resamples"]),
        bootstrap_seed=int(config["bootstrap"]["seed"]),
    )
    mode_effects = aggregate_mode_effects(effect_rows, mode.mode_id)

    paths = {
        "baseline": str(output_dir / f"{mode.mode_id}_baseline_runs.csv"),
        "interventions": str(output_dir / f"{mode.mode_id}_intervened_runs.csv"),
        "validation": str(output_dir / f"{mode.mode_id}_pairing_validation.csv"),
        "effects": str(output_dir / f"{mode.mode_id}_paired_effects.csv"),
        "manifest": str(output_dir / f"{mode.mode_id}_manifest.json"),
    }
    if mode.runtime_source == "trace":
        paths["runtime_trace"] = str(trace_path)
    write_csv_rows(paths["baseline"], baseline_rows, P4_BASELINE_FIELDS)
    write_csv_rows(paths["interventions"], intervention_rows, P4_INTERVENTION_FIELDS)
    write_csv_rows(paths["validation"], validation_rows, P4_PAIRING_VALIDATION_FIELDS)
    write_csv_rows(paths["effects"], effect_rows, P4_PAIRED_EFFECT_FIELDS)
    row_counts = {
        "baseline": len(baseline_rows),
        "interventions": len(intervention_rows),
        "validation": len(validation_rows),
        "effects": len(effect_rows),
        **trace_stats,
    }
    write_manifest(
        paths["manifest"],
        config={**config, "mode": mode.__dict__},
        command=command,
        outputs=paths,
        row_counts=row_counts,
    )

    invalid_pairs = sum(int(row["invalid_pairs"]) for row in validation_rows)
    max_violation = max((float(row["violation_rate"]) for row in validation_rows), default=0.0)
    strongest = min(mode_effects, key=lambda row: float(row["paired_delta_lfr"])) if mode_effects else {}
    ranking = [
        str(row["intervention"])
        for row in sorted(mode_effects, key=lambda row: float(row["paired_delta_lfr"]))
    ]
    summary = {
        "mode_id": mode.mode_id,
        "backend": mode.backend,
        "runtime_source": mode.runtime_source,
        "event_model": mode.event_model,
        "workloads": "|".join(mode.workloads),
        "stress_levels": "|".join(mode.stress_levels),
        "baseline_rows": len(baseline_rows),
        "intervention_rows": len(intervention_rows),
        "validation_rows": len(validation_rows),
        "effect_rows": len(effect_rows),
        "invalid_pairs": invalid_pairs,
        "max_violation_rate": fmt_float(max_violation),
        "strongest_intervention": strongest.get("intervention", ""),
        "strongest_paired_delta_lfr": strongest.get("paired_delta_lfr", "0.000000"),
        "strongest_intervention_class": strongest.get("intervention_class", ""),
    }
    return {
        "summary": summary,
        "effects_with_mode": mode_effects,
        "ranking": ranking,
    }


def aggregate_mode_effects(effect_rows: list[dict[str, object]], mode_id: str) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in effect_rows:
        grouped.setdefault(str(row["intervention"]), []).append(row)
    out = []
    for intervention, rows in grouped.items():
        deltas = [float(row["paired_delta_lfr"]) for row in rows]
        net_rates = [float(row["net_rescue_rate"]) for row in rows]
        out.append(
            {
                "mode_id": mode_id,
                "intervention": intervention,
                "intervention_class": str(rows[0].get("intervention_class", "unknown")),
                "paired_delta_lfr": fmt_float(statistics.fmean(deltas)),
                "net_rescue_rate": fmt_float(statistics.fmean(net_rates)),
            }
        )
    return out


def summarize_intervention_stability(effects: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in effects:
        grouped.setdefault(str(row["intervention"]), []).append(row)
    out = []
    for intervention, rows in grouped.items():
        deltas = [float(row["paired_delta_lfr"]) for row in rows]
        net_rates = [float(row["net_rescue_rate"]) for row in rows]
        top = min(rows, key=lambda row: float(row["paired_delta_lfr"]))
        out.append(
            {
                "intervention": intervention,
                "num_modes": len(set(str(row["mode_id"]) for row in rows)),
                "mode_ids": "|".join(sorted(set(str(row["mode_id"]) for row in rows))),
                "mean_paired_delta_lfr": fmt_float(statistics.fmean(deltas)),
                "min_paired_delta_lfr": fmt_float(min(deltas)),
                "max_paired_delta_lfr": fmt_float(max(deltas)),
                "mean_net_rescue_rate": fmt_float(statistics.fmean(net_rates)),
                "top_mode": top["mode_id"],
                "top_mode_paired_delta_lfr": top["paired_delta_lfr"],
            }
        )
    out.sort(key=lambda row: (float(row["mean_paired_delta_lfr"]), row["intervention"]))
    return out


def summarize_mode_rank_stability(mode_rankings: dict[str, list[str]]) -> dict[str, object]:
    overlaps = []
    distances = []
    serialized = {}
    for mode_id, ranking in mode_rankings.items():
        serialized[mode_id] = ranking[:10]
    for left, right in itertools.combinations(mode_rankings.values(), 2):
        overlaps.append(top_k_overlap(left, right, top_k=3))
        distances.append(pairwise_rank_distance(left, right))
    return {
        "num_modes": len(mode_rankings),
        "mean_top3_overlap": fmt_float(statistics.fmean(overlaps) if overlaps else 1.0),
        "mean_pairwise_rank_distance": fmt_float(statistics.fmean(distances) if distances else 0.0),
        "mode_rankings": json.dumps(serialized, sort_keys=True, separators=(",", ":")),
    }


def top_k_overlap(left: list[str], right: list[str], *, top_k: int) -> float:
    left_set = set(left[:top_k])
    right_set = set(right[:top_k])
    return len(left_set & right_set) / max(1, top_k)


def pairwise_rank_distance(left: list[str], right: list[str]) -> float:
    items = list(dict.fromkeys([*left, *right]))
    left_rank = {item: index for index, item in enumerate(left)}
    right_rank = {item: index for index, item in enumerate(right)}
    max_rank = len(items)
    total = 0.0
    for item in items:
        total += abs(left_rank.get(item, max_rank) - right_rank.get(item, max_rank))
    return total / max(1, len(items))
