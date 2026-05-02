#!/usr/bin/env python
"""Measure real decoder runtime by replaying P7 detector records through PyMatching."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from failureops.data_model import (
    P8_DECODER_RUNTIME_BATCH_FIELDS,
    P8_DECODER_RUNTIME_SUMMARY_FIELDS,
    P8_DECODER_RUNTIME_TRACE_FIELDS,
)
from failureops.google_rl_qec_adapter import (
    code_family_from_experiment,
    detector_indices,
    read_b8,
    read_circuit,
    read_decoder_predictions,
    read_metadata,
)
from failureops.io_utils import ensure_parent_dir, fmt_float, parse_int, write_csv_rows
from failureops.manifest import write_manifest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/failureops-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=(
            "data/raw/google_rl_qec/google_reinforcement_learning_qec/"
            "surface_code_traditional_calibration/Z/r010"
        ),
    )
    parser.add_argument(
        "--decoder-pathway",
        default="correlated_matching_decoder_with_si1000_prior",
    )
    parser.add_argument("--max-shots", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument(
        "--deadline-us",
        type=float,
        default=0.0,
        help="Optional per-shot deadline in microseconds. Zero disables deadline misses.",
    )
    parser.add_argument("--run-id", default="p8_decoder_runtime_replay")
    parser.add_argument("--trace-output", default="data/results/p8_decoder_runtime_trace.csv")
    parser.add_argument("--batch-output", default="data/results/p8_decoder_runtime_batches.csv")
    parser.add_argument("--summary-output", default="data/results/p8_decoder_runtime_summary.csv")
    parser.add_argument("--manifest-output", default="data/results/p8_decoder_runtime_manifest.json")
    parser.add_argument("--plot-output", default="figures/p8_decoder_latency_vs_syndrome_weight.png")
    args = parser.parse_args()

    if args.max_shots <= 0:
        raise ValueError("--max-shots must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.warmup_batches < 0:
        raise ValueError("--warmup-batches must be non-negative")

    result = measure_decoder_runtime(
        data_dir=Path(args.data_dir),
        decoder_pathway=args.decoder_pathway,
        max_shots=args.max_shots,
        batch_size=args.batch_size,
        repeats=args.repeats,
        warmup_batches=args.warmup_batches,
        deadline_us=args.deadline_us,
        run_id=args.run_id,
    )

    write_csv_rows(args.trace_output, result["trace_rows"], P8_DECODER_RUNTIME_TRACE_FIELDS)
    write_csv_rows(args.batch_output, result["batch_rows"], P8_DECODER_RUNTIME_BATCH_FIELDS)
    write_csv_rows(args.summary_output, result["summary_rows"], P8_DECODER_RUNTIME_SUMMARY_FIELDS)
    plot_latency_vs_syndrome_weight(result["trace_rows"], args.plot_output)
    write_manifest(
        args.manifest_output,
        config={
            "experiment_id": args.run_id,
            "data_dir": args.data_dir,
            "decoder_pathway": args.decoder_pathway,
            "max_shots": args.max_shots,
            "batch_size": args.batch_size,
            "repeats": args.repeats,
            "warmup_batches": args.warmup_batches,
            "deadline_us": args.deadline_us,
            "measurement_environment": measurement_environment(),
        },
        command=sys.argv,
        outputs={
            "trace": args.trace_output,
            "batches": args.batch_output,
            "summary": args.summary_output,
            "plot": args.plot_output,
        },
        row_counts={
            "trace": len(result["trace_rows"]),
            "batches": len(result["batch_rows"]),
            "summary": len(result["summary_rows"]),
        },
    )
    summary = result["summary_rows"][0]
    print(f"wrote {len(result['trace_rows'])} measured decoder trace rows to {args.trace_output}")
    print(f"wrote {len(result['batch_rows'])} measured batch rows to {args.batch_output}")
    print(f"wrote P8 runtime summary to {args.summary_output}")
    print(f"wrote P8 latency plot to {args.plot_output}")
    print(f"mean per-shot decode time: {summary['mean_per_shot_decode_time_us']} us")
    print(f"p95 per-shot decode time: {summary['p95_per_shot_decode_time_us']} us")
    print(f"prediction mismatch rate vs stored decoder output: {summary['prediction_mismatch_rate']}")


def measure_decoder_runtime(
    *,
    data_dir: Path,
    decoder_pathway: str,
    max_shots: int,
    batch_size: int,
    repeats: int,
    warmup_batches: int,
    deadline_us: float,
    run_id: str,
) -> dict[str, list[dict[str, object]]]:
    import pymatching
    import stim

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
    stored_predictions = read_decoder_predictions(
        data_dir,
        decoder_pathway,
        bits_per_shot=circuit.num_observables,
        max_shots=max_shots,
    )
    dem_path = data_dir / "decoding_results" / decoder_pathway / "error_model.dem"

    build_start = time.perf_counter_ns()
    matching = pymatching.Matching.from_detector_error_model(stim.DetectorErrorModel(dem_path.read_text()))
    build_elapsed_ns = time.perf_counter_ns() - build_start

    num_shots = len(detector_samples)
    if not (len(actual_flips) == len(stored_predictions) == num_shots):
        raise ValueError("detector, observable, and stored prediction files have mismatched shot counts")
    if num_shots == 0:
        raise ValueError("no detector records loaded")

    for batch in iter_batches(detector_samples, batch_size, limit=warmup_batches):
        matching.decode_batch(batch)

    workload_id = f"{metadata['experiment_name']}_{metadata['basis']}_{metadata['cycle_dir']}"
    exemplar = {
        "run_id": run_id,
        "workload_id": workload_id,
        "stress_level": "measured_decoder_replay",
        "experiment_name": metadata["experiment_name"],
        "basis": metadata["basis"],
        "cycles": parse_int(metadata["rounds"]),
        "cycle_dir": metadata["cycle_dir"],
        "source_data_dir": str(data_dir),
        "decoder_pathway": decoder_pathway,
    }

    detector_event_counts = [
        sum(1 for value in detector_samples[shot_id] if bool(value))
        for shot_id in range(num_shots)
    ]
    trace_rows = []
    batch_rows = []
    per_shot_times_us = []
    mismatch_count = 0
    logical_failure_count = 0
    deadline_miss_count = 0
    total_decode_us = 0.0
    clock_us = 0.0
    batch_id = 0

    for repeat_id in range(repeats):
        for start in range(0, num_shots, batch_size):
            stop = min(start + batch_size, num_shots)
            batch = detector_samples[start:stop]
            decode_start = time.perf_counter_ns()
            predictions = matching.decode_batch(batch)
            elapsed_ns = time.perf_counter_ns() - decode_start
            elapsed_us = elapsed_ns / 1000.0
            batch_len = stop - start
            per_shot_us = elapsed_us / batch_len
            total_decode_us += elapsed_us
            per_shot_times_us.extend([per_shot_us] * batch_len)

            batch_counts = detector_event_counts[start:stop]
            batch_rows.append(
                {
                    **exemplar,
                    "repeat_id": repeat_id,
                    "batch_id": batch_id,
                    "batch_size": batch_len,
                    "first_shot_id": start,
                    "last_shot_id": stop - 1,
                    "mean_detector_event_count": fmt_float(statistics.fmean(batch_counts)),
                    "max_detector_event_count": max(batch_counts),
                    "decode_wall_time_ns": elapsed_ns,
                    "decode_wall_time_us": fmt_float(elapsed_us),
                    "per_shot_decode_time_us": fmt_float(per_shot_us),
                    "throughput_shots_per_second": fmt_float(batch_len / (elapsed_us / 1_000_000.0)),
                }
            )

            shot_start_time = clock_us / 1_000_000.0
            shot_finish_time = (clock_us + per_shot_us) / 1_000_000.0
            deadline_miss = deadline_us > 0.0 and per_shot_us > deadline_us
            for local_index, shot_id in enumerate(range(start, stop)):
                measured_prediction = prediction_bit(predictions, local_index)
                stored_prediction = bool(stored_predictions[shot_id][0])
                observable_flip = bool(actual_flips[shot_id][0])
                prediction_matches = measured_prediction == stored_prediction
                measured_logical_failure = measured_prediction != observable_flip
                mismatch_count += int(not prediction_matches)
                logical_failure_count += int(measured_logical_failure)
                deadline_miss_count += int(deadline_miss)
                detector_events = detector_indices(detector_samples[shot_id])
                trace_rows.append(
                    {
                        **exemplar,
                        "trace_id": f"{run_id}|{repeat_id}|{shot_id}",
                        "trace_source": "p8_measured_pymatching_replay",
                        "repeat_id": repeat_id,
                        "batch_id": batch_id,
                        "batch_size": batch_len,
                        "shot_id": shot_id,
                        "seed": 0,
                        "detector_count": circuit.num_detectors,
                        "detector_event_count": len(detector_events),
                        "decoder_arrival_time": fmt_seconds(shot_start_time),
                        "decoder_start_time": fmt_seconds(shot_start_time),
                        "decoder_finish_time": fmt_seconds(shot_finish_time),
                        "decoder_latency": fmt_seconds(per_shot_us / 1_000_000.0),
                        "decoder_queue_wait": fmt_seconds(0.0),
                        "decoder_service_time": fmt_seconds(per_shot_us / 1_000_000.0),
                        "decoder_backlog": fmt_float(0.0),
                        "decoder_timeout": False,
                        "decoder_deadline_miss": deadline_miss,
                        "decoder_queue_overflow": False,
                        "runtime_stall_rounds": fmt_float(0.0),
                        "idle_exposure": fmt_float(0.0),
                        "runtime_idle_flip": False,
                        "observable_flip": observable_flip,
                        "measured_decoder_prediction": measured_prediction,
                        "stored_decoder_prediction": stored_prediction,
                        "prediction_matches_stored": prediction_matches,
                        "measured_logical_failure": measured_logical_failure,
                    }
                )
            clock_us += elapsed_us
            batch_id += 1

    total_trace_rows = len(trace_rows)
    summary_rows = [
        {
            **exemplar,
            "num_shots": num_shots,
            "batch_size": batch_size,
            "num_batches": len(batch_rows),
            "repeats": repeats,
            "matching_build_time_us": fmt_float(build_elapsed_ns / 1000.0),
            "total_decode_wall_time_us": fmt_float(total_decode_us),
            "mean_per_shot_decode_time_us": fmt_float(statistics.fmean(per_shot_times_us)),
            "median_per_shot_decode_time_us": fmt_float(statistics.median(per_shot_times_us)),
            "p95_per_shot_decode_time_us": fmt_float(percentile(per_shot_times_us, 0.95)),
            "max_per_shot_decode_time_us": fmt_float(max(per_shot_times_us)),
            "throughput_shots_per_second": fmt_float(total_trace_rows / (total_decode_us / 1_000_000.0)),
            "prediction_mismatch_count": mismatch_count,
            "prediction_mismatch_rate": fmt_float(mismatch_count / total_trace_rows),
            "measured_logical_failure_rate": fmt_float(logical_failure_count / total_trace_rows),
            "deadline_us": fmt_float(deadline_us),
            "deadline_miss_rate": fmt_float(deadline_miss_count / total_trace_rows),
        }
    ]
    summary_rows[0]["code_family"] = code_family_from_experiment(str(metadata["experiment_name"]))
    return {
        "trace_rows": trace_rows,
        "batch_rows": batch_rows,
        "summary_rows": summary_rows,
    }


def iter_batches(samples: Any, batch_size: int, *, limit: int):
    if limit == 0:
        return
    yielded = 0
    for start in range(0, len(samples), batch_size):
        yield samples[start:min(start + batch_size, len(samples))]
        yielded += 1
        if yielded >= limit:
            return


def prediction_bit(predictions: Any, index: int) -> bool:
    value = predictions[index]
    try:
        return bool(value[0])
    except (TypeError, IndexError):
        return bool(value)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(quantile * (len(ordered) - 1))))
    return ordered[index]


def fmt_seconds(value: float) -> str:
    return f"{value:.9f}"


def plot_latency_vs_syndrome_weight(rows: list[dict[str, object]], output: str) -> None:
    ensure_parent_dir(output)
    if not rows:
        return
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(int(row["detector_event_count"]), []).append(
            float(row["decoder_service_time"]) * 1_000_000.0
        )
    xs = sorted(grouped)
    means = [statistics.fmean(grouped[x]) for x in xs]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.scatter(
        [int(row["detector_event_count"]) for row in rows],
        [float(row["decoder_service_time"]) * 1_000_000.0 for row in rows],
        s=8,
        alpha=0.25,
        color="#4b78a8",
        label="shot trace",
    )
    ax.plot(xs, means, color="#b55335", linewidth=1.8, label="mean by detector count")
    ax.set_title("P8 measured decoder runtime replay")
    ax.set_xlabel("Detector-event count")
    ax.set_ylabel("Measured per-shot decode time (us)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def measurement_environment() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
    }


if __name__ == "__main__":
    main()
