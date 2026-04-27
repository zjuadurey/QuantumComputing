"""Runtime trace import/export helpers for P5a.

P5a keeps the existing runtime-service proxy but lets baseline executions be
overridden by external decoder-runtime observations. A trace can come from a CSV
or JSON file and is keyed by the paired execution identity.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from failureops.io_utils import ensure_parent_dir, fmt_float, parse_bool, parse_float, parse_int
from failureops.runtime_service import recompute_p3_failure_behavior

TRACE_KEY_FIELDS = ("run_id", "workload_id", "stress_level", "seed", "shot_id")

TRACE_NUMERIC_FIELDS = (
    "decoder_arrival_time",
    "decoder_start_time",
    "decoder_finish_time",
    "decoder_latency",
    "decoder_queue_wait",
    "decoder_service_time",
    "decoder_backlog",
    "runtime_stall_rounds",
    "idle_exposure",
)

TRACE_BOOL_FIELDS = (
    "decoder_timeout",
    "decoder_deadline_miss",
    "decoder_queue_overflow",
    "runtime_idle_flip",
)


def trace_key(row: dict[str, object], *, ignore_run_id: bool = False) -> tuple[str, ...]:
    fields = TRACE_KEY_FIELDS[1:] if ignore_run_id else TRACE_KEY_FIELDS
    return tuple(str(row.get(field, "")) for field in fields)


def load_runtime_trace(path: str | Path, *, ignore_run_id: bool = False) -> dict[tuple[str, ...], dict[str, object]]:
    rows = read_trace_rows(path)
    out = {}
    for row in rows:
        normalized = normalize_trace_row(row)
        out[trace_key(normalized, ignore_run_id=ignore_run_id)] = normalized
    return out


def read_trace_rows(path: str | Path) -> list[dict[str, object]]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        with path.open() as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            payload = payload.get("runtime_traces", payload.get("rows", []))
        if not isinstance(payload, list):
            raise ValueError(f"JSON trace {path} must contain a list or runtime_traces list")
        return [dict(row) for row in payload]
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_trace_rows(path: str | Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path = Path(path)
    ensure_parent_dir(path)
    if path.suffix.lower() == ".json":
        with path.open("w") as handle:
            json.dump({"runtime_traces": rows}, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_trace_row(row: dict[str, object]) -> dict[str, object]:
    out = dict(row)
    for field in ("seed", "shot_id"):
        if field in out:
            out[field] = parse_int(out[field])
    fill_derived_trace_times(out)
    for field in TRACE_NUMERIC_FIELDS:
        if field in out:
            out[field] = fmt_float(parse_float(out[field]))
    for field in TRACE_BOOL_FIELDS:
        if field in out:
            out[field] = parse_bool(out[field])
    return out


def fill_derived_trace_times(row: dict[str, object]) -> None:
    arrival = parse_float(row.get("decoder_arrival_time", 0.0))
    start = parse_float(row.get("decoder_start_time", arrival))
    finish = parse_float(row.get("decoder_finish_time", start))
    if "decoder_queue_wait" not in row or row.get("decoder_queue_wait") == "":
        row["decoder_queue_wait"] = max(0.0, start - arrival)
    if "decoder_service_time" not in row or row.get("decoder_service_time") == "":
        row["decoder_service_time"] = max(0.0, finish - start)
    if "decoder_latency" not in row or row.get("decoder_latency") == "":
        row["decoder_latency"] = max(0.0, finish - arrival)
    if "runtime_stall_rounds" not in row or row.get("runtime_stall_rounds") == "":
        row["runtime_stall_rounds"] = 0.0
    if "decoder_backlog" not in row or row.get("decoder_backlog") == "":
        row["decoder_backlog"] = 0.0
    if "idle_exposure" not in row or row.get("idle_exposure") == "":
        latency = parse_float(row.get("decoder_latency", 0.0))
        backlog = parse_float(row.get("decoder_backlog", 0.0))
        row["idle_exposure"] = 0.08 + 0.75 * latency + 0.10 * backlog
    for field in TRACE_BOOL_FIELDS:
        row.setdefault(field, False)


def apply_runtime_trace_to_record(record: dict[str, object], trace: dict[str, object]) -> dict[str, object]:
    out = dict(record)
    for field in TRACE_NUMERIC_FIELDS:
        if field in trace:
            if field == "decoder_queue_wait":
                continue
            if field == "decoder_service_time":
                continue
            out[field] = trace[field]
    for field in TRACE_BOOL_FIELDS:
        if field in trace:
            out[field] = trace[field]
    return recompute_p3_failure_behavior(out)


def export_runtime_trace_rows(
    baseline_rows: list[dict[str, object]],
    *,
    trace_source: str = "failureops_proxy_export",
) -> list[dict[str, object]]:
    rows = []
    for baseline in baseline_rows:
        arrival = parse_float(baseline.get("decoder_arrival_time", 0.0))
        start = parse_float(baseline.get("decoder_start_time", arrival))
        finish = parse_float(baseline.get("decoder_finish_time", start))
        rows.append(
            {
                "trace_id": f"{baseline.get('run_id')}|{baseline.get('seed')}|{baseline.get('shot_id')}",
                "trace_source": trace_source,
                "run_id": baseline.get("run_id", ""),
                "workload_id": baseline.get("workload_id", ""),
                "stress_level": baseline.get("stress_level", ""),
                "shot_id": baseline.get("shot_id", ""),
                "seed": baseline.get("seed", ""),
                "decoder_arrival_time": fmt_float(arrival),
                "decoder_start_time": fmt_float(start),
                "decoder_finish_time": fmt_float(finish),
                "decoder_latency": fmt_float(parse_float(baseline.get("decoder_latency", finish - arrival))),
                "decoder_queue_wait": fmt_float(max(0.0, start - arrival)),
                "decoder_service_time": fmt_float(max(0.0, finish - start)),
                "decoder_backlog": baseline.get("decoder_backlog", "0.000000"),
                "decoder_timeout": baseline.get("decoder_timeout", False),
                "decoder_deadline_miss": baseline.get("decoder_deadline_miss", False),
                "decoder_queue_overflow": baseline.get("decoder_queue_overflow", False),
                "runtime_stall_rounds": baseline.get("runtime_stall_rounds", "0.000000"),
                "idle_exposure": baseline.get("idle_exposure", "0.000000"),
                "runtime_idle_flip": baseline.get("runtime_idle_flip", False),
            }
        )
    return rows


def apply_trace_to_baseline_rows(
    baseline_rows: list[dict[str, object]],
    traces: dict[tuple[str, ...], dict[str, object]],
    *,
    missing_policy: str = "error",
    ignore_run_id: bool = False,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    out = []
    matched = 0
    missing = 0
    for row in baseline_rows:
        key = trace_key(row, ignore_run_id=ignore_run_id)
        trace = traces.get(key)
        if trace is None:
            missing += 1
            if missing_policy == "error":
                raise ValueError(f"missing runtime trace for key={key}")
            out.append(dict(row))
            continue
        matched += 1
        out.append(apply_runtime_trace_to_record(row, trace))
    return out, {"matched_traces": matched, "missing_traces": missing}
