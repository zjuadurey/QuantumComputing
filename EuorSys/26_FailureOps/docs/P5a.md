# P5a: Runtime Trace Mode

## Purpose

P5a is the first backend-realism step after P4.

It does not add a surface-code backend or a real decoder scheduler. Instead, it
keeps the existing runtime-service proxy but makes decoder-runtime observations
an external input source. FailureOps can now run over runtime traces loaded from
CSV or JSON.

This moves the project from:

```text
runtime behavior is generated only inside FailureOps
```

to:

```text
runtime behavior can be imported from an external trace
```

## Runtime Trace Schema

Runtime traces are keyed by:

```text
run_id
workload_id
stress_level
seed
shot_id
```

Trace fields include:

```text
decoder_arrival_time
decoder_start_time
decoder_finish_time
decoder_latency
decoder_queue_wait
decoder_service_time
decoder_backlog
decoder_timeout
decoder_deadline_miss
decoder_queue_overflow
runtime_stall_rounds
idle_exposure
runtime_idle_flip
```

CSV and JSON are both supported. JSON files may contain either a top-level list
or a mapping with `runtime_traces`.

## Scripts

Export a synthetic runtime trace from generated baseline rows:

```bash
conda run -n failureops --no-capture-output python scripts/21_export_p5_runtime_trace.py \
  --input data/results/p4_baseline_runs.csv \
  --output data/results/p5a_runtime_trace.csv
```

Run P5a from YAML:

```bash
conda run -n failureops --no-capture-output python scripts/22_run_p5a_trace_mode.py \
  --config configs/p5a_trace_mode.yaml
```

The default config exports a proxy trace if the trace file is missing, then
re-imports it through the same external-trace path.

## Outputs

```text
data/results/p5a_runtime_trace.csv
data/results/p5a_baseline_runs.csv
data/results/p5a_intervened_runs.csv
data/results/p5a_pairing_validation.csv
data/results/p5a_paired_effects.csv
data/results/p5a_manifest.json
```

The baseline/intervention/effect outputs reuse the P4 schema. Pairing validation
therefore still checks that runtime and policy interventions preserve the same
event record.

## What P5a Adds

P5a adds:

```text
external runtime trace reader
runtime trace exporter
trace-driven baseline recomputation
trace stats in manifest row counts
CSV/JSON trace support
```

## Remaining Gaps

P5a does not yet implement:

```text
real runtime trace collection
event-layer physical replay
surface-code backend pilot
trace schema version negotiation
trace timestamp alignment across multiple logical workloads
```

