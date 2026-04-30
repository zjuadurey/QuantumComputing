# P5b/P5c: Layered Events And Surface-Code Pilot

## P5b Purpose

P5b improves the noise-side intervention semantics without trying to implement
perfect physical-event replay.

Instead of thinning only `detector_events`, P5b creates a layered event record:

```text
data_events
measurement_events
idle_events
runtime_events
```

Noise interventions operate on the matching event layer, then derive the
detector-event summary from the remaining layer events. Runtime and policy
interventions preserve the same layered event record.

Run:

```bash
conda run -n failureops --no-capture-output python scripts/23_run_p5b_layered_events.py \
  --config configs/p5b_layered_events.yaml
```

Outputs:

```text
data/results/p5b_baseline_runs.csv
data/results/p5b_intervened_runs.csv
data/results/p5b_pairing_validation.csv
data/results/p5b_paired_effects.csv
data/results/p5b_manifest.json
```

## P5c Purpose

P5c is a small rotated surface-code memory pilot.

The goal is not to run a benchmark suite. The goal is to show that the P4/P5
paired engine is not tied to the repetition-code backend.

P5c uses:

```text
Stim rotated surface-code memory circuit
PyMatching decoding
P3/P5 runtime service fields
P5b layered event records
P4 pairing validation
P4 paired treatment-effect metrics
```

Run:

```bash
conda run -n failureops --no-capture-output python scripts/24_run_p5c_surface_pilot.py \
  --config configs/p5c_surface_pilot.yaml
```

Outputs:

```text
data/results/p5c_surface_baseline_runs.csv
data/results/p5c_surface_intervened_runs.csv
data/results/p5c_surface_pairing_validation.csv
data/results/p5c_surface_paired_effects.csv
data/results/p5c_surface_manifest.json
```

## Scope

P5b/P5c still do not implement:

```text
exact physical-event replay
surface-code benchmark sweeps
real decoder-runtime traces
real decoder scheduling
formal hypothesis testing
```

The intended contribution is a more realistic input surface for the already
validated paired-counterfactual engine.

