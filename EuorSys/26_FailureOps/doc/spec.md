# FailureOps Implementation Spec

## 1. Goal

Build a minimal runnable prototype of FailureOps.

The prototype should support intervention-based analysis of logical failure behavior in QEC-protected logical circuit executions.

The first implementation should not attempt to be a complete FTQC simulator.

The goal is to produce a small but reproducible experimental loop:

```text
generate baseline logical execution records
        ↓
apply paired counterfactual interventions
        ↓
compare baseline and intervened outcomes
        ↓
compute sensitivity metrics
        ↓
save CSV summaries and a simple plot
```

P0 should make the FailureOps idea executable before connecting to heavier QEC simulation backends.

---

## 2. P0 Scope

P0 should implement a toy but complete pipeline.

The pipeline should:

1. Generate repeated logical execution records.
2. Simulate physical error events and simple runtime events.
3. Assign logical success or failure outcomes.
4. Apply paired counterfactual interventions.
5. Compare baseline and intervened outcomes.
6. Save results to CSV.
7. Generate a simple summary plot.

P0 may use a toy event model and a transparent toy failure rule.

P0 must include both noise-side and runtime-side factors.

At minimum, P0 should represent:

```text
data errors
measurement errors
idle errors
decoder timeout
decoder delay
idle exposure
```

This is important because FailureOps should not collapse into ordinary noise attribution.

---

## 3. Non-Goals

P0 should not implement:

- a new decoder
- a new QEC code
- threshold estimation
- full surface-code simulation
- full lattice surgery
- real-time decoder scheduling
- distributed FTQC runtime
- quantum circuit compilation
- hardware calibration modeling
- large-scale benchmarking
- full causal inference framework

P0 is only a minimal closed loop for intervention-based failure attribution.

Decoder and runtime factors may appear as intervenable factors, but building a decoder scheduler or a full runtime is not the goal of P0.

---

## 4. Recommended Directory Structure

```text
.
├── AGENTS.md
├── README.md
├── docs/
│   ├── idea.md
│   ├── spec.md
│   └── roadmap.md
├── scripts/
│   ├── 01_generate_runs.py
│   ├── 02_apply_interventions.py
│   ├── 03_analyze_attribution.py
│   └── 04_plot_results.py
├── failureops/
│   ├── __init__.py
│   ├── data_model.py
│   ├── toy_simulator.py
│   ├── interventions.py
│   ├── metrics.py
│   └── io_utils.py
├── data/
│   └── results/
└── figures/
```

The early implementation should prefer explicit scripts and simple modules.

Avoid deep framework abstractions in P0.

---

## 5. Baseline Data Model

Each baseline execution record should represent one logical execution shot.

Minimum fields for `baseline_runs.csv`:

```text
run_id
circuit_id
shot_id
seed
num_rounds
num_operations
data_error_rate
measurement_error_rate
idle_error_rate
decoder_timeout_rate
data_error_count
measurement_error_count
idle_error_count
decoder_timeout
decoder_delay
idle_exposure
error_events
failure_round
failure_region
failure_operation
failure_mode
failure_pattern
logical_failure
```

Recommended meanings:

```text
run_id:
  ID of the experiment batch.

circuit_id:
  Name of the toy logical circuit or workload.

shot_id:
  Repeated execution index.

seed:
  Random seed used for reproducibility.

num_rounds:
  Number of simulated QEC rounds.

num_operations:
  Number of logical operations or operation slots.

data_error_rate:
  Probability or rate parameter for data-error events.

measurement_error_rate:
  Probability or rate parameter for measurement-error events.

idle_error_rate:
  Probability or rate parameter for idle-error events.

decoder_timeout_rate:
  Probability or rate parameter for decoder timeout events.

data_error_count:
  Number of sampled data-error events.

measurement_error_count:
  Number of sampled measurement-error events.

idle_error_count:
  Number of sampled idle-error events.

decoder_timeout:
  Boolean indicating whether decoder timeout occurred.

decoder_delay:
  Toy decoder delay value.

idle_exposure:
  Toy measure of accumulated idle exposure.

error_events:
  Serialized event list or compact JSON summary.

failure_round:
  Round where failure is detected or assigned.

failure_region:
  Region where failure is assigned.

failure_operation:
  Operation index where failure is assigned.

failure_mode:
  Toy failure mode, such as logical_X, logical_Z, mixed, timeout_sensitive, or none.

failure_pattern:
  Coarse failure pattern, such as noise_dominated, idle_correlated,
  timeout_correlated, delay_correlated, burst_like, or no_failure.

logical_failure:
  Boolean logical success/failure outcome.
```

The count fields are redundant with `error_events`, but they are intentionally included to make CSV inspection and attribution analysis simple.

---

## 6. Event Model

P0 may use a toy event model.

Each shot may generate events from categories:

```text
data_error
measurement_error
idle_error
decoder_delay
decoder_timeout
idle_exposure
```

Each event may contain:

```text
event_type
round_id
operation_id
region_id
weight
```

A simple JSON string is acceptable for CSV storage.

Example:

```json
[
  {
    "event_type": "data_error",
    "round_id": 3,
    "operation_id": 2,
    "region_id": "R1",
    "weight": 1.0
  },
  {
    "event_type": "decoder_delay",
    "round_id": 5,
    "operation_id": 3,
    "region_id": "runtime",
    "weight": 0.4
  }
]
```

The event model does not need to be physically complete.

However, every event type should correspond to a real QEC execution or runtime concept.

Acceptable toy concepts include:

```text
data error
measurement error
idle error
syndrome round
logical operation boundary
decoder delay
decoder timeout
idle exposure
decoder backlog
synchronization slack
```

---

## 7. Toy Failure Rule

P0 may use a transparent toy failure rule.

Example rule:

```text
logical_failure = failure_score >= failure_threshold
```

The failure score may be computed as a weighted sum of physical and runtime factors.

Possible weights:

```text
data_error: 1.0
measurement_error: 0.7
idle_error: 0.5
decoder_delay: 0.4
decoder_timeout: 1.2
idle_exposure: 0.3
```

Example threshold:

```text
failure_score >= 3.0
```

The failure rule should be deterministic given a baseline or intervened execution record.

This is important for paired counterfactual evaluation.

The implementation should make clear in comments that this is a toy proxy, not a physically complete QEC model.

The purpose of the toy rule is to make the FailureOps pipeline executable:

```text
baseline record
        ↓
intervention
        ↓
recompute failure behavior
        ↓
measure sensitivity
```

---

## 8. Paired Counterfactual Evaluation

P0 should use paired counterfactual evaluation.

For each baseline shot, interventions should preserve the same:

```text
shot_id
seed
logical workload configuration
runtime configuration
underlying event record when possible
```

The intervention should change only the targeted factor.

The intended question is:

> Would this same execution still fail if one factor were removed, weakened, or changed?

Do not regenerate a new random execution for each intervention.

Do not silently replace paired counterfactual comparison with two unrelated Monte Carlo runs.

Independent Monte Carlo comparison may be useful later, but it is not the default behavior for P0.

---

## 9. Interventions

P0 should support both noise-side and runtime-side interventions.

### 9.1 Noise-Side Interventions

```text
remove_data_errors
remove_measurement_errors
remove_idle_errors
weaken_data_errors_50pct
weaken_measurement_errors_50pct
weaken_idle_errors_50pct
```

Expected behavior:

```text
remove_data_errors:
  Remove all data_error events and set data_error_count to 0.

remove_measurement_errors:
  Remove all measurement_error events and set measurement_error_count to 0.

remove_idle_errors:
  Remove all idle_error events and set idle_error_count to 0.

weaken_data_errors_50pct:
  Multiply data_error event weights by 0.5 and update derived failure score.

weaken_measurement_errors_50pct:
  Multiply measurement_error event weights by 0.5 and update derived failure score.

weaken_idle_errors_50pct:
  Multiply idle_error event weights by 0.5 and update derived failure score.
```

### 9.2 Runtime-Side Interventions

```text
remove_decoder_timeout
reduce_decoder_delay_50pct
reduce_idle_exposure_50pct
```

Expected behavior:

```text
remove_decoder_timeout:
  Set decoder_timeout to false and remove or zero decoder_timeout events.

reduce_decoder_delay_50pct:
  Multiply decoder_delay by 0.5 and update decoder_delay event weights.

reduce_idle_exposure_50pct:
  Multiply idle_exposure by 0.5 and update idle_exposure contribution.
```

Each intervention should transform the baseline execution record and then recompute:

```text
logical_failure
failure_round
failure_region
failure_operation
failure_mode
failure_pattern
```

The same original seed and baseline event record should be reused for intervention analysis.

---

## 10. Intervention Data Model

The intervention file should contain one row per baseline shot per intervention.

Minimum fields for `intervened_runs.csv`:

```text
shot_id
intervention
baseline_logical_failure
intervened_logical_failure
baseline_failure_round
intervened_failure_round
baseline_failure_mode
intervened_failure_mode
baseline_failure_pattern
intervened_failure_pattern
rescued_failure
new_failure
```

Recommended meanings:

```text
shot_id:
  The original baseline execution id.

intervention:
  Name of the intervention applied.

baseline_logical_failure:
  Logical failure result before intervention.

intervened_logical_failure:
  Logical failure result after intervention.

baseline_failure_round:
  Failure round in the baseline execution.

intervened_failure_round:
  Failure round after intervention.

baseline_failure_mode:
  Failure mode before intervention.

intervened_failure_mode:
  Failure mode after intervention.

baseline_failure_pattern:
  Failure pattern before intervention.

intervened_failure_pattern:
  Failure pattern after intervention.

rescued_failure:
  True if baseline failed but intervention succeeded.

new_failure:
  True if baseline succeeded but intervention failed.
```

The intervention file should not be a mixed file containing arbitrary baseline-only rows.

It should be structured as:

```text
one row per baseline shot per intervention
```

This makes attribution analysis simple and reproducible.

---

## 11. Metrics

The attribution analysis should compute:

```text
intervention
num_shots
baseline_logical_failure_rate
intervened_logical_failure_rate
absolute_delta_lfr
relative_delta_lfr
baseline_failure_count
intervened_failure_count
rescued_failure_count
new_failure_count
rescue_rate
new_failure_rate
dominant_baseline_failure_pattern
dominant_intervened_failure_pattern
```

Definitions:

```text
absolute_delta_lfr:
  intervened_logical_failure_rate - baseline_logical_failure_rate

relative_delta_lfr:
  absolute_delta_lfr / baseline_logical_failure_rate

baseline_failure_count:
  Number of baseline shots with logical_failure = true.

intervened_failure_count:
  Number of intervened shots with logical_failure = true.

rescued_failure_count:
  Number of shots where baseline failed but intervention succeeded.

new_failure_count:
  Number of shots where baseline succeeded but intervention failed.

rescue_rate:
  rescued_failure_count / baseline_failure_count

new_failure_rate:
  new_failure_count / num_shots
```

For most removal or weakening interventions, `new_failure_count` should usually be zero in the toy model.

If it is not zero, the code should not hide it.

The analysis should rank interventions by their effect on logical failure behavior.

A larger negative `absolute_delta_lfr` means the intervention reduced logical failure rate more strongly.

---

## 12. Failure Pattern Analysis

P0 should compute a simple failure pattern distribution.

Possible pattern categories:

```text
no_failure
data_dominated
measurement_dominated
idle_correlated
timeout_correlated
delay_correlated
mixed
region_hotspot
operation_hotspot
```

A simple deterministic assignment rule is acceptable.

Example rule:

```text
no_failure:
  logical_failure is false

data_dominated:
  data_error contribution is the largest failure-score component

measurement_dominated:
  measurement_error contribution is the largest failure-score component

idle_correlated:
  idle_error or idle_exposure contribution is dominant

timeout_correlated:
  decoder_timeout is true and contributes substantially to failure_score

delay_correlated:
  decoder_delay contribution is substantial

mixed:
  no single factor dominates

region_hotspot:
  many harmful events concentrate in one region

operation_hotspot:
  many harmful events concentrate near one operation
```

The exact rule can be simple, but it must be documented in code comments.

Failure pattern analysis should not replace intervention-based attribution.

It is only a supporting signal.

---

## 13. Required Scripts

### 13.1 `scripts/01_generate_runs.py`

Purpose:

Generate baseline logical execution records.

Expected command:

```bash
python scripts/01_generate_runs.py \
  --num-shots 1000 \
  --num-rounds 10 \
  --num-operations 5 \
  --data-error-rate 0.03 \
  --measurement-error-rate 0.02 \
  --idle-error-rate 0.01 \
  --decoder-timeout-rate 0.02 \
  --seed 42 \
  --output data/results/baseline_runs.csv
```

Expected output:

```text
data/results/baseline_runs.csv
```

Validation:

```text
The output file exists.
The output file contains one row per shot.
The output columns match the baseline schema.
```

---

### 13.2 `scripts/02_apply_interventions.py`

Purpose:

Apply paired counterfactual interventions to baseline records.

Expected command:

```bash
python scripts/02_apply_interventions.py \
  --input data/results/baseline_runs.csv \
  --output data/results/intervened_runs.csv
```

Expected output:

```text
data/results/intervened_runs.csv
```

Validation:

```text
The output file exists.
The output file contains one row per baseline shot per intervention.
The output columns match the intervention schema.
```

---

### 13.3 `scripts/03_analyze_attribution.py`

Purpose:

Compute failure sensitivity metrics.

Expected command:

```bash
python scripts/03_analyze_attribution.py \
  --input data/results/intervened_runs.csv \
  --output data/results/attribution_summary.csv
```

Expected output:

```text
data/results/attribution_summary.csv
```

Validation:

```text
The output file exists.
The output file contains one row per intervention.
The output includes LFR deltas, rescued failures, new failures, and dominant patterns.
```

---

### 13.4 `scripts/04_plot_results.py`

Purpose:

Plot intervention effects.

Expected command:

```bash
python scripts/04_plot_results.py \
  --input data/results/attribution_summary.csv \
  --output figures/intervention_delta_lfr.png
```

Expected output:

```text
figures/intervention_delta_lfr.png
```

Expected plot:

```text
x-axis: intervention
y-axis: absolute_delta_lfr
```

Negative values indicate that the intervention reduced logical failure rate.

Validation:

```text
The output figure exists.
The plot is readable.
The intervention labels are visible or rotated if needed.
```

---

## 14. Expected Minimal Pipeline

The full P0 pipeline should be:

```bash
python scripts/01_generate_runs.py \
  --num-shots 1000 \
  --num-rounds 10 \
  --num-operations 5 \
  --data-error-rate 0.03 \
  --measurement-error-rate 0.02 \
  --idle-error-rate 0.01 \
  --decoder-timeout-rate 0.02 \
  --seed 42 \
  --output data/results/baseline_runs.csv

python scripts/02_apply_interventions.py \
  --input data/results/baseline_runs.csv \
  --output data/results/intervened_runs.csv

python scripts/03_analyze_attribution.py \
  --input data/results/intervened_runs.csv \
  --output data/results/attribution_summary.csv

python scripts/04_plot_results.py \
  --input data/results/attribution_summary.csv \
  --output figures/intervention_delta_lfr.png
```

The pipeline should produce:

```text
data/results/baseline_runs.csv
data/results/intervened_runs.csv
data/results/attribution_summary.csv
figures/intervention_delta_lfr.png
```

---

## 15. Definition of Done for P0

P0 is complete when:

1. All four scripts run from a clean checkout.
2. The pipeline generates at least:
   - `data/results/baseline_runs.csv`
   - `data/results/intervened_runs.csv`
   - `data/results/attribution_summary.csv`
   - `figures/intervention_delta_lfr.png`
3. The CSV files contain interpretable columns.
4. The plot shows intervention effects on logical failure rate.
5. The README contains the full reproduction command.
6. No large framework abstraction is introduced unnecessarily.
7. P0 includes at least one runtime-side intervention.
8. Interventions are paired with baseline shots rather than independently resampled.

P0 is not complete if it only reports baseline logical failure rate.

P0 is not complete if it only performs static failure classification without intervention comparison.

---

## 16. Success Criterion

The P0 prototype is successful if it can produce a table like:

```text
intervention                       baseline_lfr   intervened_lfr   absolute_delta_lfr
remove_data_errors                 0.084          0.011            -0.073
remove_measurement_errors          0.084          0.052            -0.032
remove_idle_errors                 0.084          0.075            -0.009
weaken_data_errors_50pct           0.084          0.045            -0.039
weaken_measurement_errors_50pct    0.084          0.067            -0.017
remove_decoder_timeout             0.084          0.029            -0.055
reduce_decoder_delay_50pct         0.084          0.061            -0.023
reduce_idle_exposure_50pct         0.084          0.041            -0.043
```

The exact numbers do not matter.

The important point is that the pipeline can quantify how failure behavior changes under both noise-side and runtime-side interventions.

A useful P0 result should allow a statement such as:

```text
This toy workload is more sensitive to decoder timeout and idle exposure
than to measurement noise.
```

That is the core FailureOps behavior.

---

## 17. Implementation Notes

P0 should prefer simple implementation choices.

Recommended:

```text
Python standard library for CSV and JSON
dataclasses for records
argparse for scripts
matplotlib for the plot
small deterministic helper functions
clear comments for the toy failure rule
```

Avoid:

```text
large dependencies
plugin systems
abstract simulator engines
deep inheritance
hidden global state
implicit file paths
unvalidated scripts
```

Field names should remain stable once introduced.

Generated artifacts should be written under:

```text
data/
results/
figures/
```

In this repository, the preferred result path is:

```text
data/results/
```

---

## 18. Later Extensions

After P0, possible extensions include:

- replace toy failure rule with a lightweight stabilizer simulation
- connect to Stim or PyMatching
- use seed-stable backend evaluation when possible
- analyze spacetime regions more realistically
- model decoder backlog or delayed decoding
- compare multiple logical circuits
- add operation-level attribution
- add region-level sensitivity heatmaps
- add failure timing histograms
- add runtime-failure correlation plots
- generate a Markdown failure attribution report

Later extensions should preserve the core pipeline:

```text
baseline execution
        ↓
intervention
        ↓
comparison
        ↓
sensitivity attribution
```

Do not turn FailureOps into a generic QEC simulator unless it directly supports intervention-based failure behavior attribution.