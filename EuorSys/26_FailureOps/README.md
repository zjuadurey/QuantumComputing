# FailureOps

FailureOps is a research prototype for intervention-based failure attribution in QEC-protected logical circuit executions.

Its goal is not to design a new decoder, a new code, or a new threshold estimator.

Its goal is to answer:

> Given a QEC-protected logical circuit execution, which intervenable factors is the logical failure behavior sensitive to?

FailureOps treats logical failure as an execution-level behavior rather than a single aggregate metric.

---

## Core Question

Given a QEC-protected logical circuit execution, FailureOps asks:

> If we change, remove, or weaken a specific factor, how much does the logical failure behavior change?

This is different from simply classifying a failure as:

```text
data error
measurement error
idle error
decoder error
runtime delay
```

Such labels may be useful, but they are not sufficient.

FailureOps is based on counterfactual intervention:

```text
baseline execution
        ↓
change one intervenable factor
        ↓
re-evaluate failure behavior
        ↓
measure the difference
        ↓
attribute sensitivity
```

The key object is not the decoder, the code, or the threshold.

The key object is:

> the failure behavior of a QEC-protected logical execution.

---

## Key Principle

FailureOps is based on one principle:

> Attribution must be bound to intervention.

A static label such as “data-error-related failure” is not enough.

A factor is important only if changing that factor changes the observed logical failure behavior.

For example, this statement is weak:

```text
This failure is caused by measurement errors.
```

This statement is closer to FailureOps:

```text
Reducing measurement errors by 50% reduces logical failure rate by 8%,
while removing decoder timeout reduces logical failure rate by 47%.
This failure class is therefore more sensitive to decoder timeout than to measurement noise.
```

FailureOps should avoid claiming a single unique cause for a logical failure.

Instead, it should report sensitivity:

> Which factors, when changed, most strongly change the failure behavior?

---

## Failure Behavior

In FailureOps, failure behavior is richer than a binary success/failure label.

The repository may track:

```text
logical failure rate
failure timing
failure mode
failure pattern
failure location
failure correlation with runtime events
failure changes under interventions
```

A failure pattern may refer to whether failures are concentrated around:

```text
specific syndrome rounds
specific logical operations
idle windows
decoder delay spikes
decoder timeout events
backlog intervals
burst-like error clusters
specific spacetime regions
```

The goal is not to assign a static label to each failure.

The goal is to measure how the failure behavior changes when an intervenable factor is changed.

---

## What This Repository Is For

This repository is for building a minimal runnable pipeline that can:

1. Generate logical execution records.
2. Simulate or load failure outcomes.
3. Apply counterfactual interventions.
4. Quantify changes in logical failure behavior.
5. Produce reproducible CSV summaries and figures.
6. Report which factors the failure behavior is most sensitive to.

The first version is intentionally small.

The goal is to make the FailureOps abstraction executable before connecting it to heavier QEC simulation backends.

---

## What This Repository Is Not For

This repository is not primarily for:

```text
designing a new decoder
designing a new QEC code
estimating thresholds
benchmarking only average logical error rate
building a complete FTQC runtime
building a full QEC simulator
```

FailureOps may later use real decoders, real codes, and richer simulation backends.

However, those are not the core contribution of the repository.

The core contribution is the attribution pipeline:

```text
logical execution records
        ↓
interventions
        ↓
failure behavior comparison
        ↓
sensitivity profile
```

---

## Current Status

Current target:

```text
P10: EuroSys main evidence pass
```

The repository now includes the original P0 toy pipeline plus P1/P2/P2.5/P3/P4
through P10 extensions. P10 adds:

```text
baseline comparison
real-data robustness checks
measured decoder-runtime replay closure
paired statistical significance
claim-to-evidence audit
```

The implementation remains intentionally small. P10 strengthens the paper-facing
evidence story; it is not a full FTQC runtime, live hardware feedback trace, or
decoder scheduler.

The purpose of P0 is not to prove a QEC result.

The purpose of P0 is to verify that the FailureOps workflow can be made executable.

---

## P0 Design Goal

P0 should demonstrate the following minimal loop:

```text
generate baseline logical execution records
        ↓
apply controlled interventions
        ↓
compare baseline and intervened outcomes
        ↓
compute sensitivity metrics
        ↓
produce attribution summary
```

The P0 pipeline should be small, transparent, and reproducible.

It should avoid heavy dependencies.

It should avoid implementing a full QEC simulator.

---

## Paired Counterfactual Evaluation

FailureOps should prefer paired counterfactual evaluation.

For each baseline shot, interventions should be evaluated using the same:

```text
shot_id
seed
underlying sampled event record
logical workload configuration
runtime configuration
```

when possible.

This allows the analysis to ask:

> Would this same execution still fail if a specific factor were removed, weakened, or changed?

This is different from simply running two independent Monte Carlo experiments and comparing their average logical failure rates.

Independent Monte Carlo runs can be useful later, but P0 should start with paired comparison because it makes attribution easier to interpret.

---

## Expected P0 Pipeline

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

---

## Expected Outputs

```text
data/results/baseline_runs.csv
data/results/intervened_runs.csv
data/results/attribution_summary.csv
figures/intervention_delta_lfr.png
```

Optional later outputs:

```text
figures/failure_timing_histogram.png
figures/failure_pattern_shift.png
figures/runtime_failure_correlation.png
reports/failureops_report.md
```

---

## Minimum P0 Record Schema

The baseline execution file should contain one row per shot.

Minimum schema for `baseline_runs.csv`:

```text
shot_id
seed
num_rounds
num_operations
data_error_count
measurement_error_count
idle_error_count
decoder_timeout
decoder_delay
idle_exposure
logical_failure
failure_round
failure_mode
failure_pattern
```

Suggested meanings:

```text
shot_id:
  Unique execution id.

seed:
  Random seed used for the shot or shot group.

num_rounds:
  Number of syndrome rounds or toy execution rounds.

num_operations:
  Number of logical operations or toy operation segments.

data_error_count:
  Number of sampled data-error events.

measurement_error_count:
  Number of sampled measurement-error events.

idle_error_count:
  Number of sampled idle-error events.

decoder_timeout:
  Whether a decoder timeout occurred.

decoder_delay:
  Toy decoder delay value.

idle_exposure:
  Toy measure of accumulated idle exposure.

logical_failure:
  Whether the execution failed logically.

failure_round:
  Round where failure is detected or assigned.

failure_mode:
  Toy failure type, such as logical_X, logical_Z, mixed, or none.

failure_pattern:
  Coarse failure pattern, such as noise_dominated, idle_correlated,
  timeout_correlated, burst_like, or none.
```

P0 may use simple synthetic definitions for these fields.

The definitions should be transparent and documented in code comments.

---

## Minimum Intervention Record Schema

The intervention file should contain one row per baseline shot per intervention.

Minimum schema for `intervened_runs.csv`:

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

Suggested meanings:

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

---

## Example Interventions

P0 starts with noise-side interventions because they are easiest to make executable.

Initial interventions include:

```text
remove_data_errors
remove_measurement_errors
remove_idle_errors
weaken_data_errors_50pct
weaken_measurement_errors_50pct
weaken_idle_errors_50pct
```

However, FailureOps should not collapse into ordinary noise attribution.

Decoder and runtime factors are first-class intervenable factors.

Later versions should include interventions such as:

```text
remove_decoder_timeout
increase_decoder_capacity_2x
reduce_decoder_latency_50pct
reduce_idle_exposure_50pct
increase_synchronization_slack
change_operation_pacing
change_decoder_timeout_policy
change_backlog_limit
```

The long-term goal is to compare physical-noise interventions and system/runtime interventions in one attribution framework.

---

## Example Metrics

The attribution summary may include:

```text
intervention
baseline_logical_failure_rate
intervened_logical_failure_rate
absolute_delta_lfr
relative_delta_lfr
rescued_failure_count
new_failure_count
rescue_rate
new_failure_rate
failure_pattern_distribution_shift
failure_mode_distribution_shift
failure_timing_shift
```

Suggested meanings:

```text
absolute_delta_lfr:
  intervened_logical_failure_rate - baseline_logical_failure_rate

relative_delta_lfr:
  absolute_delta_lfr / baseline_logical_failure_rate

rescued_failure_count:
  Number of shots that failed in baseline but succeeded after intervention.

new_failure_count:
  Number of shots that succeeded in baseline but failed after intervention.

failure_pattern_distribution_shift:
  Change in the distribution of failure_pattern values.

failure_timing_shift:
  Change in the distribution of failure_round values.
```

For ranking sensitivity, the most important interventions are usually those with larger reductions in logical failure rate and meaningful changes in failure pattern.

---

## Example Attribution Interpretation

A useful FailureOps result should look like this:

```text
Baseline logical failure rate: 3.2%

remove_decoder_timeout:
  logical failure rate = 1.1%
  absolute delta = -2.1%
  relative delta = -65.6%
  most timeout-correlated failures disappear

weaken_measurement_errors_50pct:
  logical failure rate = 2.8%
  absolute delta = -0.4%
  relative delta = -12.5%
  failure pattern mostly unchanged

weaken_idle_errors_50pct:
  logical failure rate = 2.2%
  absolute delta = -1.0%
  relative delta = -31.3%
  idle-correlated failures decrease
```

The interpretation is:

```text
This workload's failure behavior is more sensitive to decoder timeout
and idle exposure than to measurement noise.
```

This is the core value of FailureOps.

---

## Suggested Repository Layout

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

---

## Module Responsibilities

### `failureops/data_model.py`

Defines the basic data structures for:

```text
baseline execution record
intervention result record
attribution summary record
```

P0 may use dataclasses.

The fields should map directly to the CSV schema.

### `failureops/toy_simulator.py`

Generates toy logical execution records.

It should simulate:

```text
data errors
measurement errors
idle errors
decoder timeout
decoder delay
idle exposure
logical failure result
failure mode
failure pattern
```

The toy failure rule should be simple and transparent.

Example toy rule:

```text
logical_failure becomes more likely when:
  data_error_count is high
  measurement_error_count is high
  idle_error_count is high
  decoder_timeout is true
  decoder_delay is large
  idle_exposure is large
```

The toy rule does not need to be physically complete.

It only needs to make the intervention pipeline executable.

### `failureops/interventions.py`

Defines interventions as explicit transformations over execution records.

Each intervention should take a baseline record and produce an intervened record.

P0 interventions may include:

```text
remove_data_errors
remove_measurement_errors
remove_idle_errors
weaken_data_errors_50pct
weaken_measurement_errors_50pct
weaken_idle_errors_50pct
```

Later interventions may include:

```text
remove_decoder_timeout
reduce_decoder_latency_50pct
reduce_idle_exposure_50pct
increase_decoder_capacity_2x
```

### `failureops/metrics.py`

Computes attribution metrics.

Examples:

```text
logical failure rate
absolute delta LFR
relative delta LFR
rescued failure count
new failure count
failure pattern distribution shift
failure timing shift
```

### `failureops/io_utils.py`

Handles CSV input and output.

P0 should use standard Python libraries when possible.

Avoid heavy dependencies unless necessary.

---

## Development Principle

Start small.

The first goal is not to build a complete framework.

The first goal is to make the idea executable.

Every module should serve the core question:

> Does this help us measure which intervenable factors the failure behavior is sensitive to?

If a module does not help answer this question, it should not be part of P0.

---

## P0 Implementation Scope

P0 should implement:

```text
repository skeleton
baseline record generation
CSV output
basic interventions
paired intervention records
attribution summary
one simple plot
```

P0 should not implement:

```text
real surface-code simulation
real lattice surgery
real decoder integration
threshold experiments
large benchmark suite
complex visualization
```

P0 is successful if it can produce:

```text
baseline_runs.csv
intervened_runs.csv
attribution_summary.csv
intervention_delta_lfr.png
```

and if the attribution summary clearly ranks interventions by their effect on logical failure behavior.

---

## New Session Context

For new coding sessions, do not read every milestone document by default.
Start with:

```text
AGENTS.md
docs/START_HERE.md
docs/P10.md
```

Then read only the milestone document that matches the task. For example, use
`docs/P4.md` for pairing-contract work, `docs/P7.md` for real-record ingestion,
and `docs/P8.md` for measured decoder-runtime replay.

Historical setup instructions for the original P0 bootstrap are preserved in
`docs/P0.md`, `docs/spec.md`, and `docs/roadmap.md`.

---

## Original P0 Expected Command

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

---

## Original P0 Expected Output

```text
data/results/baseline_runs.csv
```

The file should contain one row per shot and follow the minimum P0 baseline schema.

---

## Roadmap

### P0: Toy intervention pipeline

Goal:

```text
Make the FailureOps abstraction executable.
```

Includes:

```text
toy execution records
toy logical failure rule
paired interventions
CSV summaries
basic plot
```

### P1: Runtime-aware interventions

Goal:

```text
Make decoder/runtime factors first-class interventions.
```

Possible additions:

```text
decoder timeout intervention
decoder latency intervention
idle exposure intervention
backlog intervention
operation pacing intervention
```

### P2: Connect to lightweight QEC simulation

Goal:

```text
Replace or supplement the toy failure rule with a lightweight QEC backend.
```

Possible directions:

```text
small repetition code
small surface-code-like memory
stim-generated syndrome records
simple decoder integration
```

### P3: FailureOps report

Goal:

```text
Generate a readable failure attribution report.
```

Possible outputs:

```text
sensitivity table
failure timing plot
failure pattern shift plot
runtime-failure correlation plot
markdown report
```

---

## Notes

The first version may use a toy event model and a transparent toy failure rule.

This is acceptable for P0.

The purpose of P0 is not to prove a QEC result.

The purpose of P0 is to verify that the FailureOps abstraction can be made executable:

```text
baseline execution
        ↓
counterfactual intervention
        ↓
failure behavior comparison
        ↓
sensitivity attribution
```

The most important thing is to preserve the core idea:

> FailureOps does not merely ask what kind of error occurred.  
> FailureOps asks how logical failure behavior changes under controlled interventions.
