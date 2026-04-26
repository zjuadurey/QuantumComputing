# FailureOps Roadmap

This roadmap defines the staged development plan for FailureOps.

The guiding principle is:

> Build the smallest runnable attribution loop first.

FailureOps should not start as a full QEC simulator, a decoder framework, or a runtime scheduler.

It should start as a minimal pipeline that can represent logical execution records, apply paired counterfactual interventions, and report how failure behavior changes.

---

## Overall Direction

FailureOps studies:

> Which intervenable factors is the logical failure behavior of a QEC-protected execution sensitive to?

The repository should therefore evolve around one core pipeline:

```text
logical execution records
        ↓
paired interventions
        ↓
failure behavior comparison
        ↓
sensitivity attribution
        ↓
report / plot
```

Each stage should preserve this pipeline.

---

## P0: Toy Intervention Pipeline

### Goal

Build the smallest runnable FailureOps loop.

P0 should make the idea executable without requiring a real QEC backend.

### Scope

P0 includes:

```text
toy logical execution simulator
event generation
toy logical failure rule
paired counterfactual intervention
CSV output
simple attribution metrics
simple plot
at least one toy runtime factor
```

The runtime factor is important. P0 should not look like ordinary noise attribution only.

At minimum, P0 should include factors such as:

```text
data_error_count
measurement_error_count
idle_error_count
decoder_timeout
decoder_delay
idle_exposure
```

### P0 Answers

P0 should answer:

> Can we represent logical failure behavior and measure how it changes under simple noise-side and runtime-side interventions?

### P0 Does Not Need

P0 does not need:

```text
real surface-code simulation
real lattice surgery
real decoder integration
threshold experiments
large benchmark suite
complex visualization
```

### Expected P0 Output

P0 should generate:

```text
data/results/baseline_runs.csv
data/results/intervened_runs.csv
data/results/attribution_summary.csv
figures/intervention_delta_lfr.png
```

---

## P0 Step 1: Generate Baseline Runs

### Goal

Generate synthetic logical execution records.

Each row represents one logical execution shot.

### Files

```text
failureops/data_model.py
failureops/io_utils.py
failureops/toy_simulator.py
scripts/01_generate_runs.py
```

### Required Output

```text
data/results/baseline_runs.csv
```

### Minimum Baseline Schema

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

### Expected Command

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

### Validation

After implementation, run the command above and confirm that:

```text
data/results/baseline_runs.csv
```

exists and contains one row per shot.

---

## P0 Step 2: Apply Paired Interventions

### Goal

Apply counterfactual interventions to the same baseline execution records.

The key rule is:

> Keep the same shot and event record when possible; change only the intervened factor.

### Files

```text
failureops/interventions.py
scripts/02_apply_interventions.py
```

### Required Output

```text
data/results/intervened_runs.csv
```

### Minimum Intervention Schema

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

### Initial Interventions

P0 should include both noise-side and runtime-side interventions.

Noise-side interventions:

```text
remove_data_errors
remove_measurement_errors
remove_idle_errors
weaken_data_errors_50pct
weaken_measurement_errors_50pct
weaken_idle_errors_50pct
```

Runtime-side interventions:

```text
remove_decoder_timeout
reduce_decoder_delay_50pct
reduce_idle_exposure_50pct
```

### Expected Command

```bash
python scripts/02_apply_interventions.py \
  --input data/results/baseline_runs.csv \
  --output data/results/intervened_runs.csv
```

### Validation

Confirm that:

```text
data/results/intervened_runs.csv
```

contains one row per baseline shot per intervention.

---

## P0 Step 3: Analyze Attribution

### Goal

Compute how each intervention changes failure behavior.

### Files

```text
failureops/metrics.py
scripts/03_analyze_attribution.py
```

### Required Output

```text
data/results/attribution_summary.csv
```

### Metrics

The attribution summary should include:

```text
intervention
num_shots
baseline_logical_failure_rate
intervened_logical_failure_rate
absolute_delta_lfr
relative_delta_lfr
rescued_failure_count
new_failure_count
rescue_rate
new_failure_rate
dominant_baseline_failure_pattern
dominant_intervened_failure_pattern
```

### Expected Command

```bash
python scripts/03_analyze_attribution.py \
  --input data/results/intervened_runs.csv \
  --output data/results/attribution_summary.csv
```

### Validation

Confirm that:

```text
data/results/attribution_summary.csv
```

exists and ranks interventions by their effect on logical failure behavior.

---

## P0 Step 4: Plot Results

### Goal

Generate a simple figure showing intervention effects.

### Files

```text
scripts/04_plot_results.py
```

### Required Output

```text
figures/intervention_delta_lfr.png
```

### Expected Plot

A simple bar plot:

```text
x-axis: intervention
y-axis: absolute_delta_lfr
```

Negative values indicate that the intervention reduced logical failure rate.

### Expected Command

```bash
python scripts/04_plot_results.py \
  --input data/results/attribution_summary.csv \
  --output figures/intervention_delta_lfr.png
```

### Validation

Confirm that:

```text
figures/intervention_delta_lfr.png
```

exists.

---

## P0 Completion Criteria

P0 is complete when the following command sequence runs successfully:

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

and produces:

```text
data/results/baseline_runs.csv
data/results/intervened_runs.csv
data/results/attribution_summary.csv
figures/intervention_delta_lfr.png
```

P0 is not complete if it only generates baseline logical failure rate without intervention-based comparison.

---

## P1: Connect to Lightweight QEC Backend

### Goal

Replace or supplement the toy failure rule with a lightweight QEC backend.

P1 should preserve the FailureOps pipeline while making the execution model closer to QEC.

### Possible Backend Choices

Possible directions:

```text
small repetition code
small surface-code-like memory
Stim-generated syndrome records
PyMatching decoding
simple detector-event pipeline
```

### Recommended P1 Direction

```text
Stim repetition-code or surface-code memory experiment
        ↓
PyMatching decoding
        ↓
record syndrome/logical outcome and available event information
        ↓
perform seed-stable baseline/intervention comparison when possible
        ↓
measure logical failure sensitivity
```

The exact replay mechanism may depend on the backend.

P1 should preserve the paired-intervention spirit without overbuilding a custom simulator.

### P1 Should Avoid

P1 should avoid:

```text
full lattice surgery
large code distances
complex FTQC workloads
custom decoder implementation
threshold-focused experiments
```

### P1 Completion Criteria

P1 is complete when FailureOps can run at least one lightweight QEC-backed experiment and produce the same type of attribution summary as P0.

---

## P2: Runtime/System Interventions

### Goal

Make decoder/runtime factors first-class interventions.

This is where FailureOps becomes more systems-oriented.

### Candidate Runtime Factors

```text
decoder timeout
decoder delay
decoder backlog
decoder service capacity
idle exposure
operation waiting time
synchronization slack
operation pacing
delayed correction
```

### Candidate Interventions

```text
remove_decoder_timeout
reduce_decoder_latency_50pct
increase_decoder_capacity_2x
reduce_idle_exposure_50pct
increase_synchronization_slack
change_operation_pacing
change_timeout_policy
```

### P2 Output

P2 should compare physical-noise interventions and runtime interventions in the same attribution table.

Example interpretation:

```text
This workload is more sensitive to decoder timeout and idle exposure
than to measurement noise.
```

### P2 Completion Criteria

P2 is complete when at least three runtime/system interventions are implemented and compared against noise-side interventions.

---

## P3: Failure Pattern Attribution

### Goal

Move beyond LFR-only attribution.

P3 should analyze how interventions change failure behavior structure.

### Failure Behavior Dimensions

```text
failure rate
failure timing
failure mode
failure pattern
runtime-failure correlation
operation-localized failure
idle-correlated failure
timeout-correlated failure
```

### Candidate Outputs

```text
failure_timing_histogram.png
failure_pattern_shift.png
runtime_failure_correlation.png
reports/failureops_report.md
```

### P3 Completion Criteria

P3 is complete when FailureOps can show not only whether LFR changed, but also how failure patterns shifted under interventions.

---

## P4: Realistic Case Studies

### Goal

Apply FailureOps to more realistic logical execution workloads.

### Candidate Case Studies

```text
logical memory under different runtime delays
small logical Clifford workload
small surface-code memory experiment
decoder timeout stress test
idle exposure stress test
operation pacing stress test
```

### P4 Should Still Avoid

P4 should still avoid turning into:

```text
a full FTQC architecture simulator
a new decoder paper
a threshold paper
a scheduler paper
```

The case studies should remain centered on attribution.

### P4 Completion Criteria

P4 is complete when the repository contains at least two reproducible case studies with clear intervention-based attribution results.

---

## Long-Term Direction

The long-term goal is to make FailureOps a lightweight failure attribution system for QEC logical executions.

A mature FailureOps workflow should produce:

```text
sensitivity table
failure timing analysis
failure pattern shift analysis
runtime-factor comparison
markdown failure report
reproducible scripts
```

A mature result should allow users to say:

```text
This logical workload fails mostly under decoder timeout and idle exposure.
Reducing measurement noise helps, but less than reducing runtime-induced delay.
The dominant failure pattern shifts after removing timeout events.
```

This is the intended value of FailureOps.

---

## Non-Goals Across All Phases

Across all phases, do not turn FailureOps into:

```text
a new QEC decoder
a new QEC code
a threshold estimator
a generic QEC simulator
a pure logical error rate benchmark
a full FTQC runtime
a decoder scheduler as the main artifact
a full causal inference framework
```

Existing decoders, codes, and simulators may be used as dependencies.

They are not the central contribution.

The central contribution is:

```text
intervention-based failure behavior attribution
```

---

## Development Discipline

Before adding a feature, ask:

> Does this help measure which intervenable factors logical failure behavior is sensitive to?

If not, do not add it in the current phase.

Prefer:

```text
small scripts
explicit CSV files
clear field names
simple validation commands
minimal dependencies
reproducible outputs
```

Avoid:

```text
abstract plugin systems
large framework layers
unnecessary inheritance
hidden state
unvalidated scripts
unexplained toy rules
```

The repository should remain centered on the FailureOps closed loop:

```text
baseline execution
        ↓
intervention
        ↓
comparison
        ↓
sensitivity attribution
```