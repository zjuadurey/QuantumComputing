# AGENTS.md

## Project Role

This repository is a research prototype for **FailureOps**, a systems-oriented framework for analyzing failure behavior in QEC-protected logical circuit executions.

The goal is to build small, runnable experimental pipelines that reveal how logical failures respond to counterfactual interventions on system-controllable factors.

Prefer minimal runnable workflows over large abstract frameworks.

The first priority is to make the FailureOps idea executable.

## Core Research Definition

FailureOps studies:

> Given a QEC-protected logical circuit execution, which intervenable factors is its logical failure behavior sensitive to?

The object of study is not the decoder, not the code, and not the threshold.

The object of study is:

> the failure behavior of a logical-level execution under QEC protection.

FailureOps does not merely ask:

> Did the failure come from data errors or measurement errors?

Instead, it asks:

> If we change, remove, weaken, delay, prioritize, or isolate a certain factor, how much does the logical failure rate or failure pattern change?

Attribution must be tied to intervention.

Label statistics without intervention are not sufficient for FailureOps attribution.

## FailureOps Attribution Rule

FailureOps attribution is intervention-based and sensitivity-based.

A useful attribution result should compare:

```text
baseline execution
        vs.
intervened execution
```

and report how the failure behavior changes.

Do not treat static labels such as the following as the final result:

```text
data-error-related failure
measurement-error-related failure
idle-error-related failure
decoder-related failure
```

These labels may be useful intermediate fields, but they are not the core output.

The core output should be a sensitivity profile:

```text
which intervention changes logical failure behavior the most
```

## Paired Counterfactual Rule

Prefer paired counterfactual evaluation.

When possible, an intervention should be evaluated against the same:

```text
shot_id
seed
sampled event record
logical workload configuration
runtime configuration
```

as the baseline execution.

The intended question is:

> Would this same execution still fail if one factor were removed, weakened, or changed?

Do not silently replace paired intervention with two unrelated Monte Carlo runs unless the task explicitly asks for independent resampling.

## What This Project Is Not

Do not frame or implement this project as:

- a new QEC decoder
- a new quantum error-correcting code
- a threshold-estimation framework
- a pure logical error rate benchmark
- a generic error classification tool
- a full FTQC architecture simulator
- a decoder resource scheduler as the main artifact
- a full causal inference framework

The project may use existing decoders, codes, or simulators as dependencies, but they are not the main contribution.

Decoder and runtime factors may appear as intervenable factors, such as:

```text
decoder timeout
decoder delay
decoder backlog
decoder capacity
idle exposure
synchronization slack
operation pacing
```

However, building a decoder scheduler or a full runtime is not the goal of the early repository.

## Current P0 Scope

The current scope is:

```text
P0: toy intervention pipeline
```

P0 should implement a minimal closed loop:

```text
generate baseline logical execution records
        ↓
apply paired interventions
        ↓
compare baseline and intervened outcomes
        ↓
compute attribution metrics
        ↓
write CSV summaries and simple figures
```

P0 may use:

```text
toy event model
toy logical failure rule
CSV files
simple Python scripts
standard library or lightweight dependencies
```

P0 should not implement:

```text
real surface-code simulation
real lattice surgery
real decoder integration
threshold experiments
large benchmark suite
complex visualization
plugin systems
abstract simulation engines
```

## Coding Principles

- Keep the first implementation small and runnable.
- Prefer explicit scripts over deep framework abstractions.
- Avoid premature generality.
- Avoid heavy dependencies unless clearly justified.
- Do not implement features outside the current task scope.
- Do not rewrite unrelated files.
- Every script should be runnable from the command line.
- Every generated artifact should be saved under `data/`, `results/`, or `figures/`.
- Prefer transparent toy rules over hidden complex logic.
- Prefer readable CSV outputs over opaque binary formats.
- Keep field names stable once they are introduced.

## Expected Repository Style

Use this style:

```text
scripts/01_generate_runs.py
scripts/02_apply_interventions.py
scripts/03_analyze_attribution.py
scripts/04_plot_results.py
```

Avoid this style in the early phase:

```text
failureops/core/abstract_runtime.py
failureops/simulator/base_engine.py
failureops/intervention/plugin_manager.py
failureops/analysis/causal_graph.py
```

The first priority is a reproducible closed loop, not a complete software architecture.

## Expected Output Files

The early pipeline should produce files such as:

```text
data/results/baseline_runs.csv
data/results/intervened_runs.csv
data/results/attribution_summary.csv
figures/intervention_delta_lfr.png
```

The exact filenames may vary by task, but generated artifacts should be easy to find and reproduce.

## Validation Rule

After making code changes, run the smallest available command that validates the change.

If a script is added, run it.

If a test is added, run the test.

If a plot script is added, run it and confirm the figure is generated.

If the command cannot be run, report:

1. The exact command that should have been run.
2. The exact reason it could not be run.
3. Whether the failure is due to missing dependency, missing input file, environment issue, or implementation error.

## Output Rule

When a task is completed, report:

1. Which files were changed.
2. Which command was run.
3. Which output files were generated.
4. What remains unimplemented.

Do not claim completion if the validation command was not run successfully.

## Scope Discipline

Before adding any module or feature, check whether it helps answer:

> Which intervenable factors is the logical failure behavior sensitive to?

If the answer is no, do not add it in the current phase.

Examples:

```text
new decoder:
  not needed unless the task is decoder intervention comparison

new code family:
  not needed unless the task is code-dependent sensitivity comparison

threshold sweep:
  not needed unless the task is sensitivity across physical error rates

complex visualization:
  not needed unless it directly shows failure behavior shift

scheduler implementation:
  not needed unless the task explicitly studies runtime intervention
```

The repository should remain centered on the FailureOps pipeline:

```text
logical execution records
        ↓
interventions
        ↓
failure behavior comparison
        ↓
sensitivity attribution
```