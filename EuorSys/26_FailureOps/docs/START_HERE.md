# FailureOps Start Here

Use this file as the first context document for new coding sessions. It is the
compressed map of the repository; read detailed milestone docs only when the
task touches that phase.

## Current Status

Current target:

```text
P10: EuroSys main evidence pass
```

FailureOps is now a paired counterfactual attribution prototype for QEC logical
failure behavior. The main paper-facing evidence path is:

```text
public real QEC detector records
        ↓
paired decoder / prior / runtime-replay interventions
        ↓
rescued vs induced logical-failure transitions
        ↓
paired significance, robustness, and claim audit
```

The strongest current claim is:

> FailureOps attributes QEC logical failure behavior by paired interventions,
> and the method can be evaluated on public real QEC detector records plus
> measured decoder-runtime replay.

Do not claim live hardware feedback timing or production control-stack runtime
attribution.

## What To Read

Default new-session context:

```text
AGENTS.md
docs/START_HERE.md
docs/P10.md
```

Read older milestone docs only when needed:

```text
docs/P0.md   toy closed-loop pipeline
docs/P1.md   Stim/PyMatching repetition-code backend
docs/P2.md   runtime/system interventions
docs/P2_5.md robustness and pattern-shift analysis
docs/P3.md   systems evaluation harness
docs/P4.md   pairing contracts, hashes, paired metrics
docs/P5a.md  runtime trace import/export
docs/P5b_P5c.md layered events and surface-code pilot
docs/P6.md   cross-mode evaluation
docs/P7.md   Google RL QEC real-record ingestion
docs/P7_5.md real-data attribution analyses
docs/P8.md   measured decoder-runtime replay
docs/P9.md   evidence bundle
docs/P10.md  EuroSys main evidence pass
```

Long-form conceptual docs:

```text
docs/idea.md     research definition
docs/spec.md     original P0 implementation spec
docs/roadmap.md  historical staged roadmap
```

## Core Invariants

FailureOps attribution is intervention-based:

```text
baseline execution
        vs.
intervened execution
```

The core output is not a static label. It is a sensitivity profile:

```text
which intervention changes logical failure behavior the most
```

Prefer paired counterfactual evaluation. Preserve the same shot identity and
event record whenever the intervention semantics require it.

## Current Evidence Surface

Primary real-record evidence:

```text
data/results/p7_google_rl_qec_decoder_effect_matrix.csv
data/results/p7_5_paired_vs_unpaired_variance.csv
data/results/p7_5_rescue_induction_features.csv
data/results/p7_5_decoder_prior_interventions.csv
```

Expanded Google real-record evidence:

```text
data/results/p10_google_rl_qec_v2_decoder_effect_matrix.csv
data/results/p10_google_rl_qec_v2_decoder_effect_aggregate.csv
data/results/p10_google_rl_qec_v2_rescue_induction_features.csv
data/results/p10_google_rl_qec_v2_paired_vs_unpaired_variance.csv
```

Measured decoder-runtime replay:

```text
data/results/p8_decoder_runtime_trace.csv
data/results/p8_decoder_runtime_summary.csv
```

P10 paper-facing outputs:

```text
data/results/p10_baseline_comparison.csv
data/results/p10_realdata_robustness.csv
data/results/p10_runtime_deadline_interventions.csv
data/results/p10_runtime_deadline_summary.csv
data/results/p10_effect_significance.csv
data/results/p10_claim_audit.csv
figures/p10_eurosys_main_evidence.png
```

External public-dataset replication:

```text
data/results/p10_qec3v5_external_decoder_effect_matrix.csv
data/results/p10_qec3v5_external_decoder_effect_aggregate.csv
data/results/p10_external_sanity_checks.csv
```

## Current External-Data Interpretation

Google RL QEC:

```text
primary public real-record dataset
shot-level detector records
paired decoder-pathway and prior interventions
```

Google qec3v5 2022:

```text
external public real-record replication
shot-level detector records
paired pymatching vs correlated-matching intervention
```

ETH Zurich 2020:

```text
independent public surface-code evidence
figure-level CSV only
not shot-level FailureOps-attribution-compatible
```

DAQEC/IBM:

```text
independent IBM aggregate/session-level sanity evidence
not shot-level detector-record attribution
```

## Main Commands

Run P10 after upstream artifacts exist:

```bash
conda run -n failureops --no-capture-output python scripts/31_run_p10_eurosys_main_evidence.py
```

Run expanded Google RL QEC v2 evidence:

```bash
conda run -n failureops --no-capture-output python scripts/35_run_google_rl_qec_v2_evidence.py
```

This v2 path is used as expanded same-family evidence. In `C6b`, the intended
reading is broad net-rescuing sign stability across the larger corpus plus
negative subgroup means across the observed control modes and bases, not an
independent replication claim.

Run qec3v5 external replication:

```bash
conda run -n failureops --no-capture-output python scripts/32_run_qec3v5_external_replication.py
```

Run independent external sanity checks:

```bash
conda run -n failureops --no-capture-output python scripts/33_run_external_sanity_checks.py
```

Run tests:

```bash
conda run -n failureops --no-capture-output pytest tests
```

## Paper Boundary

Supported:

```text
paired FailureOps methodology
public real QEC detector-record evaluation
external qec3v5 real-record replication
real decoder-pathway and prior interventions
measured decoder-runtime replay
controlled runtime deadline intervention closure
independent non-Google aggregate/figure-level sanity boundaries
```

Not claimed:

```text
live hardware feedback timing
production control-stack queueing attribution
hardware idle exposure caused by measured decoder latency
new decoder or new QEC code
threshold estimation
```
