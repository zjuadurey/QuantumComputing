# P10 EuroSys Style Guide

This note reframes the current FailureOps code and results in a way that reads
more like a EuroSys submission package and less like an internal milestone log.

The goal is not to change the science. The goal is to foreground the systems
thesis, the canonical evidence path, and the bounded claims.

## The EuroSys-Facing Spine

Lead with this pipeline:

```text
public real QEC detector records
        ↓
paired decoder / prior / runtime-replay interventions
        ↓
rescued vs induced logical-failure transitions
        ↓
scope-wise significance, robustness, and claim audit
```

This is the shortest faithful description of what the artifact does.

## What To Foreground

Use these files as the canonical paper-facing package:

```text
scripts/31_run_p10_eurosys_main_evidence.py
scripts/35_run_google_rl_qec_v2_evidence.py
scripts/36_run_google_decoder_priors_evidence.py
scripts/32_run_qec3v5_external_replication.py
scripts/37_build_p10_eurosys_digest.py
```

Historical milestone rebuilders remain available under:

```text
scripts/archive/
```

Use these outputs as the primary evidence surface:

```text
data/results/p10_claim_audit.csv
data/results/p10_realdata_robustness.csv
data/results/p10_effect_significance.csv
data/results/p10_eurosys_digest.csv
figures/p10_google_rl_qec_v2_evidence.png
figures/p10_eurosys_main_evidence.png
```

If a reviewer only opens one compact summary table, point them to:

```text
data/results/p10_eurosys_digest.csv
```

## What To De-Emphasize

Do not lead with:

```text
historical milestone docs
toy closed-loop phases
decoder implementation detail
large lists of intermediate CSVs
generic QEC benchmarking language
```

Those materials are still useful, but they should sit behind the main systems
story rather than competing with it.

## Preferred Result Order

Present the results in this order:

1. **Main real-record claim**  
   FailureOps exposes paired intervention-sensitive logical failure behavior on
   public detector records.

2. **Why pairing is necessary**  
   Pairing lowers estimator variance and preserves rescued versus induced
   transition semantics.

3. **Why ordinary baselines are insufficient**  
   Baseline LFR and static burden can correlate with difficult conditions, but
   they do not attribute which intervention changes failure behavior.

4. **Runtime replay closure**  
   Measured decoder service time can be turned into a paired deadline
   intervention without claiming live hardware queue traces.

5. **External and expanded evidence**  
   qec3v5 supports external real-record replication; Google RL QEC v2 supports
   broader same-family stability; the decoder-prior corpus supports a
   corpus-level intervention family.

This order reads much more like a systems paper than a dataset tour.

## Preferred Code Framing

Talk about the code in layers:

```text
layer 1: dataset-specific evidence builders
  32_run_qec3v5_external_replication.py
  35_run_google_rl_qec_v2_evidence.py
  36_run_google_decoder_priors_evidence.py

layer 2: canonical paper-facing aggregator
  31_run_p10_eurosys_main_evidence.py

layer 3: reviewer-facing summary
  37_build_p10_eurosys_digest.py
```

This keeps the repository legible to a systems audience: a small number of
explicit scripts, stable CSVs, and one canonical evidence pass.

## Preferred Claim Framing

Good EuroSys-facing wording looks like this:

```text
FailureOps does not propose a new decoder. It provides a paired debugging and
attribution interface for protected logical executions.
```

```text
The key question is not where errors occurred most often, but which
controllable intervention would have changed the observed logical failure
behavior on the same execution records.
```

```text
Measured runtime evidence is presented as replay-based deadline intervention
closure, not as live control-stack timing attribution.
```

## Reviewer Quickstart

Rebuild the default paper-facing outputs:

```bash
conda run -n failureops --no-capture-output python scripts/31_run_p10_eurosys_main_evidence.py
conda run -n failureops --no-capture-output python scripts/37_build_p10_eurosys_digest.py
```

Then inspect:

```text
data/results/p10_eurosys_digest.csv
data/results/p10_claim_audit.csv
figures/p10_google_rl_qec_v2_evidence.png
figures/p10_eurosys_main_evidence.png
```

## Boundary Discipline

Keep these boundaries explicit:

```text
supported:
  public real detector-record evaluation
  paired decoder-pathway and prior interventions
  measured-runtime replay
  qec3v5 external real-record replication

not claimed:
  live hardware feedback timing
  production control-stack queueing attribution
  hardware idle exposure caused by measured decoder latency
  new decoder design
  threshold estimation
```

That boundary discipline is part of what makes the package feel EuroSys-ready
rather than overstated.
