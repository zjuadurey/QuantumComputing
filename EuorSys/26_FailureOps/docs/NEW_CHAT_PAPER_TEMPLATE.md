# New Chat Paper Template

Use this template when starting a fresh chat for **paper writing based on the
current FailureOps results**, without changing the codebase unless a clear bug
is discovered.

## Recommended Template

Copy the block below into a new chat:

```text
We are working in the FailureOps repository at:
/home/adurey/QuantumComputing/EuorSys/26_FailureOps

Goal for this chat:
Use the current data and figures to write the paper. Do not modify the codebase,
script layout, statistical framing, or CSV schemas unless I explicitly ask, or
unless you find a clear bug that blocks a paper claim.

Before doing anything else, read these files first:
- AGENTS.md
- docs/START_HERE.md
- docs/P10.md
- docs/P10_FIGURE_SELECTION.md
- docs/P10_EUROSYS_STYLE.md

Then read these current canonical result files:
- data/results/p10_claim_audit.csv
- data/results/p10_realdata_robustness.csv
- data/results/p10_eurosys_digest.csv

If needed for exact numbers, also inspect:
- data/results/p10_baseline_comparison.csv
- data/results/p10_effect_significance.csv
- data/results/p10_runtime_deadline_summary.csv
- data/results/p10_google_decoder_priors_prior_effects_aggregate.csv
- data/results/p10_google_rl_qec_v2_decoder_effect_aggregate.csv

Use the current repository state as frozen unless I explicitly ask for code
changes. Assume the scripts/ layout has already been cleaned:
- active public entry points remain in scripts/
- historical milestone rebuilders are in scripts/archive/

Important claim boundaries:
- supported:
  - paired FailureOps methodology
  - public real detector-record evaluation
  - decoder-pathway and prior interventions
  - measured decoder-runtime replay
  - qec3v5 external real-record replication
  - Google RL QEC v2 same-family expansion evidence
- not claimed:
  - live hardware feedback timing
  - production control-stack queueing attribution
  - hardware idle exposure caused by measured decoder latency
  - new decoder design
  - threshold estimation
  - independent non-Google shot-level replication

Current paper-facing statistical framing:
- scope-wise Holm correction by analysis_scope
- default P10 prior input is the full Google decoder-prior corpus
- canonical paper-facing outputs are:
  - data/results/p10_claim_audit.csv
  - data/results/p10_realdata_robustness.csv
  - data/results/p10_eurosys_digest.csv
  - figures/p10_google_rl_qec_v2_evidence.png
  - figures/p10_eurosys_main_evidence.png

When answering, prioritize:
1. paper framing
2. contribution wording
3. abstract / intro / figure captions / limitations
4. exact claim wording aligned with the current CSV outputs

Please start by summarizing:
1. the core systems thesis,
2. the strongest supported claims,
3. the boundaries we must not overclaim,
4. and a recommended outline for the next writing step.
```

## Short Version

Use this shorter version if you want to get moving quickly:

```text
Please help me write the FailureOps paper from the current repository state, not
by changing code. First read:
AGENTS.md
docs/START_HERE.md
docs/P10.md
docs/P10_FIGURE_SELECTION.md
docs/P10_EUROSYS_STYLE.md

Then use:
data/results/p10_claim_audit.csv
data/results/p10_realdata_robustness.csv
data/results/p10_eurosys_digest.csv

Treat the current code and statistical framing as frozen unless there is a
clear bug. Keep all wording aligned with the current P10 outputs and their
claim boundaries. Start by summarizing the strongest paper claims and proposing
the next writing step.
```

## Best Use

This template works best when you want help with:

```text
abstract drafting
intro opening paragraphs
contribution lists
figure caption writing
claim tightening
limitations / threats-to-validity wording
result-section organization
```

It is not the right template for:

```text
changing experiment logic
adding new datasets
rewriting the pipeline
reopening statistical framing
major repository refactoring
```
