# Scripts

This directory is intentionally split into two layers.

## Active entry points

These are the scripts that define the current public surface of the repository:

```text
01_generate_runs.py
02_apply_interventions.py
03_analyze_attribution.py
04_plot_results.py
31_run_p10_eurosys_main_evidence.py
32_run_qec3v5_external_replication.py
33_run_external_sanity_checks.py
35_run_google_rl_qec_v2_evidence.py
36_run_google_decoder_priors_evidence.py
37_build_p10_eurosys_digest.py
```

Use these if you are:

```text
trying the toy FailureOps loop
rebuilding the current P10 EuroSys-facing evidence pass
inspecting the paper-facing outputs
```

## Archived milestone scripts

Older milestone, intermediate rebuild, and historical analysis scripts now live
under:

```text
scripts/archive/
```

They are kept for:

```text
historical reproducibility
milestone-specific debugging
rebuilding upstream artifacts when needed
```

They are not the recommended starting point for new users.

## Common paths

Rebuild the main P10 package:

```bash
conda run -n failureops --no-capture-output python scripts/31_run_p10_eurosys_main_evidence.py
conda run -n failureops --no-capture-output python scripts/37_build_p10_eurosys_digest.py
```

Rebuild the larger public real-record evidence:

```bash
conda run -n failureops --no-capture-output python scripts/35_run_google_rl_qec_v2_evidence.py
conda run -n failureops --no-capture-output python scripts/36_run_google_decoder_priors_evidence.py
conda run -n failureops --no-capture-output python scripts/32_run_qec3v5_external_replication.py
```

Rebuild archived upstream P7.5/P8 artifacts:

```bash
conda run -n failureops --no-capture-output python scripts/archive/27_run_p7_sweep.py
conda run -n failureops --no-capture-output python scripts/archive/28_run_p7_5_analysis.py
conda run -n failureops --no-capture-output python scripts/archive/29_measure_p8_decoder_runtime.py
```

Some older milestone docs still discuss these archived scripts conceptually. If
you see an old command rooted at `scripts/XX_...`, the corresponding file may
now live at `scripts/archive/XX_...`.
