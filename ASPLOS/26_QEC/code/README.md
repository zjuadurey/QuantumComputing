# Minimal BBCode Code-Capacity Loop

This directory is a minimal, runnable check that the `qldpc + ldpc` software path for a small bivariate bicycle (BB) code works end to end.

Current scope:

- 6 small BB code instances
- 1 simple code-capacity noise model
- 2 baseline decoders: `BP-OSD` and `BP-LSD`
- Wilson confidence intervals for LER
- average / p50 / p95 / p99 decoder latency
- multi-code comparison figures
- 1 latency CSV

This is intentionally narrow. It does not do circuit-level noise, routing, FPGA work, or large family benchmarks.

## Environment

Create a Python 3.10 environment:

```bash
conda create -n qec_py310 python=3.10 -y
conda activate qec_py310
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Notes:

- The scripts automatically set local writable cache directories for `qldpc/galois/numba` and `matplotlib`, because those imports can fail in shared environments if cache paths are not writable.
- `stim` and `sinter` are included in the environment because they are part of the intended stack, but this first minimal code-capacity loop decodes directly from parity-check matrices and does not yet need circuit generation.

## Scripts

- `scripts/01_build_bbcode.py`
  Builds 6 small BB code instances with `qldpc.codes.BBCode`, exports `Hx`, `Hz`, logical operators, per-code metadata, and a manifest.
- `scripts/02_code_capacity_eval.py`
  Loads the saved code artifacts, runs a Z-only code-capacity Monte Carlo sweep over `p in [0.01, 0.02, 0.03, 0.05]`, and evaluates `BP-OSD` plus `BP-LSD`.
- `scripts/03_plot_results.py`
  Reads `data/results/results.csv` and saves per-code plots plus cross-code comparison plots.
- `scripts/04_analyze_decoder_deltas.py`
  Builds paired `BP-LSD - BP-OSD` delta tables and a delta latency plot from repeated runs.

## Code Instances

The current build script emits these small instances, all aligned with `qldpc` BBCode examples or tests:

```text
bb_n64_k10:
  orders = (8, 4)
  poly_a = 1 + y + x*y + x^5
  poly_b = 1 + x^2 + x^3 + x^5*y^2

bb_n72_k12:
  orders = (6, 6)
  poly_a = x^3 + y + y^2
  poly_b = y^3 + x + x^2

bb_n80_k6:
  orders = (8, 5)
  poly_a = 1 + y + x*y + x^5
  poly_b = 1 + x^2 + x^3 + x^5*y^2

bb_n96_k10:
  orders = (12, 4)
  poly_a = 1 + y + x*y + x^9
  poly_b = 1 + x^2 + x^7 + x^9*y^2

bb_n108_k4:
  orders = (9, 6)
  poly_a = 1 + y + x*y + x^6
  poly_b = 1 + x^2 + x^4 + x^6*y^2

bb_n144_k12:
  orders = (12, 6)
  poly_a = x^3 + y + y^2
  poly_b = y^3 + x + x^2
```

The current metadata records:

- `bb_n64_k10`: `n = 64`, `k = 10`, `d = null`
- `bb_n72_k12`: `n = 72`, `k = 12`, `d = 6`
- `bb_n80_k6`: `n = 80`, `k = 6`, `d = null`
- `bb_n96_k10`: `n = 96`, `k = 10`, `d = null`
- `bb_n108_k4`: `n = 108`, `k = 4`, `d = null`
- `bb_n144_k12`: `n = 144`, `k = 12`, `d = 12`

When `qldpc` does not provide a runtime distance, the scripts leave `d` empty unless the example already appears in `qldpc`'s own test comments with a known `[[n, k, d]]` label.

## End-To-End Run

From this directory:

```bash
conda activate qec_py310
python scripts/01_build_bbcode.py
python scripts/02_code_capacity_eval.py
python scripts/03_plot_results.py
python scripts/04_analyze_decoder_deltas.py
```

If you want a slightly larger sweep:

```bash
python scripts/02_code_capacity_eval.py --shots 1000
```

## Expected Outputs

- `data/codes/bbcode_manifest.json`
- `data/codes/bbcode_n64_k10.npz`
- `data/codes/bbcode_n64_k10.json`
- `data/codes/bbcode_n72_k12.npz`
- `data/codes/bbcode_n72_k12.json`
- `data/codes/bbcode_n80_k6.npz`
- `data/codes/bbcode_n80_k6.json`
- `data/codes/bbcode_n96_k10.npz`
- `data/codes/bbcode_n96_k10.json`
- `data/codes/bbcode_n108_k4.npz`
- `data/codes/bbcode_n108_k4.json`
- `data/codes/bbcode_n144_k12.npz`
- `data/codes/bbcode_n144_k12.json`
- `data/results/results.csv`
- `data/results/results_repeated.csv`
- `data/results/results_repeated_summary.csv`
- `data/results/decoder_delta_summary.csv`
- `data/results/code_level_summary.csv`
- `data/results/ler_vs_p_with_ci.png`
- `data/results/latency_avg_vs_p.png`
- `data/results/latency_p95_vs_p.png`
- `data/results/compare_codes_ler_by_decoder.png`
- `data/results/compare_codes_p95_latency_by_decoder.png`
- `data/results/compare_codes_p95_latency_with_errorbars.png`
- `data/results/bb_n144_latency_focus.png`
- `data/results/delta_p95_latency_by_code.png`

## Result Columns

`data/results/results.csv` contains:

- `code_name`
- `p`
- `decoder_name`
- `repeat_id`
- `num_shots`
- `num_failures`
- `logical_error_rate`
- `ler_ci_low`
- `ler_ci_high`
- `avg_latency_ms_per_shot`
- `p50_latency_ms_per_shot`
- `p95_latency_ms_per_shot`
- `p99_latency_ms_per_shot`
- `n`
- `k`
- `d`
- `mx_rows`
- `mx_cols`
- `mz_rows`
- `mz_cols`

## Worth Continuing?

- Minimal loop status: ran successfully in this checkout for 6 BB instances. The current run completed `build -> eval -> plot` and produced the updated CSV plus CI/tail-latency figures.
- Current signal: across these 6 small code-capacity runs, `BP-OSD` and `BP-LSD` have identical or near-identical LER within the sampled resolution. The code-level summary shows positive mean `BP-LSD - BP-OSD` p95 latency on all 6 instances, so the runtime penalty remains the more stable effect.
- Most likely blocker: `qldpc` import/cache quirks and the exact logical-failure definition for CSS decoding. The current scripts handle both explicitly.
- Stack outlook:
  - Multiple BB family comparisons: yes, this stack is already enough.
  - Circuit-level / routing: not yet; that likely needs a stronger `stim/sinter` and circuit-construction layer on top.
  - Systems benchmark work: possible later, but only after several code families and decoder/runtime settings are stable.
- Natural next step after this loop: add a few more BB instances or rerun the current six with more seeds / repeats, then decide if the latency trend is stable enough to justify broader code-capacity benchmarking.
