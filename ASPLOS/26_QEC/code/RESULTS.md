# Current Results

This note summarizes the current 6-code repeated run produced by:

```bash
python scripts/01_build_bbcode.py
python scripts/02_code_capacity_eval.py
python scripts/03_plot_results.py
python scripts/04_analyze_decoder_deltas.py
```

## Repeated-Run Setup

- BB instances: `bb_n64_k10`, `bb_n72_k12`, `bb_n80_k6`, `bb_n96_k10`, `bb_n108_k4`, `bb_n144_k12`
- decoders: `BP-OSD`, `BP-LSD`
- noise model: independent Z-only code-capacity noise
- sweep: `p in [0.01, 0.02, 0.03, 0.05]`
- repeats: `5`
- shots per repeat: `1000`
- total repeated rows: `6 codes x 4 p x 2 decoders x 5 repeats = 240`

Main output files:

- `data/results/results_repeated.csv`
- `data/results/results_repeated_summary.csv`
- `data/results/decoder_delta_summary.csv`
- `data/results/code_level_summary.csv`
- `data/results/ler_vs_p_with_ci.png`
- `data/results/compare_codes_p95_latency_with_errorbars.png`
- `data/results/delta_p95_latency_by_code.png`

## Most Useful Figures

- `ler_vs_p_with_ci.png`
  This is the quickest check that `BP-LSD` still does not separate from `BP-OSD` in LER.
- `compare_codes_p95_latency_with_errorbars.png`
  This is the main cross-instance tail-latency figure.
- `delta_p95_latency_by_code.png`
  This is the clearest direct view of `BP-LSD - BP-OSD` for runtime.

## What The 6-Code Run Shows

- Across all 6 small BB instances, `BP-OSD` and `BP-LSD` remain effectively tied in LER at the current resolution.
- The code-level summary does not show a meaningful mean LER advantage for `BP-LSD` on any tested instance.
- The mean `delta_p95_latency_over_p` is positive for all 6 instances, so the cross-instance average still points in the same direction: `BP-LSD` is usually slower.

From `data/results/code_level_summary.csv`:

- `bb_n72_k12`: mean `delta_p95_latency_over_p = +0.0215 ms`
- `bb_n96_k10`: mean `delta_p95_latency_over_p = +0.0363 ms`
- `bb_n144_k12`: mean `delta_p95_latency_over_p = +0.0573 ms`
- `bb_n64_k10`: mean `delta_p95_latency_over_p = +0.0249 ms`
- `bb_n80_k6`: mean `delta_p95_latency_over_p = +0.0309 ms`
- `bb_n108_k4`: mean `delta_p95_latency_over_p = +0.0740 ms`

This means the earlier runtime-tradeoff conclusion still holds after expanding from 3 codes to 6.

## Which Instance Looks Most Like A Counterexample

`bb_n80_k6` is the closest thing to a counterexample, but it is not a strong one.

- At `p = 0.03`, its mean `delta_p95_latency` is slightly negative.
- The magnitude is very small, and repeat-to-repeat sign is mixed.
- At `p = 0.05`, the same code shows a clear positive `delta_p95_latency`, with `BP-LSD` slower again.
- Its code-level average over all 4 `p` points is still positive: `+0.0309 ms`.

So `bb_n80_k6` is better described as a near-tie with one locally noisy point, not as a stable reversal of the decoder ordering.

## What Happened To `bb_n144_k12`

`bb_n144_k12` was the most suspicious case in the earlier 3-code run. In the current broader run it no longer looks like a real counterexample.

- At low `p`, it is close to a tie.
- At `p = 0.05`, `BP-LSD` is clearly slower in both average and p95 latency.
- Its code-level mean `delta_p95_latency_over_p` is `+0.0573 ms`.

The current reading is: `bb_n144_k12` is not showing a stable latency inversion. It is showing low-`p` near-tie behavior and high-`p` slowdown for `BP-LSD`.

## Lightweight Structural Read

The current artifacts include simple structure fields such as `n`, matrix shape, density, and syndrome-weight statistics.

The useful facts are:

- There is no clean one-feature rule such as “denser checks always mean larger LSD penalty” or “larger `n` alone explains everything”.
- `bb_n108_k4` shows the largest mean p95 penalty even though it is not the densest code.
- `bb_n144_k12` is sparser than the smaller codes, but still shows a substantial positive p95 penalty.
- Average BP iteration counts remain close between `BP-OSD` and `BP-LSD` across the tested instances.

The practical interpretation is modest:

- The runtime penalty does not look driven by extra BP iterations.
- It is more consistent with extra post-BP work in the LSD stage.
- Heavier syndrome regimes at larger or harder points seem to make the penalty easier to see, but the relationship is not strictly monotone in `n`, density, or matrix shape.

## Current Conclusion

- The most credible summary at this point is:
  - on these 6 small BB code-capacity instances, `BP-LSD` does not show a visible LER advantage over `BP-OSD`;
  - the runtime penalty remains positive on average for every tested code;
  - the earlier “mixed latency” cases shrink to near-ties or local noise rather than stable reversals.

## What To Do Next

- It still makes sense to stay in the code-capacity setting.
- The current evidence is enough to justify a small systems-style benchmark expansion within code-capacity, because the basic runtime tradeoff now holds on a broader small-BB family instead of only 3 hand-picked instances.
- The most natural next step is still incremental:
  - add a few more small BB instances or
  - keep the 6 current codes and increase repeats / seeds.

That is a better next move than jumping to circuit-level noise before the current runtime trend is fully stabilized.
