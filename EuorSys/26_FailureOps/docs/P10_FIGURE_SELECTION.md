# P10 Figure Selection

This note recommends one paper main figure and one supplementary figure from
the current P10 evidence pass.

The goal is not to choose the busiest figure. The goal is to choose the figure
that most directly supports the EuroSys story:

```text
paired intervention-sensitive failure behavior exists on public real detector records
pairing is methodologically necessary
the signal remains stable across a broader real-record corpus
```

## Recommended Main Figure

**Figure file**

```text
figures/p10_google_rl_qec_v2_evidence.png
```

**Why this should be the main figure**

1. It is the strongest single empirical picture of the central claim on the
   largest public real-record corpus currently in the repository.
2. It combines the two most EuroSys-relevant messages in one place:
   paired logical-failure sensitivity and pairing advantage.
3. It keeps the paper honest about scope: this is expanded same-family public
   real-record evidence, not an independent-lab replication claim.

**What the figure is saying**

- The expanded Google RL QEC v2 corpus covers `496` conditions and
  `4.96M` paired shots.
- The overall mean paired delta LFR is `-0.013661`.
- `461/496` conditions are net-rescuing.
- The mean unpaired/paired bootstrap standard-deviation ratio is `1.783838`.
- Mean paired delta remains negative in both observed control modes and both
  logical bases:
  - `traditional_calibration`: `-0.014388`
  - `traditional_calibration_and_rl_fine_tuning`: `-0.012934`
  - `X`: `-0.011489`
  - `Z`: `-0.015833`

**Suggested caption**

```text
Figure X: Paired decoder-pathway sensitivity on the expanded Google RL QEC v2 real-record corpus. Across 496 public real-record conditions (4.96M paired shots), switching from the SI1000-prior decoder pathway to the frequency-calibrated-prior pathway yields a broadly net-rescuing paired effect: the overall mean paired delta logical-failure rate is -0.0137, and 461/496 conditions are net-rescuing. The top-left panel shows that mean paired sensitivity generally strengthens with longer memory duration, while the top-right panel shows that pairing continues to reduce estimator variance across the same cycle regimes (mean unpaired/paired bootstrap standard-deviation ratio 1.78). The bottom-left panel aggregates effects across experiment families and shows negative mean paired delta in both code families and both observed control modes. The bottom-right panel summarizes rescued, induced, and unchanged transition profiles, highlighting that the observed effect is not explained by static burden alone. We treat this figure as expanded same-family public real-record evidence rather than independent external replication.
```

## Recommended Supplementary Figure

**Figure file**

```text
figures/p10_eurosys_main_evidence.png
```

**Why this should be the supplementary figure**

1. It is valuable, but it is more of a paper dashboard than a first-punch main
   figure.
2. Its strength is breadth: baseline comparison, subgroup robustness,
   measured-runtime replay, and focused prior interventions.
3. It supports the full systems story after the reader already accepts the
   main paired-sensitivity result.

**What the figure is saying**

- Static burden and baseline LFR correlate with FailureOps rankings
  (`Spearman = 0.949906` and `0.962664`) but still lack rescued/induced
  attribution semantics.
- On the original real-record matrix, mean sensitivity becomes stronger with
  cycle depth (`slope = -0.003111` per 10 cycles; `corr = -0.989246`).
- Measured runtime replay closes into paired deadline interventions:
  a `4 us` deadline miss rate of `0.936` induces a paired delta LFR of
  `+0.3072`, while the effect vanishes by `7 us`.
- In the focused prior sweep, `3/5` variants exclude zero by bootstrap CI.

**Suggested caption**

```text
Figure S1: Additional P10 evidence on attribution semantics, subgroup robustness, measured-runtime replay, and focused prior interventions. The top-left panel shows that static detector burden and plain baseline logical-failure rate can rank difficult conditions similarly to FailureOps, but they do not recover rescued versus induced paired transitions and therefore do not provide intervention attribution. The top-right panel shows that mean paired sensitivity becomes stronger with cycle depth and remains broadly stable across control-mode and logical-basis subgroups on the original real-record condition matrix. The bottom-left panel closes measured decoder service-time traces into paired runtime deadline interventions: aggressive deadlines sharply increase logical failure, while the effect disappears once the deadline exceeds the observed service-time tail. The bottom-right panel shows that a focused prior sweep produces nontrivial paired effects, with 3 of 5 prior variants excluding zero. This figure is best used as supporting evidence for the broader systems story after the main real-record paired-sensitivity result is established.
```

## Recommendation Summary

If the paper gets only one high-attention results figure, use:

```text
main: figures/p10_google_rl_qec_v2_evidence.png
```

If the appendix or supplementary material gets one compact support figure, use:

```text
supplement: figures/p10_eurosys_main_evidence.png
```

This split keeps the front of the paper centered on the strongest real-record
paired-sensitivity result, while moving the denser multi-claim support figure
into a role where its breadth helps instead of competing for attention.
