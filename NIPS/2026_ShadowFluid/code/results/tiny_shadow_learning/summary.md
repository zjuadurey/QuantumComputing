# Tiny Shadow Learning Probe

Prepared for Reviewer oB6r.

## Benchmark

- Single-cosine potential with `q = (1, 0)`.
- Grid: `N = 16`.
- Cutoff: `K0 = 4`.
- Couplings: `alpha in {0.2, 0.6, 1}`.
- Time points: `t in {0, 0.2, 0.4, 0.6}`, with `dt = 0.2`.
- Initial-condition family: Gaussian vortex only (multiple deterministic seeds inside the same family).
- Total supervised samples: `96`.

## Features and Target

- Shadow feature type: flattened real/imaginary parts of the reduced coherence matrices `Z_1(t)` and `Z_2(t)` built from the existing ShadowFluid reduced evolution.
- Low-pass baseline feature type: the task-level low-pass density field at the same cutoff `K0`.
- Downstream target: next-step unresolved high-frequency energy `E_HF(t+dt)`.

## Learner

- Same learner for both feature families: `PCA(feature_dim <= 16) + Ridge(alpha=1)`.
- Train-size sweep: `[16, 32, 64]`.
- Splits: `3` deterministic random splits with shared train/test partitions across feature families.
- Cached dataset: `dataset.npz` with sample manifest `dataset_manifest.csv`.

## Main Result Table

| Train size | Feature type | Learner | Feature dim | Test MSE | Std | Relative gain vs low-pass |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | Task-level low-pass density | PCA + Ridge | 16 | 2.369252e-07 | 3.520957e-09 | 0 |
| 16 | Shadow coherence Z | PCA + Ridge | 16 | 2.015528e-07 | 7.443794e-09 | 0.149298 |
| 32 | Task-level low-pass density | PCA + Ridge | 16 | 2.316244e-07 | 1.431862e-08 | 0 |
| 32 | Shadow coherence Z | PCA + Ridge | 16 | 1.676996e-07 | 1.573179e-08 | 0.275985 |
| 64 | Task-level low-pass density | PCA + Ridge | 16 | 2.596946e-07 | 2.087637e-08 | 0 |
| 64 | Shadow coherence Z | PCA + Ridge | 16 | 1.509508e-07 | 1.257508e-08 | 0.418737 |

## Short Interpretation

In a small downstream-learning probe prepared for Reviewer oB6r, we compared two matched-dimensional input representations under the same PCA + Ridge learner: (i) coherence-aware Shadow features obtained from the reduced `Z(t)` representation and (ii) task-level low-pass density inputs at the same cutoff `K0 = 4`. The downstream target was the next-step unresolved high-frequency energy `E_HF(t+dt)`, which is not directly available from the task truncation. On the single-cosine benchmark with `N = 16`, `q = (1,0)`, `alpha in {0.2, 0.6, 1}`, and `t in {0, 0.2, 0.4, 0.6}`, in the tested regime, the coherence-aware Shadow features achieve lower test MSE than the task-level low-pass density inputs at all reported train sizes. Across train sizes `[16, 32, 64]`, the relative gain of the Shadow features ranges from 0.149298 to 0.418737. This small probe suggests that ShadowFluid can act as a compact coherence-aware frontend for downstream learning, while remaining far from a full AI benchmark.
