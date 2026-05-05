# Response to Reviewer wCX6

We thank the reviewer for the thoughtful feedback. We agree that the paper’s AI4Science positioning, the scope of the demonstrated benefit, and the distinction from prior shadow Hamiltonian simulation should have been stated more clearly. We address these points below.

## 1. AI4Science positioning and KDD relevance

We agree that the main contribution is an **error-certified reduced simulation framework** for scientific dynamical systems, rather than a learned surrogate in the narrow sense. Its AI4Science role is to provide a **compact, task-aligned, physically grounded, and certifiable representation** that can support downstream scientific learning.

To make this concrete, we added a **small downstream-learning probe** in the tested single-cosine regime. Using the same learner for both feature families—PCA to 16 dimensions followed by Ridge regression—we compared ShadowFluid’s reduced coherence representation $Z(t)$ against reconstructed low-pass density features for predicting the next-step unresolved high-frequency energy $E_{\\mathrm{HF}}(t+\\Delta t)$:

| Train size | Low-pass density MSE | Shadow coherence $Z$ MSE | Relative gain |
|---|---:|---:|---:|
| 16 | 2.369252e-7 | 2.015528e-7 | 14.9% |
| 32 | 2.316244e-7 | 1.676996e-7 | 27.6% |
| 64 | 2.596946e-7 | 1.509508e-7 | 41.9% |

Thus, under the same learner and matched feature dimension, using $Z(t)$ yields lower test MSE than reconstructed low-pass density features. This makes the AI4Science connection concrete: ShadowFluid is not only a reduced simulator, but also a certified reduced representation for downstream scientific learning.

## 2. Concretely demonstrated quantum implementation benefit

We respectfully clarify that the current paper is not purely classical. In addition to the classical validation of the error-control theory, we also **synthesized and transpiled** the full and reduced targets as quantum circuits and compared their compiled resource costs.

| Case | Full target | Reduced target | Qubits | 2Q gate count | 2Q gate depth |
|---|---|---|---:|---:|---:|
| State preparation | 128 amplitudes | 18 retained amplitudes packed into 32 | 7→5 | 268→55 | 260→55 |
| V = 0 time evolution | 64×64 active-mode unitary | 9×9 reduced unitary padded to 16×16 | 6→4 | 174→29 | 164→28 |

These quantum-circuit experiments show a concrete **implementation-level benefit** of the operator-first reduction: the reduced target yields materially smaller compiled quantum workloads. In particular, the substantial reductions in two-qubit gate count and two-qubit depth directly indicate fewer entangling operations and fewer sequential entangling layers after transpilation.

At the representation level, ShadowFluid replaces a full-state target of dimension $d = N^2$ by a task-aligned reduced object of size $M = |K||R|$. Accordingly, in the quantum-native operator-first workflow discussed in the paper, the relevant burden scales with task bandwidth and coupling-local reference structure rather than full grid resolution, and the measurement target is correspondingly reduced from $O(N^2)$ full-field quantities to $O(|K||R|)$ operator expectations. Thus, while we do not claim a hardware-level end-to-end quantum-advantage demonstration, it is more accurate to describe the current paper as combining classical theory validation with concrete quantum implementation evidence.

## 3. About novelty beyond prior shadow Hamiltonian simulation

We agree that the distinction from prior shadow Hamiltonian simulation should have been made more explicit. The key novelty is **not** simply applying a general shadow-Hamiltonian idea to fluid dynamics. Rather, prior shadow Hamiltonian simulation provides a reduced evolution principle under strict operator closure, whereas ShadowFluid turns this principle into a **task-driven and certifiable reduced-simulation method for fluid observables in the practically relevant approximate-closure regime**.

Concretely, ShadowFluid adds:

(1) a **task-driven multi-reference coherence dictionary** tailored to low-frequency fluid observables in the Fourier basis;

(2) a **coupling-graph-based dictionary construction** for PDE-style Fourier-space Hamiltonians;

(3) a **state-independent, a priori computable commutator-leakage certificate** for approximate closure, together with empirical validation of the hierarchy from computable bound to operator-level discrepancy to downstream task error on 2D Schrödinger-flow benchmarks.

In short, if Somma et al. establish the shadow-evolution principle under closure, ShadowFluid contributes the **task construction, approximate-closure treatment, and computable certification machinery** needed to make that principle usable for reduced quantum fluid simulation.

Thank you for your insightful suggestions.