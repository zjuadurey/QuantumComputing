# Response to Reviewer oB6r

We thank the reviewer for the thoughtful feedback. We address the three main concerns below.

## 1. On the current scope and the role of linear Schrödinger dynamics

We agree that the current paper is restricted to **linear Schrödinger-type dynamics**, and we will make this scope clearer in the revision. At the same time, this does **not** mean the framework is limited to trivial irrotational flows. As stated in Section 4.1 of the manuscript, our flow encoding uses a **two-component spinor**, so the velocity field can carry **nonzero vorticity**; only a **single-component wavefunction** is restricted to irrotational flow.

We also agree that extending beyond the current linear setting is important. However, this is a broader challenge for quantum fluid simulation rather than a paper-specific omission. Meng et al. [6] likewise handle the vortex example through a **linear two-component Schrödinger equation**, while more general body forces, especially Newtonian-fluid dissipation, return to nonlinear equations that remain open for efficient quantum treatment.

## 2. About Fourier-sparse cosine potentials and the generality of the coupling-graph construction

We agree that the original experiments should better show what happens when the Hamiltonian becomes less structured. We therefore added a **structured multi-component stress test** that progressively densifies the Fourier coupling graph.

Under alpha_tot = 0.5, K0 = 5, t = 0.5, and R_hops = 1, increasing the number of Fourier components from J = 1 to J = 8 enlarges the reference set from 3 to 17 and the reduced size from 243 to 1377, while the density error remains near 0.03:

| Fourier components J | R size | Reduced size | Density error | Delta Z_F | Leakage l_rms |
|---|---:|---:|---:|---:|---:|
| 1 | 3 | 243 | 0.029049 | 0.007733 | 0.242161 |
| 4 | 9 | 729 | 0.030072 | 0.001924 | 0.137493 |
| 8 | 17 | 1377 | 0.029789 | 0.003565 | 0.101099 |

We also quantified the closure-depth tradeoff on the densest case:

| Closure depth R_hops | R size | Reduced size | Density error | Delta Z_F | Leakage l_rms |
|---|---:|---:|---:|---:|---:|
| 1 | 17 | 1377 | 0.029789 | 0.003565 | 0.101099 |
| 3 | 115 | 9315 | 0.029789 | 0.000478 | 0.077749 |

These results clarify the regime boundary: denser couplings do shrink the dimensionality advantage by enlarging R, but do not automatically destroy approximation quality in the tested regime. Deeper closure improves operator fidelity, but at a substantial cost in reduced size. In the revision, we will state more explicitly that the current construction is validated for **structured Fourier-coupled regimes**.

## 3. KDD/ML relevance and the relation to learning-based AI4Science workflows

We agree that the current paper is not a learned surrogate or data-mining paper in the narrow sense. Our contribution is an **error-certified reduced simulation framework** for scientific modeling. At the same time, its AI4Science relevance is concrete: ShadowFluid produces a compact, task-aligned, and physically grounded reduced representation that can interface with downstream learning.

Following the reviewer’s suggestion in a lightweight form, we added a **small downstream-learning probe** in the tested single-cosine regime. Using the same learner for both feature families—PCA to 16 dimensions followed by Ridge regression—we compared reduced coherence features Z(t) against reconstructed task-level low-pass density features for predicting the next-step unresolved high-frequency energy E_HF(t + Delta t).

| Train size | Low-pass MSE | Shadow Z MSE | Gain |
|---|---:|---:|---:|
| 16 | 2.369252e-7 | 2.015528e-7 | 14.9% |
| 32 | 2.316244e-7 | 1.676996e-7 | 27.6% |
| 64 | 2.596946e-7 | 1.509508e-7 | 41.9% |

Across all reported train sizes, using Z(t) as features consistently yields lower test MSE than reconstructed task-level low-pass density features.

Thus, in the tested regime, the reduced representation produced by ShadowFluid is directly useful as a downstream learning feature layer. This is **not** a full ML benchmark or neural PDE surrogate comparison, but it shows that ShadowFluid can serve as a **physically constrained feature extractor**.

Thanks again for your constructive comments.