# Response to Reviewer wCX6

We thank the reviewer for the thoughtful feedback and for the helpful comments on the paper’s positioning, the scope of the current validation and measurement-reduction discussion, and the need to distinguish ShadowFluid more clearly from prior shadow Hamiltonian simulation. We address these points below.

## 1. On the paper’s AI4Science positioning and KDD-track relevance

We agree that the submitted version did not include a trained AI component, and that this boundary was not stated sharply enough. Our main contribution remains an **error-certified reduced simulation framework** for scientific dynamical systems, rather than a learned surrogate in the narrow sense.

To make the AI4Science connection more concrete, we added a **small downstream-learning probe** in the tested single-cosine regime. In this probe, we use the same learner for both feature families---PCA to $16$ dimensions followed by Ridge regression---and compare **coherence-aware Shadow features** against **task-level low-pass density inputs** for predicting the next-step unresolved high-frequency energy $E_{\mathrm{HF}}(t+\Delta t)$, which is **not** directly available from the task truncation:

Train size|Low-pass density MSE|Shadow coherence $Z$ MSE|Relative gain of Shadow
|---|---:|---:|---:|
16|$2.369252\times 10^{-7}$|$2.015528\times 10^{-7}$|14.9\%
32|$2.316244\times 10^{-7}$|$1.676996\times 10^{-7}$|27.6\%
64|$2.596946\times 10^{-7}$|$1.509508\times 10^{-7}$|41.9\%

Thus, in the tested regime, under the same downstream learner and matched feature dimension, the coherence-aware Shadow representation achieves **lower test MSE** than task-level low-pass inputs on a downstream unresolved scalar quantity. We emphasize that this is **not** a full AI benchmark; rather, it is a concrete hybrid-learning example showing that ShadowFluid can serve as a **compact coherence-aware frontend for downstream learning**.

In the revision, we will sharpen this positioning in the introduction and discussion: the main contribution remains an error-certified reduced simulation framework, while the added probe provides a concrete AI4Science-relevant integration beyond a future-work statement.

## 2. On classical-only validation and the scope of the measurement-reduction claim

We agree that the current paper does not present a hardware-level demonstration of quantum advantage. Our intended claim is narrower: ShadowFluid replaces full-state evolution with reduced operator dynamics under exact closure, and provides controlled truncation with an *a priori* computable leakage certificate under approximate closure. The discussion of measurement reduction is therefore meant as a deployment motivation in the quantum-native regime, rather than as an experimentally demonstrated end-to-end advantage in this paper.

Beyond the classical matrix-form implementation reported in the manuscript, we also implemented a noiseless Qiskit-based circuit realization and used it as an implementation-level cross-check against the classical version. The two agree up to very small numerical discrepancy, providing an implementation-level consistency check. At the same time, this does **not** constitute a hardware-level validation of quantum advantage.

We will revise the abstract, discussion, and limitations sections to state this scope more explicitly.

## 3. On novelty beyond prior shadow Hamiltonian simulation

We agree that the distinction from prior shadow Hamiltonian simulation should have been made more explicit. Our contribution is not simply to apply a general shadow-Hamiltonian idea to a new domain, but to turn it into a task-driven and certifiable methodology for approximate closure in quantum fluid observables. Relative to the prior framework, the present paper contributes:  
(1) a **task-driven multi-reference coherence dictionary** tailored to low-frequency fluid observables in the Fourier basis;  
(2) a **coupling-graph-based construction** of the reference set, tied to PDE-style Fourier-space Hamiltonians;  
(3) an extension from exact closure to the practically relevant **approximate-closure regime**; and  
(4) a **state-independent, \textit{a priori} computable commutator-leakage quantity**, together with empirical validation of the hierarchy between the computable bound, the operator-level discrepancy, and the downstream task error on $2$D Schrödinger-flow benchmarks.

In the revision, we will strengthen the related-work and contribution sections to make this distinction much more explicit.

We thank the reviewer again for the constructive feedback. These clarifications sharpen the intended scope of the paper and better distinguish ShadowFluid from prior shadow Hamiltonian simulation.

---

# Response to Reviewer oB6r

We thank the reviewer for the thoughtful feedback and for recognizing the motivation of the problem, the elegance of the operator-first framework, the value of the error analysis, and the clarity of the related-work discussion. We also appreciate the reviewer’s comments on scope, benchmark simplicity, and KDD/ML relevance. We address these points below.

## 1. On the current scope being limited to linear Schrödinger dynamics

We agree that the current paper is restricted to **linear Schrödinger-type dynamics** and does not claim to solve general nonlinear CFD or Navier--Stokes dynamics. Our goal is narrower: to establish an **error-certified reduced simulation framework** in a setting where closure structure and truncation behavior can be analyzed rigorously. We will sharpen this boundary in the introduction, discussion, and limitations sections.

## 2. On the use of Fourier-sparse cosine potentials and the generality of the coupling-graph construction

We agree that the original experiments should do more to show what happens as the Hamiltonian becomes less structured. To address this point, we added a **structured multi-component stress test** that progressively densifies the coupling graph.

In the fixed-total-coupling setting $(\alpha_{\mathrm{tot}}=0.5, K_0=5, t=0.5, R_{\mathrm{hops}}=1)$, increasing the number of Fourier components from $J=1$ to $J=8$ enlarges the reference set from $|R|=3$ to $17$ and the reduced size from $243$ to $1377$, while the density error remains near $3\times 10^{-2}$:

Fourier components $J$|R size|Reduced size|Density error|$\Delta Z_F$|Leakage $\ell_{\mathrm{rms}}$
|---|---:|---:|---:|---:|---:|
1|3|243|0.029049|0.007733|0.242161
4|9|729|0.030072|0.001924|0.137493
8|17|1377|0.029789|0.003565|0.101099

We also quantified the closure-depth tradeoff on the densest structured case $(J=8)$:

Closure depth $R_{\mathrm{hops}}$|R size|Reduced size|Density error|$\Delta Z_F$|Leakage $\ell_{\mathrm{rms}}$
|---|---:|---:|---:|---:|---:|
1|17|1377|0.029789|0.003565|0.101099
3|115|9315|0.029789|0.000478|0.077749

These added results clarify the regime boundary: denser couplings do shrink the dimensionality advantage by enlarging $R$, but do not automatically destroy approximation quality in the tested regime. In particular, deeper closure improves operator fidelity, but at a clearly visible cost in reduced size. In the revision, we will state more explicitly that the current construction is validated for **structured Fourier-coupled regimes**, rather than claimed to cover fully broadband or non-periodic Hamiltonians.

## 3. On KDD/ML relevance and the relation to learning-based AI4Science pipelines

We agree that the current paper is not a learned-surrogate or data-mining paper in the narrow sense. Our intended contribution is an **error-certified reduced simulation framework** for scientific modeling.

To make the AI/ML connection more concrete, we added a **small downstream-learning probe** in the tested single-cosine regime. Using the same learner (PCA to $16$ dimensions + Ridge), we compared **Shadow coherence $Z$ features** against **task-level low-pass density inputs** for predicting the next-step unresolved high-frequency energy $E_{\mathrm{HF}}(t+\Delta t)$, which is **not** directly available from the task truncation. The probe uses the same single-cosine family with $q=(1,0)$, $N=16$, $K_0=4$, $\alpha\in\{0.2,0.6,1.0\}$, $t\in\{0,0.2,0.4,0.6\}$, and one Gaussian-vortex initial-condition family.

Train size|Low-pass MSE|Shadow MSE|Gain
|---|---:|---:|---:|
16|$2.369252\times 10^{-7}$|$2.015528\times 10^{-7}$|14.9\%
32|$2.316244\times 10^{-7}$|$1.676996\times 10^{-7}$|27.6\%
64|$2.596946\times 10^{-7}$|$1.509508\times 10^{-7}$|41.9\%

The same direction also holds at the split level: across all $3$ random splits and all reported train sizes, the Shadow features consistently achieve lower test MSE than the task-level low-pass inputs.

Thus, in the tested regime, under the same downstream learner and matched feature dimension, the coherence-aware Shadow representation achieves **lower test MSE** than task-level low-pass inputs on this downstream unresolved prediction task. We stress that this is **not** a full ML benchmark, but a concrete hybrid-learning example showing that ShadowFluid can serve as a **compact coherence-aware frontend for downstream learning**.

We thank the reviewer again for the constructive suggestions. These clarifications and additions sharpen the scope of the paper, make the Hamiltonian assumptions more explicit, and better locate the work relative to broader AI4Science and ML directions.

---

# Response to Reviewer swV6

We thank the reviewer for the constructive feedback. We agree that the submission should have been clearer on scope, approximate closure, error decomposition, proof presentation, and several experimental details. We address these points below.

## 1. On fit to KDD/data science

We agree that the current paper is not a learned-model or data-mining paper in the narrow sense. Our main contribution remains an **error-certified reduced simulation framework** for scientific dynamical systems.

To make the connection to downstream AI4Science workflows more concrete, we added a **small downstream-learning probe** in the single-cosine regime. Using the same learner for both feature families---PCA to $16$ dimensions followed by Ridge regression---we compared **coherence-aware Shadow features** against **task-level low-pass density inputs** for predicting the next-step unresolved high-frequency energy $E_{\mathrm{HF}}(t+\Delta t)$, which is **not** directly available from the task truncation:

Train size|Low-pass density MSE|Shadow coherence $Z$ MSE|Relative gain of Shadow
|---|---:|---:|---:|
16|$2.369252\times 10^{-7}$|$2.015528\times 10^{-7}$|14.9\%
32|$2.316244\times 10^{-7}$|$1.676996\times 10^{-7}$|27.6\%
64|$2.596946\times 10^{-7}$|$1.509508\times 10^{-7}$|41.9\%

Thus, in the regime studied, under the same downstream learner and matched feature dimension, the coherence-aware Shadow representation achieves **lower test MSE** than task-level low-pass inputs on a downstream unresolved scalar quantity. We stress that this is **not** a full ML benchmark, but it does provide a concrete hybrid-learning example showing that ShadowFluid can serve as a **compact coherence-aware frontend for downstream learning**.

## 2. On the positioning of approximate closure

We agree that this point should have been positioned more carefully. Our setting is narrower than BBGKY-type, Gaussian, or kinetic moment closure. What we study is **operator-subspace closure induced by Hamiltonian commutators**: exact closure means the commutator stays in the dictionary span; approximate closure means projecting/truncating commutator dynamics onto a task-selected operator subspace and quantifying the residual leakage outside that subspace. We will revise the related-work and method sections to make this distinction explicit.

## 3. On separating low-pass truncation error from shadow-approximation error

We agree that this separation should have been shown more explicitly. Our interpretation compares, under the same cutoff $K_0$, (i) the full-state solution, (ii) the exact full low-pass solution, and (iii) the ShadowFluid approximation. The existing $V=0$ sanity check already isolates these effects cleanly: ShadowFluid matches the full low-pass baseline up to machine precision, so the total density error is entirely explained by the spectral truncation induced by the cutoff.

To make this interpretation explicit in the $V \neq 0$ regime, we added a direct decomposition experiment. Here, `eps_cutoff` measures the normalized gap between the full low-pass baseline and the full solution, `eps_shadow` measures the normalized gap between the ShadowFluid result and the full low-pass baseline, and `eps_total` measures the normalized gap between the ShadowFluid result and the full solution.

$\alpha$|$K_0$|$\varepsilon_{\mathrm{cutoff}}$|$\varepsilon_{\mathrm{shadow}}$|$\varepsilon_{\mathrm{total}}$|shadow fraction
|---|---:|---:|---:|---:|---:|
0.2|8|0.006038|0.000079|0.006038|0.0131
0.5|6|0.020021|0.000886|0.020004|0.0443
1.0|4|0.055400|0.010831|0.054553|0.1985

Across all $V \neq 0$ single-cosine settings, the **dominant source of density error is the task cutoff $K_0$**, while the additional shadow approximation error remains secondary. Even in the strongest tested case $(\alpha=1, K_0=4)$, the shadow contribution stays below $20\%$ of the total density error.

## 4. On the proof sketch and experimental reporting

We agree that the theoretical presentation was too compressed. The core argument is that the error dynamics can be written in variation-of-constants form, and unitary conjugation preserves the Frobenius norm; this is why the resulting certificate grows linearly in time rather than exponentially. We will unpack this logic more explicitly and clarify how the leakage term leads to the fully computable bound used in the experiments.

We will also state the numerical precision used in the experiments (`complex128` for state evolution and `float64` for density-level quantities), make the benchmark potential more explicit (for the standard case, $V(x)=\alpha \cos(q\cdot x)$ with $q=(1,0)$), and present the role of the full low-pass baseline more directly.

We thank the reviewer again for the constructive feedback. These clarifications improve the precision of the presentation, better ground the notion of approximate closure, and make the error decomposition more explicit.

---

# Response to Reviewer gBE6

We thank the reviewer for the thoughtful feedback and for recognizing the principled error theory and task-driven dimension reduction perspective of ShadowFluid. We address the main concerns below.

## 1. On the narrow Hamiltonian class and regime boundary

We agree that the submitted version mainly studies Fourier-sparse cosine potentials, and that denser couplings can enlarge the reference set and reduce the dimensionality advantage. To make this regime boundary more explicit, we added a **structured multi-component stress test** that progressively densifies the coupling graph.

In the fixed-total-coupling setting $(\alpha_{\mathrm{tot}}=0.5, K_0=5, t=0.5, R_{\mathrm{hops}}=1)$, increasing the number of Fourier components from $J=1$ to $J=8$ enlarges the reference set from $|R|=3$ to $17$ and the reduced size from $243$ to $1377$, while keeping the density error near $3\times 10^{-2}$ and $\Delta Z_F$ in the $10^{-3}$ to $10^{-2}$ range:

Fourier components $J$|R size|Reduced size|Density error|$\Delta Z_F$|Leakage $\ell_{\mathrm{rms}}$
|---|---:|---:|---:|---:|---:|
1|3|243|0.029049|0.007733|0.242161
2|5|405|0.028939|0.024563|0.214447
4|9|729|0.030072|0.001924|0.137493
6|13|1053|0.030358|0.006105|0.116444
8|17|1377|0.029789|0.003565|0.101099

We also quantified the closure-depth tradeoff on the densest structured case $(J=8)$:

Closure depth $R_{\mathrm{hops}}$|R size|Reduced size|Density error|$\Delta Z_F$|Leakage $\ell_{\mathrm{rms}}$
|---|---:|---:|---:|---:|---:|
1|17|1377|0.029789|0.003565|0.101099
2|55|4455|0.029789|0.002468|0.085052
3|115|9315|0.029789|0.000478|0.077749

These results clarify the **regime boundary** of the current construction rather than claiming full generality over broadband or non-periodic Hamiltonians.

## 2. On the current scope: linear dynamics and controlled validation

We agree that the present paper is restricted to **linear Schrödinger-type dynamics** and does not address nonlinear PDEs such as Navier--Stokes. We will sharpen this boundary in the introduction, discussion, and limitations sections.

We also agree that the current paper does not experimentally demonstrate hardware-level quantum advantage. Our intended claim is narrower: the paper establishes an **error-certified reduced simulation framework** together with controlled validation of its approximation behavior. Beyond the classical matrix-form experiments in the manuscript, we also implemented the same reduced dynamics as a Qiskit-based quantum-circuit realization and used it as an implementation-level cross-check in a noiseless setting against the classical version. This does **not** constitute a hardware-level validation of quantum advantage.

## 3. On robustness and future extensions

We agree that the submitted version relied too heavily on a single initial-condition family. To address this, we added two additional initial-condition families under the same standard single-cosine benchmark setting $(\alpha=0.5, q=(1,0), t=0.5, R_{\mathrm{hops}}=1)$: a deterministic multi-scale packet and a random band-limited state with fixed seed.

Across all tested initial-condition families, the same qualitative hierarchy is preserved: the **dominant source of density error is the task cutoff $K_0$**, while the additional shadow approximation error remains secondary. Representative results at the same cutoff $K_0=6$ are shown below:

Initial condition|$K_0$|$\varepsilon_{\mathrm{cutoff}}$|$\varepsilon_{\mathrm{shadow}}$|$\varepsilon_{\mathrm{total}}$|shadow fraction
|---|---:|---:|---:|---:|---:|
Gaussian vortex|6|0.020021|0.000886|0.020004|0.0443
Multi-scale packet|6|0.026353|0.000832|0.026458|0.0315
Random band-limited state|6|0.594620|0.020003|0.593415|0.0337

The same trend also holds at the larger cutoff $K_0=8$, where the shadow fractions remain small (Gaussian vortex: $0.0338$, multi-scale packet: $0.0202$, random band-limited state: $0.0493$). These results show that the qualitative error hierarchy is not specific to the Gaussian-vortex initialization used in the original submission.

We also agree that adaptive dictionary construction, tighter state-aware estimators, explicit physicality management of the reduced object, and broader comparisons to learned ROM baselines are important next-step extensions. In the revision, we will reflect these directions more explicitly in the limitations and future-work discussion, and clarify that the current leakage-based bound is a **state-independent worst-case certificate**, the present dictionary construction is structural rather than adaptive, and the reduced block is used as a task-level carrier rather than a standalone physical density operator.

We thank the reviewer again for the constructive feedback. These clarifications and additions make the regime boundary more explicit, sharpen the scope of the claims, and more clearly separate the demonstrated contributions of ShadowFluid from broader future extensions.