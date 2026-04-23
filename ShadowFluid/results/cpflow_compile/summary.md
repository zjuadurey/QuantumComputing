# CPFlow Compile Comparison

## Cases run

- Representative grid: `nx = 3`, `N = 8`.
- `V=0` sanity case with canonical vortex state, `K0 = 1.5`, `t = 0.5`.
- `V!=0` representative case with single-cosine potential, `alpha = 0.5`, `q = (1, 0)`, `K0 = 1.5`, `t = 0.5`, `R_hops = 1`.

## Targets synthesized

- Full initial-state target on the full spinor register.
- Reduced Shadow initial-state target on the packed retained-mode register.
- Full active-mode evolution unitary for `V=0` and `V!=0`.
- Reduced `shadow_K` and `shadow_R` active-mode evolution unitaries for `V!=0`.

## Fixed compilation conditions

- Topology family: `line_nn`.
- Two-qubit basis: `cz`.
- Generic backend: `transpile` with optimization level `3` and fixed seed `7`.
- Quality metrics: `state_infidelity` for state preparation and `unitary_loss` for unitary targets.
- CPFlow target loss budget when available: `1.000000e-06`.

## Produced artifacts

- Raw long-form CSV: `results/cpflow_compile/raw_compile_metrics.csv`.
- Compact paired table: `results/cpflow_compile/main_table.csv`.

## Notes on reduced `V!=0` evolution

- `shadow_K` and `shadow_R` are kept separate in the raw CSV.
- They are aggregated into a single reduced `V!=0` time-evolution entry in `main_table.csv` by summing the two circuit costs and taking the worse achieved quality.

## Factual takeaway

- In this representative run, ShadowFluid creates smaller synthesis targets than the corresponding full targets.
- The generic compilation rows already show lower two-qubit complexity on the reduced targets; CPFlow rows were not generated in this run because CPFlow was unavailable in the selected environment.
