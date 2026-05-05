# Rebuttal Stress Tests

These stress tests were prepared for the response to Reviewer oB6r.

This directory contains reproducible stress tests for the rebuttal discussion on
structured multi-component potentials and closure-depth tradeoffs.

## Files

- `fixed_alpha_total.csv`: sweep graph density `J` and closure depth `R_hops` with fixed total coupling.
- `dense_time_fixed_alpha_total.csv`: time sweep for the densest structured case under fixed total coupling.
- `fixed_alpha_each.csv`: sweep graph density `J` with fixed per-component coupling.
- `dense_hops_fixed_alpha_each.csv`: closure-depth sweep for the densest case at fixed per-component coupling.

## Snapshot: Fixed Total Coupling (`R_hops = 1`)

| J | alpha_total | alpha_each | R_size | reduced_size | err_rho_vs_full | err_Z_frob | leakage_apriori |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.5 | 0.5 | 3 | 243 | 0.029049 | 7.732637e-03 | 0.242161 |
| 2 | 0.5 | 0.25 | 5 | 405 | 0.028939 | 0.024563 | 0.214447 |
| 4 | 0.5 | 0.125 | 9 | 729 | 0.030072 | 1.923587e-03 | 0.137493 |
| 6 | 0.5 | 0.083333 | 13 | 1053 | 0.030358 | 6.105370e-03 | 0.116444 |
| 8 | 0.5 | 0.0625 | 17 | 1377 | 0.029789 | 3.565241e-03 | 0.101099 |

## Snapshot: Dense Case Time Sweep (Fixed Total Coupling)

| t | R_hops | R_size | err_rho_vs_full | err_Z_frob | leakage_apriori | bound_apriori |
| --- | --- | --- | --- | --- | --- | --- |
| 0.2 | 1 | 17 | 0.032471 | 1.419610e-03 | 0.101099 | 0.02022 |
| 0.5 | 1 | 17 | 0.029789 | 3.565241e-03 | 0.101099 | 0.050549 |
| 0.8 | 1 | 17 | 0.021994 | 5.608151e-03 | 0.101099 | 0.080879 |
| 1 | 1 | 17 | 0.023126 | 6.823885e-03 | 0.101099 | 0.101099 |
| 0.2 | 2 | 55 | 0.032471 | 1.053828e-03 | 0.085052 | 0.01701 |
| 0.5 | 2 | 55 | 0.029789 | 2.467696e-03 | 0.085052 | 0.042526 |
| 0.8 | 2 | 55 | 0.021994 | 3.513758e-03 | 0.085052 | 0.068041 |
| 1 | 2 | 55 | 0.023126 | 3.964287e-03 | 0.085052 | 0.085052 |
| 0.2 | 3 | 115 | 0.032471 | 2.250984e-04 | 0.077749 | 0.01555 |
| 0.5 | 3 | 115 | 0.029789 | 4.784983e-04 | 0.077749 | 0.038875 |
| 0.8 | 3 | 115 | 0.021994 | 6.099083e-04 | 0.077749 | 0.062199 |
| 1 | 3 | 115 | 0.023126 | 6.639389e-04 | 0.077749 | 0.077749 |

## Snapshot: Fixed Per-Component Coupling (`R_hops = 1`)

| J | alpha_total | alpha_each | R_size | reduced_size | err_rho_vs_full | err_Z_frob | leakage_apriori |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.1 | 0.1 | 3 | 243 | 0.029582 | 1.542146e-03 | 0.048432 |
| 2 | 0.2 | 0.1 | 5 | 405 | 0.02941 | 9.766295e-03 | 0.085779 |
| 4 | 0.4 | 0.1 | 9 | 729 | 0.030007 | 1.518420e-03 | 0.109994 |
| 6 | 0.6 | 0.1 | 13 | 1053 | 0.030498 | 7.334497e-03 | 0.139732 |
| 8 | 0.8 | 0.1 | 17 | 1377 | 0.029794 | 5.720391e-03 | 0.161758 |

## Snapshot: Dense Case Closure Sweep (Fixed Per-Component Coupling)

| R_hops | alpha_total | R_size | reduced_size | err_rho_vs_full | err_Z_frob | leakage_apriori |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.8 | 17 | 1377 | 0.029794 | 5.720391e-03 | 0.161758 |
| 2 | 0.8 | 55 | 4455 | 0.029794 | 3.968166e-03 | 0.136083 |
| 3 | 0.8 | 115 | 9315 | 0.029794 | 7.781130e-04 | 0.124399 |
