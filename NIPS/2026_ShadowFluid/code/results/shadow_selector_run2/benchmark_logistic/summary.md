# Shadow Selector Benchmark

- Model: `logistic`
- Hidden sizes: `[32, 16]`
- Train samples: `12`
- Test samples: `4`

## Candidate Classification

- Accuracy: `0.571429`
- Precision: `0.500000`
- Recall: `0.666667`
- F1: `0.571429`

## Strategy Summary

| Strategy | Samples | Mean objective | Mean oracle gap | Mean density error | Mean leakage | Exact match rate |
| --- | --- | --- | --- | --- | --- | --- |
| coupling_greedy | 4 | 0.192877 | 0.001327 | 0.095766 | 0.971117 | 0.500000 |
| learned | 4 | 0.192853 | 0.001302 | 0.095766 | 0.970872 | 0.000000 |
| low_energy | 4 | 0.194644 | 0.003093 | 0.095766 | 0.988784 | 0.250000 |
| oracle | 4 | 0.191551 | 0.000000 | 0.095766 | 0.957852 | 1.000000 |
| random | 4 | 0.195476 | 0.003925 | 0.095766 | 0.997098 | 0.000000 |
