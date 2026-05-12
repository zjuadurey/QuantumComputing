# Shadow Selector Benchmark

- Model: `mlp`
- Hidden sizes: `[32, 16]`
- Train samples: `12`
- Test samples: `4`

## Candidate Classification

- Accuracy: `0.700000`
- Precision: `0.666667`
- Recall: `1.000000`
- F1: `0.800000`

## Strategy Summary

| Strategy | Samples | Mean objective | Mean oracle gap | Mean density error | Mean leakage | Exact match rate |
| --- | --- | --- | --- | --- | --- | --- |
| coupling_greedy | 4 | 0.150280 | 0.000000 | 0.084764 | 0.655153 | 1.000000 |
| learned | 4 | 0.150280 | 0.000000 | 0.084764 | 0.655153 | 1.000000 |
| low_energy | 4 | 0.150585 | 0.000305 | 0.084764 | 0.658206 | 0.750000 |
| oracle | 4 | 0.150280 | 0.000000 | 0.084764 | 0.655153 | 1.000000 |
| random | 4 | 0.152266 | 0.001986 | 0.084764 | 0.675015 | 0.000000 |
