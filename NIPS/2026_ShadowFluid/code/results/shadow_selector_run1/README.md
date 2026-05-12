# Shadow Selector Dataset

This dataset is built from the existing ShadowFluid rollout engine.

Each sample stores:
- a random multi-component potential
- a clipped candidate reference pool
- an oracle budgeted reference set found by exhaustive search
- per-candidate features for a lightweight learned selector

Samples: 16
Grid: N = 16
Cutoff: K0 = 4.0
Budget: 4
Pool hops: 1
Pool max candidates: 7
Eval seeds: [0, 1]
Eval times: [0.3, 0.6]

Files: samples.csv, candidates.csv