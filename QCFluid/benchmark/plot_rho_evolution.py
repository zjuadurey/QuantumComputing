
# plot_rho_evolution.py
# python plot_rho_evolution.py --method quantum_spectral --case case_36044494
# ────────────────────────────────────────────────────────────
import json, random, argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from tqdm import tqdm

# ---------------- CLI ----------------
parser = argparse.ArgumentParser(
    description="Plot 10 density snapshots from a single-vortex dataset case")
parser.add_argument("--root", default="dataset_root",
                    help="dataset_root directory (default: ./dataset_root)")
parser.add_argument("--method", default="classical_spectral",
                    choices=["classical_spectral", "quantum_spectral", "rk4"],
                    help="simulation method subfolder")
parser.add_argument("--case", default=None,
                    help="case folder name (default: random case)")
parser.add_argument("--spacing", default="equal",
                    choices=["equal", "every10"],
                    help="snapshot rule: 'equal'=10 evenly spaced "
                         "or 'every10'=steps 0,10,...,90")
args = parser.parse_args()

# ---------------- locate case ----------------
case_root = Path(args.root) / "single_vortex" / args.method
if not case_root.exists():
    raise FileNotFoundError(f"{case_root} not found")

if args.case:
    case_dir = case_root / args.case
    if not case_dir.is_dir():
        raise FileNotFoundError(case_dir)
else:
    all_cases = sorted(p for p in case_root.glob("case_*") if p.is_dir())
    case_dir = random.choice(all_cases)
print(f"Using case: {case_dir.relative_to(Path(args.root))}")

# ---------------- load rho + meta ----------------
rho = np.load(case_dir / "rho.npy")          # shape: (Nt, N, N)
with open(case_dir / "meta.json") as fp:
    meta = json.load(fp)
Nt, N, _ = rho.shape
dt = meta["dt"]
times = np.arange(Nt) * dt

# ---------------- choose 10 indices ----------------
if args.spacing == "every10":
    idx = np.arange(0, min(Nt, 100), 10)     # up to 0─90
    if len(idx) < 10:                        # 不足 10 帧，改平均分
        idx = np.linspace(0, Nt-1, 10, dtype=int)
else:  # equal
    idx = np.linspace(0, Nt-1, 10, dtype=int)

print("Snapshot steps:", idx)

# ---------------- plot ----------------
nrows, ncols = 2, 5
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5), sharex=True, sharey=True)

for k, ax in enumerate(axes.flat):
    step = idx[k]
    im = ax.imshow(rho[step], origin="lower", cmap="viridis")
    ax.set_title(f"t={times[step]:.1f}")
    ax.set_xticks([]); ax.set_yticks([])

# 单一颜色条
cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.04)
cbar.set_label("density ρ")

# 超标题
fig.suptitle(
    f"(x0,y0)=({meta['x0']:.2f},{meta['y0']:.2f}), σ={meta['sigma']:.2f}  "
    f"[{args.method}]",
    y=1.03, fontsize=12
)

plt.tight_layout()
plt.show()
