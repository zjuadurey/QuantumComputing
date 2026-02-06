"""experiments/plot_v2_exp7.py

Exp7 (v2): Error vs time with confidence bands.

Style choice (per experiments/plot.md):
- Line plots with confidence bands (mean +/- std across seeds).
"""

from __future__ import annotations

import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

import plot_common as pc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/sweep.csv")
    ap.add_argument("--figdir", default="figs_v2")
    ap.add_argument("--nx", type=int, default=6)
    args = ap.parse_args()

    rows = pc.load_sweep_csv(args.inp)
    if not rows:
        raise SystemExit(f"No rows found in {args.inp}")

    rows = [r for r in rows if int(r.get("nx")) == int(args.nx)]
    if not rows:
        raise SystemExit(f"No rows for nx={args.nx}")

    pc.apply_mpl_style()
    figdir = pc.ensure_figdir(args.figdir)

    K0_vals = sorted(float(x) for x in pc.unique_sorted(rows, "K0"))
    t_vals = sorted(float(x) for x in pc.unique_sorted(rows, "t"))
    colors = pc.color_cycle(len(K0_vals))

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), constrained_layout=True)
    panels = [
        (axes[0], "err_rho_vs_full", r"$\varepsilon_\rho(t)$ vs full"),
        (axes[1], "err_momentum_vs_full", r"$\varepsilon_{\mathbf{J}}(t)$ vs full"),
    ]

    for ax, key, title in panels:
        for color, k0 in zip(colors, K0_vals):
            mu = []
            sd = []
            for t in t_vals:
                rs = [r for r in rows if abs(float(r.get("K0")) - k0) <= 1e-12 and abs(float(r.get("t")) - t) <= 1e-12]
                m, s, _n = pc.mean_std([r.get(key) for r in rs])
                mu.append(m)
                sd.append(s)
            y = np.asarray(mu, dtype=float)
            s = np.asarray(sd, dtype=float)
            lo = np.maximum(y - s, 1e-18)
            hi = np.maximum(y + s, 1e-18)
            ax.plot(t_vals, y, color=color, label=f"K0={k0:g}")
            ax.fill_between(t_vals, lo, hi, color=pc.alpha_band(color, 0.18), linewidth=0)

        ax.set_yscale("log")
        ax.set_xlabel("t")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.28)

    axes[0].legend(loc="best", frameon=True)
    fig.suptitle(f"Exp7 (v2): Error vs time with uncertainty (nx={int(args.nx)})")
    out = figdir / f"exp7_v2_error_vs_time_bands_nx{int(args.nx)}.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
