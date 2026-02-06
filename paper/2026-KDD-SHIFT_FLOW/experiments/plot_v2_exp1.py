"""experiments/plot_v2_exp1.py

Exp1 (v2): Accuracy vs retained modes (M) / cutoff (K0).

Style choice (per experiments/plot.md):
- Use line plots with confidence bands (mean +/- std) across seeds.
"""

from __future__ import annotations

import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

import plot_common as pc


def _clip_pos(x: float, floor: float = 1e-18) -> float:
    return max(float(x), float(floor))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/sweep.csv")
    ap.add_argument("--figdir", default="figs_v2")
    ap.add_argument("--t", type=float, default=None, help="evaluation time (default: max t in file)")
    ap.add_argument("--y", choices=["log", "linear"], default="log")
    args = ap.parse_args()

    rows = pc.load_sweep_csv(args.inp)
    if not rows:
        raise SystemExit(f"No rows found in {args.inp}")

    t_eval = pc.max_t(rows) if args.t is None else float(args.t)
    rows_t = pc.filter_close(rows, "t", t_eval)
    if not rows_t:
        raise SystemExit(f"No rows at t={t_eval} in {args.inp}")

    pc.apply_mpl_style()
    figdir = pc.ensure_figdir(args.figdir)

    nx_vals = pc.unique_sorted(rows_t, "nx")
    K0_vals = pc.unique_sorted(rows_t, "K0")

    # K0 -> M mapping from the first seen row
    K0_to_M: dict[float, int] = {}
    for r in rows_t:
        k0 = r.get("K0")
        m = r.get("M")
        if k0 is None or m is None:
            continue
        if float(k0) not in K0_to_M:
            K0_to_M[float(k0)] = int(m)

    # x-axis (M) in ascending order
    x_pairs = sorted((K0_to_M[float(k0)], float(k0)) for k0 in K0_vals if float(k0) in K0_to_M)
    Ms = [int(m) for (m, _k0) in x_pairs]
    K0s_sorted = [float(k0) for (_m, k0) in x_pairs]

    colors = pc.color_cycle(len(nx_vals))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)

    panels = [
        (axes[0], "err_rho_vs_full", r"$\varepsilon_\rho$ (shadow vs full)"),
        (axes[1], "err_momentum_vs_full", r"$\varepsilon_{\mathbf{J}}$ (shadow vs full)"),
    ]

    for ax, key, title in panels:
        for color, nx in zip(colors, nx_vals):
            means = []
            stds = []
            for K0 in K0s_sorted:
                rs = [r for r in rows_t if r.get("nx") == nx and abs(float(r.get("K0")) - K0) <= 1e-12]
                mu, sd, _n = pc.mean_std([r.get(key) for r in rs])
                means.append(mu)
                stds.append(sd)

            y = np.asarray(means, dtype=float)
            s = np.asarray(stds, dtype=float)
            lo = np.maximum(y - s, 1e-18)
            hi = np.maximum(y + s, 1e-18)

            ax.plot(Ms, y, marker="o", color=color, label=f"nx={nx}")
            ax.fill_between(Ms, lo, hi, color=pc.alpha_band(color, 0.18), linewidth=0)

        ax.set_xlabel("M (# retained k-modes)")
        ax.set_title(title)
        if args.y == "log":
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.28)

        # show K0 on tick labels (2-line)
        labels = [f"{m}\n(K0={k0:g})" for m, k0 in zip(Ms, K0s_sorted)]
        ax.set_xticks(Ms)
        ax.set_xticklabels(labels)

    axes[0].legend(loc="best", frameon=True)
    fig.suptitle(f"Exp1 (v2): Accuracy improves with M (t={t_eval:g})")

    out = figdir / "exp1_v2_accuracy_vs_M_with_bands.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
