"""experiments/plot_exp1.py

Exp1: Accuracy vs M/K0.

Plots mean +- std across seeds at a chosen evaluation time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_common as pc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/sweep.csv")
    ap.add_argument("--figdir", default="figs")
    ap.add_argument("--t", type=float, default=None, help="evaluation time (default: max t in file)")
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

    # map K0 -> (M) using the first seen row
    K0_to_M = {}
    for r in rows_t:
        k0 = r.get("K0")
        if k0 is None:
            continue
        if k0 not in K0_to_M:
            K0_to_M[k0] = int(r.get("M"))

    colors = pc.color_cycle(len(nx_vals))
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6), constrained_layout=True)

    for ax, key, title in [
        (axes[0], "err_rho_vs_full", r"Density error vs full  $\varepsilon_\rho$"),
        (axes[1], "err_momentum_vs_full", r"Momentum error vs full  $\varepsilon_{\mathbf{J}}$"),
    ]:
        for c, nx in zip(colors, nx_vals):
            xs = []
            ys = []
            yerr = []
            for K0 in K0_vals:
                rs = [r for r in rows_t if r.get("nx") == nx and r.get("K0") == K0]
                mu, sd, _n = pc.mean_std([r.get(key) for r in rs])
                xs.append(float(K0))
                ys.append(mu)
                yerr.append(sd)
            ax.errorbar(xs, ys, yerr=yerr, marker="o", lw=1.6, ms=4, capsize=2, color=c, label=f"nx={nx}")

        ax.set_yscale("log")
        ax.set_xlabel("K0 (low-frequency radius)")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.35)

        # annotate M in tick labels
        ticks = [float(k) for k in K0_vals]
        labels = []
        for k in K0_vals:
            m = K0_to_M.get(k, None)
            if m is None:
                labels.append(f"{k:g}")
            else:
                labels.append(f"{k:g}\n(M={m})")
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)

    axes[0].legend(loc="best", frameon=True)
    fig.suptitle(f"Exp1: Shadow accuracy vs full (t={t_eval:g})")
    out = figdir / "exp1_accuracy_vs_K0_M.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
