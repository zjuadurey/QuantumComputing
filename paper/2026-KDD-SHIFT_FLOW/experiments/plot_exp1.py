"""experiments/plot_exp1.py

Exp 1 — Shadow accuracy vs full simulation, sweeping K0 / M.
Paper-ready figure: smooth fill-between bands, pastel palette.

Dependencies: numpy, matplotlib.  Optional: scipy (for PCHIP smooth).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_common as pc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/sweep.csv")
    ap.add_argument("--figdir", default="figs")
    ap.add_argument("--t", type=float, default=None,
                    help="evaluation time (default: max t in file)")
    args = ap.parse_args()

    # ── data ────────────────────────────────────────────────
    rows = pc.load_sweep_csv(args.inp)
    if not rows:
        raise SystemExit(f"No rows found in {args.inp}")

    t_eval = pc.max_t(rows) if args.t is None else float(args.t)
    rows_t = pc.filter_close(rows, "t", t_eval)
    if not rows_t:
        raise SystemExit(f"No rows at t={t_eval} in {args.inp}")

    nx_vals = pc.unique_sorted(rows_t, "nx")
    K0_vals = pc.unique_sorted(rows_t, "K0")

    K0_to_M: dict[float, int] = {}
    for r in rows_t:
        k0 = r.get("K0")
        if k0 is not None and k0 not in K0_to_M:
            K0_to_M[k0] = int(r.get("M"))

    # ── style ───────────────────────────────────────────────
    pc.apply_paper_rcparams()
    figdir = pc.ensure_figdir(args.figdir)
    colors = pc.PALETTE_PASTEL[: len(nx_vals)]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)

    panels = [
        (axes[0], "err_rho_vs_full",
         r"Density error  $\varepsilon_\rho$"),
        (axes[1], "err_momentum_vs_full",
         r"Momentum error  $\varepsilon_{\mathbf{J}}$"),
    ]

    for ax, key, title in panels:
        for c, nx in zip(colors, nx_vals):
            xs, ys, yerr = [], [], []
            for K0 in K0_vals:
                rs = [r for r in rows_t
                      if r.get("nx") == nx and r.get("K0") == K0]
                mu, sd, _ = pc.mean_std([r.get(key) for r in rs])
                xs.append(float(K0))
                ys.append(mu)
                yerr.append(sd)

            xs = np.asarray(xs)
            ys = np.asarray(ys)
            yerr = np.asarray(yerr)

            # smooth band + line
            xf, yf, lo, hi = pc.smooth_errorband(
                xs, ys, yerr, log_scale=True,
            )
            ax.fill_between(xf, lo, hi, color=c, alpha=0.30, linewidth=0)
            ax.plot(xf, yf, color=c, lw=1.5, label=f"$n_x = {nx}$")
            # markers at actual data points
            ax.scatter(
                xs, ys, color=c, s=16, zorder=5,
                edgecolors="white", linewidths=0.4,
            )

        ax.set_yscale("log")
        ax.set_title(title, pad=6)

        # x-ticks: K0 value + M
        ticks = [float(k) for k in K0_vals]
        labels = []
        for k in K0_vals:
            m = K0_to_M.get(k)
            labels.append(f"{k:g}\nM={m}" if m is not None else f"{k:g}")
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_xlabel(r"$K_0$")

        pc.set_paper_style(ax)

    axes[0].set_ylabel("Relative error")
    axes[0].legend(frameon=False, loc="upper right", handlelength=1.5)

    # panel labels
    for ax, lbl in zip(axes, ["(a)", "(b)"]):
        ax.text(
            -0.08, 1.06, lbl, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top",
        )

    # ── save ────────────────────────────────────────────────
    stem = "exp1_accuracy_vs_K0_M"
    for fmt in ("pdf", "png"):
        out = figdir / f"{stem}.{fmt}"
        fig.savefig(out, dpi=(300 if fmt == "png" else None))
        print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
