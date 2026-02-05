"""experiments/plot_exp5.py

Exp5: task-only evaluation (E_LP error vs K0/M).
"""

from __future__ import annotations

import argparse

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
        raise SystemExit(f"No rows at t={t_eval}")

    pc.apply_mpl_style()
    figdir = pc.ensure_figdir(args.figdir)

    nx_vals = pc.unique_sorted(rows_t, "nx")
    K0_vals = pc.unique_sorted(rows_t, "K0")
    colors = pc.color_cycle(len(nx_vals))

    fig, ax = plt.subplots(1, 1, figsize=(6.8, 3.8), constrained_layout=True)
    for c, nx in zip(colors, nx_vals):
        xs = []
        ys = []
        yerr = []
        for K0 in K0_vals:
            rs = [r for r in rows_t if r.get("nx") == nx and r.get("K0") == K0]
            mu, sd, _n = pc.mean_std([r.get("err_E_LP") for r in rs])
            xs.append(float(K0))
            ys.append(mu)
            yerr.append(sd)
        ax.errorbar(xs, ys, yerr=yerr, marker="o", lw=1.6, ms=4, capsize=2, color=c, label=f"nx={nx}")

    ax.set_xlabel("K0 (low-frequency radius)")
    ax.set_ylabel(r"task-only error  $|E_{LP}^{shadow} - E_{LP}^{base}|/|E_{LP}^{base}|$")
    ax.set_title(f"Exp5: Task-only E_LP at t={t_eval:g}")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", frameon=True)

    out = figdir / "exp5_task_only_E_LP_error.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
