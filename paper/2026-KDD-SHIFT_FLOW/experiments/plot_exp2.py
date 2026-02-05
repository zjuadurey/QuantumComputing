"""experiments/plot_exp2.py

Exp2: Pareto frontier (error vs cost proxy).

We use a combined error metric: max(err_rho, err_momentum) aggregated over seeds
at an evaluation time.
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
    ap.add_argument("--x", choices=["q_shift", "M"], default="q_shift")
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
    colors = pc.color_cycle(len(nx_vals))

    fig, ax = plt.subplots(1, 1, figsize=(6.6, 4.0), constrained_layout=True)
    for c, nx in zip(colors, nx_vals):
        xs = []
        ys = []
        for K0 in K0_vals:
            rs = [r for r in rows_t if r.get("nx") == nx and r.get("K0") == K0]
            comb = []
            for r in rs:
                er = float(r.get("err_rho_vs_full"))
                em = float(r.get("err_momentum_vs_full"))
                comb.append(max(er, em))
            mu, _sd, _n = pc.mean_std(comb)
            # x proxy from first row
            x_val = rs[0].get(args.x) if rs else None
            if x_val is None:
                continue
            xs.append(float(x_val))
            ys.append(mu)
        ax.plot(xs, ys, marker="o", lw=1.8, ms=4.5, color=c, label=f"nx={nx}")

        # baseline qubit proxy marker (full baseline cost)
        rs0 = [r for r in rows_t if r.get("nx") == nx]
        if rs0 and args.x == "q_shift":
            qb = float(rs0[0].get("q_base"))
            ax.scatter([qb], [min(ys) if ys else 1e-18], marker="*", s=90, color=c, edgecolor="k", linewidth=0.5)

    ax.set_yscale("log")
    ax.set_xlabel(args.x)
    ax.set_ylabel(r"combined error  max($\varepsilon_\rho$, $\varepsilon_{\mathbf{J}}$)")
    ax.set_title(f"Exp2: Pareto at t={t_eval:g}")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(loc="best", frameon=True)

    out = figdir / f"exp2_pareto_error_vs_{args.x}.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
