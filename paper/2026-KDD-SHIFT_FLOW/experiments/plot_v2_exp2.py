"""experiments/plot_v2_exp2.py

Exp2 (v2): Pareto frontier (error vs cost) with bubble size for runtime.

Style choice (per experiments/plot.md):
- Pareto frontier scatter (optionally bubble plot) to show trade-offs.
"""

from __future__ import annotations

import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

import plot_common as pc


def _pareto_front(points: list[tuple[float, float]]) -> set[int]:
    """Return indices of Pareto-optimal points (minimize both x and y)."""
    idx = sorted(range(len(points)), key=lambda i: (points[i][0], points[i][1]))
    best_y = float("inf")
    keep: set[int] = set()
    for i in idx:
        x, y = points[i]
        if y < best_y:
            keep.add(i)
            best_y = y
    return keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/sweep.csv")
    ap.add_argument("--figdir", default="figs_v2")
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

    # aggregate by (nx, K0)
    pts = []
    labels = []
    for nx in nx_vals:
        for K0 in K0_vals:
            rs = [r for r in rows_t if r.get("nx") == nx and abs(float(r.get("K0")) - float(K0)) <= 1e-12]
            if not rs:
                continue

            comb = []
            rts = []
            for r in rs:
                er = float(r.get("err_rho_vs_full"))
                em = float(r.get("err_momentum_vs_full"))
                comb.append(max(er, em))
                rts.append(float(r.get("rt_total_s")))

            mu_e, sd_e, _n = pc.mean_std(comb)
            mu_rt, _sd_rt, _n = pc.mean_std(rts)
            x_val = float(rs[0].get(args.x))
            y_val = float(mu_e)
            pts.append(
                {
                    "nx": int(nx),
                    "K0": float(K0),
                    "M": int(rs[0].get("M")),
                    "x": float(x_val),
                    "y": float(y_val),
                    "rt": float(mu_rt),
                    "y_sd": float(sd_e),
                    "q_base": int(rs[0].get("q_base")),
                }
            )
            labels.append(f"K0={float(K0):g} (M={int(rs[0].get('M'))})")

    if not pts:
        raise SystemExit("No aggregated points")

    # bubble sizes
    rts = np.asarray([p["rt"] for p in pts], dtype=float)
    rt_min = float(np.min(rts))
    rt_max = float(np.max(rts))
    size = 40.0 + 260.0 * (rts - rt_min) / (rt_max - rt_min + 1e-12)

    xy = [(p["x"], p["y"]) for p in pts]
    front_idx = _pareto_front(xy)

    fig, ax = plt.subplots(1, 1, figsize=(6.6, 4.2), constrained_layout=True)

    # scatter per nx
    for color, nx in zip(colors, nx_vals):
        idx = [i for i, p in enumerate(pts) if p["nx"] == int(nx)]
        if not idx:
            continue
        xs = [pts[i]["x"] for i in idx]
        ys = [pts[i]["y"] for i in idx]
        ss = [size[i] for i in idx]
        ec = ["k" if i in front_idx else color for i in idx]
        lw = [0.9 if i in front_idx else 0.0 for i in idx]
        ax.scatter(xs, ys, s=ss, c=color, alpha=0.85, edgecolors=ec, linewidths=lw, label=f"nx={nx}")

    ax.set_yscale("log")
    ax.set_xlabel(args.x)
    ax.set_ylabel(r"combined error  max($\varepsilon_\rho$, $\varepsilon_{\mathbf{J}}$)")
    ax.set_title(f"Exp2 (v2): Pareto at t={t_eval:g} (bubble=size runtime)")
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(loc="best", frameon=True)

    # baseline q_base markers (one per nx)
    if args.x == "q_shift":
        for color, nx in zip(colors, nx_vals):
            rs = [p for p in pts if p["nx"] == int(nx)]
            if not rs:
                continue
            qb = float(rs[0]["q_base"])
            ax.axvline(qb, color=color, lw=1.0, alpha=0.25)

    out = figdir / f"exp2_v2_pareto_bubble_vs_{args.x}.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
