"""experiments/plot_exp7.py

Exp7: error vs time curves for several K0.
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
    ap.add_argument("--nx", type=int, default=6)
    args = ap.parse_args()

    rows = pc.load_sweep_csv(args.inp)
    if not rows:
        raise SystemExit(f"No rows found in {args.inp}")

    rows = [r for r in rows if r.get("nx") == int(args.nx)]
    if not rows:
        raise SystemExit(f"No rows for nx={args.nx}")

    pc.apply_mpl_style()
    figdir = pc.ensure_figdir(args.figdir)

    K0_vals = pc.unique_sorted(rows, "K0")
    t_vals = pc.unique_sorted(rows, "t")
    t_vals = sorted(float(t) for t in t_vals)
    colors = pc.color_cycle(len(K0_vals))

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), constrained_layout=True)

    for ax, key, title in [
        (axes[0], "err_rho_vs_full", r"$\varepsilon_\rho(t)$ (vs full)"),
        (axes[1], "err_momentum_vs_full", r"$\varepsilon_{\mathbf{J}}(t)$ (vs full)"),
    ]:
        for c, K0 in zip(colors, K0_vals):
            ys = []
            for t in t_vals:
                rs = [r for r in rows if r.get("K0") == K0 and abs(float(r.get("t")) - t) <= 1e-12]
                mu, _sd, _n = pc.mean_std([r.get(key) for r in rs])
                ys.append(mu)
            ax.plot(t_vals, ys, lw=1.8, color=c, label=f"K0={K0:g}")
        ax.set_yscale("log")
        ax.set_xlabel("t")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.35)

    axes[0].legend(loc="best", frameon=True)
    fig.suptitle(f"Exp7: Error vs time (nx={int(args.nx)})")
    out = figdir / f"exp7_error_vs_time_nx{int(args.nx)}.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
