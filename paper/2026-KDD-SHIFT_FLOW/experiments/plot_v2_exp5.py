"""experiments/plot_v2_exp5.py

Exp5 (v2): Task-only evaluation (E_LP) without reconstructing full fields.

Style choice (per experiments/plot.md):
- Bar-line combo with dual y-axes:
  * bars: M (#retained modes) as a cost background
  * lines: task-only error vs K0 with confidence bands
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
    K0_vals = sorted(float(x) for x in pc.unique_sorted(rows_t, "K0"))

    # K0 -> M mapping
    K0_to_M = {}
    for r in rows_t:
        k0 = r.get("K0")
        m = r.get("M")
        if k0 is None or m is None:
            continue
        K0_to_M[float(k0)] = int(m)

    Ms = [K0_to_M[k0] for k0 in K0_vals]

    colors = pc.color_cycle(len(nx_vals))

    fig, ax = plt.subplots(1, 1, figsize=(7.8, 3.9), constrained_layout=True)

    # bars: M on secondary axis
    ax2 = ax.twinx()
    x = np.arange(len(K0_vals))
    ax2.bar(x, Ms, width=0.7, color="#cccccc", alpha=0.6, label="M")
    ax2.set_ylabel("M (# retained modes)")

    # lines: err_E_LP across seeds
    eps = 1e-18
    for color, nx in zip(colors, nx_vals):
        mu_list = []
        sd_list = []
        for k0 in K0_vals:
            rs = [r for r in rows_t if r.get("nx") == nx and abs(float(r.get("K0")) - k0) <= 1e-12]
            mu, sd, _n = pc.mean_std([r.get("err_E_LP") for r in rs])
            mu_list.append(mu + eps)
            sd_list.append(sd)
        y = np.asarray(mu_list, dtype=float)
        s = np.asarray(sd_list, dtype=float)
        lo = np.maximum(y - s, eps)
        hi = np.maximum(y + s, eps)
        ax.plot(x, y, marker="o", color=color, label=f"nx={nx}")
        ax.fill_between(x, lo, hi, color=pc.alpha_band(color, 0.18), linewidth=0)

    ax.set_yscale("log")
    ax.set_xlabel("K0 (low-frequency radius)")
    ax.set_ylabel(r"task-only error  $|E_{LP}^{sh} - E_{LP}^{base}|/|E_{LP}^{base}|$")
    ax.set_title(f"Exp5 (v2): Task-only E_LP without reconstruction (t={t_eval:g})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k0:g}" for k0 in K0_vals])
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(loc="upper right", frameon=True)

    out = figdir / "exp5_v2_task_only_E_LP_bar_line.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
