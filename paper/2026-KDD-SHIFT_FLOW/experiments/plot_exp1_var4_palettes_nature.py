"""experiments/plot_exp1_var4_palettes_nature.py

Exp1 variant 4: grouped bars with std error bars.

This script generates 8 "Nature-like" palette variants:
- restrained saturation
- either monochrome gradients or gray+accent schemes

Output:
  figs_exp1_var4_nature/exp1_var4_nature_palette*_*.pdf
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

import plot_common as pc


PALETTES = [
    (
        "navy_grad",
        [
            "#0B1F3A",  # deep navy
            "#2F4B6C",  # muted blue
            "#8AA2B6",  # steel
        ],
    ),
    (
        "teal_grad",
        [
            "#073B3A",  # deep teal
            "#2E6F6D",  # muted teal
            "#A7C7C5",  # pale teal
        ],
    ),
    (
        "olive_grad",
        [
            "#1F2D2E",  # deep green-gray
            "#3E6259",  # muted green
            "#B9C6BE",  # pale green-gray
        ],
    ),
    (
        "warm_earth",
        [
            "#2D2322",  # espresso
            "#6B4F3A",  # umber
            "#CDBBA7",  # sand
        ],
    ),
    (
        "gray_accent_blue",
        [
            "#232323",  # charcoal
            "#7A7A7A",  # mid gray
            "#2A4D69",  # deep blue accent
        ],
    ),
    (
        "gray_accent_emerald",
        [
            "#232323",  # charcoal
            "#7A7A7A",  # mid gray
            "#2D6A4F",  # deep green accent
        ],
    ),
    (
        "slate_duotone",
        [
            "#1F2937",  # slate-800
            "#4B5563",  # slate-600
            "#9CA3AF",  # slate-400
        ],
    ),
    (
        "steel_duotone",
        [
            "#1C2C3A",  # ink
            "#516B7A",  # steel
            "#BCCAD2",  # mist
        ],
    ),
]


def _style_metric(metric: str):
    if metric == "rho":
        return r"$\varepsilon_\rho$", "err_rho_vs_full"
    if metric == "mom":
        return r"$\varepsilon_{\mathbf{J}}$", "err_momentum_vs_full"
    raise ValueError(metric)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/sweep.csv")
    ap.add_argument("--figdir", default="figs_exp1_var4_nature")
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

    nx_vals = sorted(int(x) for x in pc.unique_sorted(rows_t, "nx"))
    K0_vals = sorted(float(x) for x in pc.unique_sorted(rows_t, "K0"))

    # K0 -> M mapping
    K0_to_M = {}
    for r in rows_t:
        k0 = r.get("K0")
        m = r.get("M")
        if k0 is None or m is None:
            continue
        K0_to_M[float(k0)] = int(m)

    # aggregate mean/std
    keys = ["err_rho_vs_full", "err_momentum_vs_full"]
    stats = {}
    for nx in nx_vals:
        for K0 in K0_vals:
            rs = [r for r in rows_t if int(r.get("nx")) == nx and abs(float(r.get("K0")) - K0) <= 1e-12]
            d = {}
            for k in keys:
                mu, sd, _n = pc.mean_std([r.get(k) for r in rs])
                d[k] = (mu, sd)
            stats[(nx, K0)] = d

    x = np.arange(len(K0_vals), dtype=float)
    width = 0.75 / max(len(nx_vals), 1)

    outs = []
    for idx, (name, colors) in enumerate(PALETTES, start=1):
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.9), constrained_layout=True)

        for ax, metric in zip(axes, ["rho", "mom"]):
            ylab, key = _style_metric(metric)
            for i, nx in enumerate(nx_vals):
                color = colors[i % len(colors)]
                mu = [stats[(nx, K0)][key][0] for K0 in K0_vals]
                sd = [stats[(nx, K0)][key][1] for K0 in K0_vals]
                xi = x - 0.375 + (i + 0.5) * width

                ax.bar(
                    xi,
                    mu,
                    width=width,
                    color=color,
                    alpha=0.88,
                    edgecolor=(0, 0, 0, 0.28),
                    linewidth=0.7,
                    label=f"nx={nx}",
                    zorder=2,
                )
                ax.errorbar(
                    xi,
                    mu,
                    yerr=sd,
                    fmt="none",
                    ecolor=(0, 0, 0, 0.55),
                    elinewidth=0.85,
                    capsize=2,
                    zorder=3,
                )

            ax.set_yscale("log")
            ax.set_title(f"{ylab} (shadow vs full)")
            ax.set_xlabel("K0")

            # Nature-like: very subtle grid (or off)
            ax.grid(False)

            labels = [f"{k0:g}\n(M={K0_to_M.get(float(k0), 0)})" for k0 in K0_vals]
            ax.set_xticks(x)
            ax.set_xticklabels(labels)

        axes[0].legend(loc="best", frameon=False)
        fig.suptitle(f"Exp1 var4 (Nature palette {idx}): {name}  (t={t_eval:g})")

        out = figdir / f"exp1_var4_nature_palette{idx:02d}_{name}.pdf"
        fig.savefig(out)
        plt.close(fig)
        outs.append(out)

    for o in outs:
        print(f"Wrote: {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
