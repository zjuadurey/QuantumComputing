"""experiments/plot_v2_exp8.py

Exp8 (new): Heatmap overview of error vs (nx, K0).

Style choice (per experiments/plot.md):
- Heatmaps for matrix-like comparisons across multiple axes.
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

    nx_vals = sorted(int(x) for x in pc.unique_sorted(rows_t, "nx"))
    K0_vals = sorted(float(x) for x in pc.unique_sorted(rows_t, "K0"))

    def mat(key: str) -> np.ndarray:
        out = np.full((len(nx_vals), len(K0_vals)), np.nan, dtype=float)
        for i, nx in enumerate(nx_vals):
            for j, k0 in enumerate(K0_vals):
                rs = [r for r in rows_t if int(r.get("nx")) == nx and abs(float(r.get("K0")) - k0) <= 1e-12]
                mu, _sd, _n = pc.mean_std([r.get(key) for r in rs])
                out[i, j] = mu
        return out

    rho = mat("err_rho_vs_full")
    mom = mat("err_momentum_vs_full")

    eps = 1e-18
    rho_l = np.log10(np.maximum(rho, eps))
    mom_l = np.log10(np.maximum(mom, eps))

    vmin = float(np.nanmin([np.nanmin(rho_l), np.nanmin(mom_l)]))
    vmax = float(np.nanmax([np.nanmax(rho_l), np.nanmax(mom_l)]))

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9), constrained_layout=True)

    im0 = axes[0].imshow(rho_l, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0].set_title(r"$\log_{10} \varepsilon_\rho$ (vs full)")
    axes[0].set_xlabel("K0")
    axes[0].set_ylabel("nx")
    axes[0].set_xticks(np.arange(len(K0_vals)))
    axes[0].set_xticklabels([f"{k0:g}" for k0 in K0_vals])
    axes[0].set_yticks(np.arange(len(nx_vals)))
    axes[0].set_yticklabels([str(nx) for nx in nx_vals])

    im1 = axes[1].imshow(mom_l, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
    axes[1].set_title(r"$\log_{10} \varepsilon_{\mathbf{J}}$ (vs full)")
    axes[1].set_xlabel("K0")
    axes[1].set_ylabel("nx")
    axes[1].set_xticks(np.arange(len(K0_vals)))
    axes[1].set_xticklabels([f"{k0:g}" for k0 in K0_vals])
    axes[1].set_yticks(np.arange(len(nx_vals)))
    axes[1].set_yticklabels([str(nx) for nx in nx_vals])

    cbar = fig.colorbar(im1, ax=axes, fraction=0.046, pad=0.02)
    cbar.set_label("log10 error")

    fig.suptitle(f"Exp8: Heatmap overview at t={t_eval:g}")
    out = figdir / "exp8_heatmap_error_vs_nx_K0.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
