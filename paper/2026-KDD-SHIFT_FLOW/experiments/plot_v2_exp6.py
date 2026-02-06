"""experiments/plot_v2_exp6.py

Exp6 (v2): Multi-case robustness across seeds.

Style choice (per experiments/plot.md):
- Violin plots (preferred over boxplots) to show distribution shape.
- Plot log10(error) to handle multi-order differences.
"""

from __future__ import annotations

import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

import plot_common as pc


def _log10_safe(x: float, eps: float = 1e-18) -> float:
    return math.log10(max(float(x), float(eps)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/sweep.csv")
    ap.add_argument("--figdir", default="figs_v2")
    ap.add_argument("--nx", type=int, default=6)
    ap.add_argument("--t", type=float, default=None, help="evaluation time (default: max t in file)")
    args = ap.parse_args()

    rows = pc.load_sweep_csv(args.inp)
    if not rows:
        raise SystemExit(f"No rows found in {args.inp}")

    t_eval = pc.max_t(rows) if args.t is None else float(args.t)
    rows_t = pc.filter_close(rows, "t", t_eval)
    rows_t = [r for r in rows_t if int(r.get("nx")) == int(args.nx)]
    if not rows_t:
        raise SystemExit(f"No rows for nx={args.nx} at t={t_eval}")

    pc.apply_mpl_style()
    figdir = pc.ensure_figdir(args.figdir)

    K0_vals = sorted(float(x) for x in pc.unique_sorted(rows_t, "K0"))

    data_rho = []
    data_mom = []
    labels = []
    for k0 in K0_vals:
        rs = [r for r in rows_t if abs(float(r.get("K0")) - k0) <= 1e-12]
        rho = [_log10_safe(float(r.get("err_rho_vs_full"))) for r in rs]
        mom = [_log10_safe(float(r.get("err_momentum_vs_full"))) for r in rs]
        data_rho.append(rho)
        data_mom.append(mom)
        m = int(rs[0].get("M")) if rs else 0
        labels.append(f"{k0:g}\n(M={m})")

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), constrained_layout=True)

    parts = axes[0].violinplot(data_rho, showmeans=True, showmedians=False, showextrema=False)
    for b in parts["bodies"]:
        b.set_facecolor(pc.color_cycle(1)[0])
        b.set_alpha(0.6)
    axes[0].set_title(r"Seed distribution: $\log_{10} \varepsilon_\rho$ (vs full)")
    axes[0].set_xticks(np.arange(1, len(labels) + 1))
    axes[0].set_xticklabels(labels)
    axes[0].grid(True, which="both", alpha=0.28)

    parts = axes[1].violinplot(data_mom, showmeans=True, showmedians=False, showextrema=False)
    for b in parts["bodies"]:
        b.set_facecolor(pc.color_cycle(2)[1])
        b.set_alpha(0.6)
    axes[1].set_title(r"Seed distribution: $\log_{10} \varepsilon_{\mathbf{J}}$ (vs full)")
    axes[1].set_xticks(np.arange(1, len(labels) + 1))
    axes[1].set_xticklabels(labels)
    axes[1].grid(True, which="both", alpha=0.28)

    fig.suptitle(f"Exp6 (v2): Multi-case robustness (nx={int(args.nx)}, t={t_eval:g})")
    out = figdir / f"exp6_v2_violin_log10_nx{int(args.nx)}.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
