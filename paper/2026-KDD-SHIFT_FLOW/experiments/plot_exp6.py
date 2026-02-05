"""experiments/plot_exp6.py

Exp6: multi-case robustness across seeds (boxplots).
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
    ap.add_argument("--t", type=float, default=None, help="evaluation time (default: max t in file)")
    args = ap.parse_args()

    rows = pc.load_sweep_csv(args.inp)
    if not rows:
        raise SystemExit(f"No rows found in {args.inp}")

    t_eval = pc.max_t(rows) if args.t is None else float(args.t)
    rows_t = pc.filter_close(rows, "t", t_eval)
    rows_t = [r for r in rows_t if r.get("nx") == int(args.nx)]
    if not rows_t:
        raise SystemExit(f"No rows for nx={args.nx} at t={t_eval}")

    pc.apply_mpl_style()
    figdir = pc.ensure_figdir(args.figdir)

    K0_vals = pc.unique_sorted(rows_t, "K0")

    data_rho = []
    data_mom = []
    labels = []
    for K0 in K0_vals:
        rs = [r for r in rows_t if r.get("K0") == K0]
        data_rho.append([float(r.get("err_rho_vs_full")) for r in rs])
        data_mom.append([float(r.get("err_momentum_vs_full")) for r in rs])
        m = int(rs[0].get("M")) if rs else 0
        labels.append(f"{K0:g}\n(M={m})")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), constrained_layout=True)
    axes[0].boxplot(data_rho, labels=labels, showfliers=False)
    axes[0].set_yscale("log")
    axes[0].set_title(r"Seed robustness: $\varepsilon_\rho$")
    axes[0].grid(True, which="both", alpha=0.35)

    axes[1].boxplot(data_mom, labels=labels, showfliers=False)
    axes[1].set_yscale("log")
    axes[1].set_title(r"Seed robustness: $\varepsilon_{\mathbf{J}}$")
    axes[1].grid(True, which="both", alpha=0.35)

    fig.suptitle(f"Exp6: Multi-case across seeds (nx={args.nx}, t={t_eval:g})")
    out = figdir / f"exp6_multicase_boxplot_nx{int(args.nx)}.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
