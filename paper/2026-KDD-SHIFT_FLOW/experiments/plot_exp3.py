"""experiments/plot_exp3.py

Exp3: scaling with nx (runtime/cost vs nx) at fixed K0.
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
    ap.add_argument("--K0", type=float, default=None, help="fixed K0 (default: max K0 in file)")
    args = ap.parse_args()

    rows = pc.load_sweep_csv(args.inp)
    if not rows:
        raise SystemExit(f"No rows found in {args.inp}")

    K0_vals = pc.unique_sorted(rows, "K0")
    if not K0_vals:
        raise SystemExit("No K0 values")
    K0_sel = float(max(K0_vals)) if args.K0 is None else float(args.K0)
    rows_k = pc.filter_close(rows, "K0", K0_sel)
    if not rows_k:
        raise SystemExit(f"No rows at K0={K0_sel}")

    pc.apply_mpl_style()
    figdir = pc.ensure_figdir(args.figdir)

    nx_vals = pc.unique_sorted(rows_k, "nx")
    nx_vals = sorted(int(x) for x in nx_vals)

    # aggregate runtimes across seeds and t
    rt_base_full = []
    rt_base_lp = []
    rt_shadow = []
    rt_total = []
    qb = []
    qs = []
    M = []

    for nx in nx_vals:
        rs = [r for r in rows_k if r.get("nx") == nx]
        mu_bf, _sd_bf, _n = pc.mean_std([r.get("rt_baseline_full_s") for r in rs])
        mu_blp, _sd_blp, _n = pc.mean_std([r.get("rt_baseline_lp_s") for r in rs])
        mu_s, _sd_s, _n = pc.mean_std([r.get("rt_shadow_s") for r in rs])
        mu_t, _sd_t, _n = pc.mean_std([r.get("rt_total_s") for r in rs])
        rt_base_full.append(mu_bf)
        rt_base_lp.append(mu_blp)
        rt_shadow.append(mu_s)
        rt_total.append(mu_t)
        qb.append(float(rs[0].get("q_base")))
        qs.append(float(rs[0].get("q_shift")))
        M.append(float(rs[0].get("M")))

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), constrained_layout=True)

    ax = axes[0]
    ax.plot(nx_vals, rt_base_full, marker="o", lw=1.8, label="baseline(full) FFT")
    ax.plot(nx_vals, rt_base_lp, marker="o", lw=1.8, ls="--", label="baseline(low-pass) FFT")
    ax.plot(nx_vals, rt_shadow, marker="o", lw=1.8, label="shadow")
    ax.plot(nx_vals, rt_total, marker="o", lw=1.8, label="total")
    ax.set_xlabel("nx")
    ax.set_ylabel("runtime per point (s)")
    ax.set_yscale("log")
    ax.set_title(f"Runtime scaling at K0={K0_sel:g}")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(loc="best", frameon=True)

    ax = axes[1]
    ax.plot(nx_vals, qb, marker="o", lw=1.8, label="q_base")
    ax.plot(nx_vals, qs, marker="o", lw=1.8, label="q_shift")
    ax.set_xlabel("nx")
    ax.set_ylabel("qubit proxy")
    ax.set_title("Cost proxies")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(loc="best", frameon=True)

    # annotate M
    for x, m, y in zip(nx_vals, M, qs):
        ax.text(x, y, f"M={int(m)}", fontsize=8, ha="center", va="bottom")

    out = figdir / "exp3_scaling_nx_runtime_cost.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
