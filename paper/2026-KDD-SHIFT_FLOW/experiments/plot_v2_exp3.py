"""experiments/plot_v2_exp3.py

Exp3 (v2): Scaling with nx and runtime breakdown.

Style choice (per experiments/plot.md):
- Stacked bar chart for runtime composition (baseline full / baseline low-pass /
  shadow / metrics).
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

import plot_common as pc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/sweep.csv")
    ap.add_argument("--figdir", default="figs_v2")
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

    nx_vals = sorted(int(x) for x in pc.unique_sorted(rows_k, "nx"))

    # aggregate across seeds and t
    comp_names = [
        ("rt_baseline_full_s", "baseline(full)"),
        ("rt_baseline_lp_s", "baseline(low-pass)"),
        ("rt_shadow_s", "shadow"),
        ("rt_metrics_s", "metrics"),
    ]
    comps = {k: [] for k, _ in comp_names}

    qb = []
    qs = []
    M = []

    for nx in nx_vals:
        rs = [r for r in rows_k if int(r.get("nx")) == int(nx)]
        if not rs:
            continue
        for k, _lbl in comp_names:
            mu, _sd, _n = pc.mean_std([r.get(k) for r in rs])
            comps[k].append(mu)
        qb.append(float(rs[0].get("q_base")))
        qs.append(float(rs[0].get("q_shift")))
        M.append(float(rs[0].get("M")))

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9), constrained_layout=True)

    # --- stacked runtime ---
    ax = axes[0]
    x = np.arange(len(nx_vals))
    bottom = np.zeros_like(x, dtype=float)
    colors = pc.color_cycle(len(comp_names))
    for (k, lbl), col in zip(comp_names, colors):
        y = np.asarray(comps[k], dtype=float)
        ax.bar(x, y, bottom=bottom, color=col, alpha=0.9, label=lbl, width=0.7)
        bottom = bottom + y

    ax.set_xticks(x)
    ax.set_xticklabels([str(nx) for nx in nx_vals])
    ax.set_xlabel("nx")
    ax.set_ylabel("runtime per point (s)")
    ax.set_yscale("log")
    ax.set_title(f"Runtime breakdown at K0={K0_sel:g}")
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(loc="best", frameon=True)

    # --- cost proxies ---
    ax = axes[1]
    ax.plot(nx_vals, qb, marker="o", label="q_base")
    ax.plot(nx_vals, qs, marker="o", label="q_shift")
    ax.set_xlabel("nx")
    ax.set_ylabel("qubit proxy")
    ax.set_title("Cost proxies")
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(loc="best", frameon=True)

    # annotate M per nx
    for nx, y, m in zip(nx_vals, qs, M):
        ax.text(nx, y, f"M={int(m)}", fontsize=8, ha="center", va="bottom")

    out = figdir / "exp3_v2_scaling_runtime_stacked.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
