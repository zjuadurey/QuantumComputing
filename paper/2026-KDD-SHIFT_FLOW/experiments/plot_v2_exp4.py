"""experiments/plot_v2_exp4.py

Exp4 (v2): Qiskit spot-check validation.

Style choice (per experiments/plot.md):
- Line plot with a range band (min-max across sampled points) to show the
  numerical agreement between FFT baseline and Qiskit baseline.
"""

from __future__ import annotations

import argparse
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

import plot_common as pc


def _finite(xs):
    out = []
    for x in xs:
        try:
            v = float(x)
        except Exception:
            continue
        if math.isfinite(v):
            out.append(v)
    return out


def _range(vals):
    v = _finite(vals)
    if not v:
        return float("nan"), float("nan"), float("nan")
    return float(np.min(v)), float(np.max(v)), float(np.median(v))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/qiskit_spotcheck.csv")
    ap.add_argument("--figdir", default="figs_v2")
    args = ap.parse_args()

    rows = []
    with open(args.inp, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    if not rows:
        raise SystemExit(f"No rows found in {args.inp}")

    pc.apply_mpl_style()
    figdir = pc.ensure_figdir(args.figdir)

    nx_vals = sorted({int(float(r["nx"])) for r in rows})

    def agg(key: str):
        lo = []
        hi = []
        med = []
        for nx in nx_vals:
            rs = [r for r in rows if int(float(r["nx"])) == nx]
            vmin, vmax, vmed = _range([r.get(key) for r in rs])
            lo.append(vmin)
            hi.append(vmax)
            med.append(vmed)
        return np.asarray(lo), np.asarray(hi), np.asarray(med)

    rho_lo, rho_hi, rho_med = agg("err_rho_fft_vs_qiskit")
    mom_lo, mom_hi, mom_med = agg("err_momentum_fft_vs_qiskit")
    omg_lo, omg_hi, omg_med = agg("err_omega_fft_vs_qiskit")

    fft_lo, fft_hi, fft_med = agg("rt_fft_s")
    q_lo, q_hi, q_med = agg("rt_qiskit_s")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), constrained_layout=True)

    ax = axes[0]
    c1, c2, c3 = pc.color_cycle(3)
    ax.plot(nx_vals, rho_med, marker="o", color=c1, label=r"median $\varepsilon_\rho$")
    ax.fill_between(nx_vals, rho_lo, rho_hi, color=pc.alpha_band(c1, 0.18), linewidth=0)
    ax.plot(nx_vals, mom_med, marker="o", color=c2, label=r"median $\varepsilon_{\mathbf{J}}$")
    ax.fill_between(nx_vals, mom_lo, mom_hi, color=pc.alpha_band(c2, 0.18), linewidth=0)

    if np.isfinite(omg_med).any():
        ax.plot(nx_vals, omg_med, marker="o", color=c3, label=r"median $\varepsilon_\omega$")
        ax.fill_between(nx_vals, omg_lo, omg_hi, color=pc.alpha_band(c3, 0.18), linewidth=0)

    ax.set_yscale("log")
    ax.set_xlabel("nx")
    ax.set_ylabel("error (FFT vs Qiskit)")
    ax.set_title("Exp4 (v2): Baseline equivalence")
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(loc="best", frameon=True)

    ax = axes[1]
    ax.plot(nx_vals, fft_med, marker="o", color=c1, label="median FFT runtime")
    ax.fill_between(nx_vals, fft_lo, fft_hi, color=pc.alpha_band(c1, 0.18), linewidth=0)
    ax.plot(nx_vals, q_med, marker="o", color=c2, label="median Qiskit runtime")
    ax.fill_between(nx_vals, q_lo, q_hi, color=pc.alpha_band(c2, 0.18), linewidth=0)
    ax.set_yscale("log")
    ax.set_xlabel("nx")
    ax.set_ylabel("runtime (s)")
    ax.set_title("Runtime (spot-check points)")
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(loc="best", frameon=True)

    out = figdir / "exp4_v2_qiskit_spotcheck_bands.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
