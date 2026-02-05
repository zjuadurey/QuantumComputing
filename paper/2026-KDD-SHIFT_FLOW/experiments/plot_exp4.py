"""experiments/plot_exp4.py

Exp4 (optional): Qiskit spot-check figure.

This plots the agreement between the FFT baseline and Qiskit statevector
baseline on a small set of points produced by:
  experiments/run_qiskit_spotcheck.py

Output:
  figs/exp4_qiskit_spotcheck.pdf
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


def _to_int(x: str) -> int:
    return int(float(x.strip()))


def _to_float(x: str) -> float:
    s = x.strip()
    if s == "" or s.lower() == "nan":
        return float("nan")
    return float(s)


def _finite(vals):
    return [float(v) for v in vals if v is not None and math.isfinite(float(v))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/qiskit_spotcheck.csv")
    ap.add_argument("--figdir", default="figs")
    args = ap.parse_args()

    rows = []
    with open(args.inp, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                {
                    "nx": _to_int(row["nx"]),
                    "N": _to_int(row["N"]),
                    "K0": _to_float(row["K0"]),
                    "t": _to_float(row["t"]),
                    "seed": _to_int(row["seed"]),
                    "err_rho": _to_float(row["err_rho_fft_vs_qiskit"]),
                    "err_mom": _to_float(row["err_momentum_fft_vs_qiskit"]),
                    "err_omg": _to_float(row["err_omega_fft_vs_qiskit"]),
                    "rt_fft": _to_float(row["rt_fft_s"]),
                    "rt_q": _to_float(row["rt_qiskit_s"]),
                }
            )

    if not rows:
        raise SystemExit(f"No rows found in {args.inp}")

    pc.apply_mpl_style()
    figdir = pc.ensure_figdir(args.figdir)

    nx_vals = sorted({r["nx"] for r in rows})

    # aggregate by nx
    max_rho = []
    max_mom = []
    max_omg = []
    med_fft = []
    med_q = []

    for nx in nx_vals:
        rs = [r for r in rows if r["nx"] == nx]
        max_rho.append(max(_finite([r["err_rho"] for r in rs])) if rs else float("nan"))
        max_mom.append(max(_finite([r["err_mom"] for r in rs])) if rs else float("nan"))

        omg_vals = _finite([r["err_omg"] for r in rs])
        max_omg.append(max(omg_vals) if omg_vals else float("nan"))

        fft_vals = _finite([r["rt_fft"] for r in rs])
        q_vals = _finite([r["rt_q"] for r in rs])
        med_fft.append(float(np.median(fft_vals)) if fft_vals else float("nan"))
        med_q.append(float(np.median(q_vals)) if q_vals else float("nan"))

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6), constrained_layout=True)

    ax = axes[0]
    ax.plot(nx_vals, max_rho, marker="o", lw=1.8, label=r"max $\varepsilon_\rho$")
    ax.plot(nx_vals, max_mom, marker="o", lw=1.8, label=r"max $\varepsilon_{\mathbf{J}}$")
    if any(math.isfinite(x) for x in max_omg):
        ax.plot(nx_vals, max_omg, marker="o", lw=1.8, label=r"max $\varepsilon_\omega$")
    ax.set_yscale("log")
    ax.set_xlabel("nx")
    ax.set_ylabel("error (FFT vs Qiskit)")
    ax.set_title("Exp4: FFT baseline vs Qiskit")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(loc="best", frameon=True)

    ax = axes[1]
    ax.plot(nx_vals, med_fft, marker="o", lw=1.8, label="median FFT runtime")
    ax.plot(nx_vals, med_q, marker="o", lw=1.8, label="median Qiskit runtime")
    ax.set_yscale("log")
    ax.set_xlabel("nx")
    ax.set_ylabel("runtime (s)")
    ax.set_title("Runtime (spot-check points)")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(loc="best", frameon=True)

    out = figdir / "exp4_qiskit_spotcheck.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
