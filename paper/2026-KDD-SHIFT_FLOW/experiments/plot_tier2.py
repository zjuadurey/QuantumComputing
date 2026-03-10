"""experiments/plot_tier2.py

Generate figure and LaTeX data table for Tier 2 (multi-component V) experiments.

Produces:
  - figs/v1_tier2_multi.pdf  (3-panel: density error, task energy error, ||ΔZ||_F vs K0)
  - figs/v1_tier2_multi_table.tex

Usage:
  python experiments/plot_tier2.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "sweep_v1.csv"
FIGS = ROOT / "figs"


def main():
    df = pd.read_csv(RESULTS)
    t2 = df[df["J"] > 1].copy()
    print(f"Tier 2 rows: {len(t2)}")

    t_val = 0.5
    sub = t2[np.isclose(t2["t"], t_val)].sort_values(["J", "K0"])

    if sub.empty:
        print("No Tier 2 data at t=0.5")
        return 1

    # Also get Tier 1 single-component baselines for comparison
    t1 = df[(df["J"] == 1) & np.isclose(df["t"], t_val)]

    # --- Figure: 3-panel ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Styles per J
    styles = {
        5:  {"color": "C0", "marker": "o", "label": "$J=5$, $|\\mathcal{R}|=11$"},
        10: {"color": "C1", "marker": "s", "label": "$J=10$, $|\\mathcal{R}|=21$"},
    }

    # Add Tier 1 reference lines (alpha=1.0 and alpha=2.0) as dashed
    t1_refs = {
        1.0: {"color": "gray", "ls": "--", "label": "Tier 1, $\\alpha=1.0$"},
        2.0: {"color": "gray", "ls": ":",  "label": "Tier 1, $\\alpha=2.0$"},
    }

    metrics = [
        ("err_rho_vs_full", r"$\varepsilon_\rho$", "Density error"),
        ("err_E_LP", r"$\varepsilon_{E_\mathrm{LP}}$", "Task energy error"),
        ("err_Z_frob", r"$\|\Delta Z\|_F$", "Frobenius error"),
    ]

    for ax, (col, ylabel, title) in zip(axes, metrics):
        # Tier 1 reference
        for alpha_ref, sty in t1_refs.items():
            ss = t1[np.isclose(t1["alpha"], alpha_ref)].sort_values("K0")
            if not ss.empty:
                ax.semilogy(ss["K0"], ss[col], ls=sty["ls"], color=sty["color"],
                           label=sty["label"], linewidth=1.2, alpha=0.7)

        # Tier 2 curves
        for J_val in sorted(sub["J"].unique()):
            ss = sub[sub["J"] == J_val].sort_values("K0")
            sty = styles[J_val]
            ax.semilogy(ss["K0"], ss[col], marker=sty["marker"], color=sty["color"],
                       label=sty["label"], markersize=5, linewidth=1.5)

        ax.set_xlabel(r"$K_0$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    FIGS.mkdir(parents=True, exist_ok=True)
    out_pdf = FIGS / "v1_tier2_multi.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
    print(f"Saved: {out_pdf}")
    plt.close(fig)

    # --- LaTeX table ---
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Multi-component potential data ($t=0.5$, Figure~\ref{fig:v1_tier2_multi}).}",
        r"\label{tab:v1_tier2_multi}",
        r"\small",
        r"\begin{tabular}{rrrrrrrr}",
        r"\toprule",
        r"$J$ & $|\mathcal{R}|$ & $K_0$ & $M$ & $\varepsilon_\rho$ & $\varepsilon_{E_\mathrm{LP}}$ & $\ell_\mathrm{rms}$ & $\|\Delta Z\|_F$ \\",
        r"\midrule",
    ]
    for _, row in sub.sort_values(["J", "K0"]).iterrows():
        lines.append(
            f"{int(row['J'])} & {int(row['R_size'])} & {row['K0']:.1f} & {int(row['M_K'])} & "
            f"{row['err_rho_vs_full']:.2e} & {row['err_E_LP']:.2e} & "
            f"{row['leakage_apriori']:.2e} & {row['err_Z_frob']:.2e} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    table_path = FIGS / "v1_tier2_multi_table.tex"
    table_path.write_text("\n".join(lines))
    print(f"Saved: {table_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
