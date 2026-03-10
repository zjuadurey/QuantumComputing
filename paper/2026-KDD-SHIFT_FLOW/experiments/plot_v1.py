"""experiments/plot_v1.py

Generate figures and LaTeX data tables for V!=0 experiments.

Reads results/sweep_v1.csv and produces:
  - Fig: leakage & error vs K0 (at fixed alpha, t)
  - Fig: three-curve comparison (bound / Z-error / task-error)
  - Fig: error vs alpha (coupling strength)
  - Fig: error vs t (temporal stability)
  - LaTeX tables after each figure

Usage:
  python experiments/plot_v1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "sweep_v1.csv"
FIGS = ROOT / "figs"
TABLES = ROOT / "figs"  # put .tex tables next to figs


def load_data() -> pd.DataFrame:
    df = pd.read_csv(RESULTS)
    return df


def _save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight", dpi=150)
    print(f"  Saved: figs/{name}.pdf")
    plt.close(fig)


def _save_table(lines: list[str], name: str):
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / f"{name}_table.tex"
    path.write_text("\n".join(lines))
    print(f"  Saved: figs/{name}_table.tex")


# ================================================================
# Fig 1: Error vs K0 for different alphas (fixed t=0.5)
# ================================================================

def fig_error_vs_K0(df: pd.DataFrame):
    t_val = 0.5
    sub = df[np.isclose(df["t"], t_val)]
    if sub.empty:
        print("  Skipping fig_error_vs_K0: no data at t=0.5")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    alphas = sorted(sub["alpha"].unique())
    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(alphas)))

    metrics = [
        ("err_rho_vs_full", r"$\varepsilon_\rho$", "Density error"),
        ("err_E_LP", r"$\varepsilon_{E_\mathrm{LP}}$", "Task energy error"),
        ("leakage_apriori", r"$\ell_\mathrm{rms}$", "A priori leakage"),
    ]

    for ax, (col, ylabel, title) in zip(axes, metrics):
        for i, alpha in enumerate(alphas):
            ss = sub[np.isclose(sub["alpha"], alpha)].sort_values("K0")
            ax.semilogy(ss["K0"], ss[col], "o-", color=cmap[i],
                       label=rf"$\alpha={alpha}$", markersize=4)
        ax.set_xlabel(r"$K_0$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, "v1_error_vs_K0")

    # LaTeX table
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Data for V$\neq$0 error vs $K_0$ (Figure~\ref{fig:v1_error_vs_K0}, $t=0.5$).}",
        r"\label{tab:v1_error_vs_K0}",
        r"\small",
        r"\begin{tabular}{rrrrrrr}",
        r"\toprule",
        r"$\alpha$ & $K_0$ & $M$ & $\varepsilon_\rho$ & $\varepsilon_{E_\mathrm{LP}}$ & $\ell_\mathrm{rms}$ & $\|\Delta Z\|_F$ \\",
        r"\midrule",
    ]
    for _, row in sub.sort_values(["alpha", "K0"]).iterrows():
        lines.append(
            f"{row['alpha']:.1f} & {row['K0']:.1f} & {int(row['M_K'])} & "
            f"{row['err_rho_vs_full']:.2e} & {row['err_E_LP']:.2e} & "
            f"{row['leakage_apriori']:.2e} & {row['err_Z_frob']:.2e} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _save_table(lines, "v1_error_vs_K0")


# ================================================================
# Fig 2: Three-curve comparison (bound vs actual)
# ================================================================

def fig_three_curves(df: pd.DataFrame):
    t_val = 0.5
    alpha_val = 1.0

    sub = df[np.isclose(df["t"], t_val) & np.isclose(df["alpha"], alpha_val)]
    if sub.empty:
        # try closest alpha
        alpha_val = df["alpha"].unique()[len(df["alpha"].unique())//2]
        sub = df[np.isclose(df["t"], t_val) & np.isclose(df["alpha"], alpha_val)]
    if sub.empty:
        print("  Skipping fig_three_curves: no matching data")
        return

    sub = sub.sort_values("K0")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(sub["K0"], sub["bound_apriori"], "s--", color="C0",
               label=r"A priori bound ($t \cdot \ell_\mathrm{rms} \sqrt{MR}$)", markersize=5)
    ax.semilogy(sub["K0"], sub["err_Z_frob"], "o-", color="C1",
               label=r"Actual $\|\Delta Z\|_F$", markersize=5)
    ax.semilogy(sub["K0"], sub["err_rho_vs_full"], "^-", color="C2",
               label=r"Density error $\varepsilon_\rho$", markersize=5)
    ax.semilogy(sub["K0"], sub["err_E_LP"], "v-", color="C3",
               label=r"Task error $\varepsilon_{E_\mathrm{LP}}$", markersize=5)

    ax.set_xlabel(r"$K_0$")
    ax.set_ylabel("Error / Bound")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "v1_three_curves")

    # LaTeX table
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        rf"\caption{{Error hierarchy data ($\alpha={alpha_val}$, $t={t_val}$, Figure~\ref{{fig:v1_three_curves}}).}}",
        r"\label{tab:v1_three_curves}",
        r"\small",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"$K_0$ & $M$ & Bound & $\|\Delta Z\|_F$ & $\varepsilon_\rho$ & $\varepsilon_{E_\mathrm{LP}}$ \\",
        r"\midrule",
    ]
    for _, row in sub.iterrows():
        lines.append(
            f"{row['K0']:.1f} & {int(row['M_K'])} & "
            f"{row['bound_apriori']:.2e} & {row['err_Z_frob']:.2e} & "
            f"{row['err_rho_vs_full']:.2e} & {row['err_E_LP']:.2e} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _save_table(lines, "v1_three_curves")


# ================================================================
# Fig 3: Error vs time (temporal stability)
# ================================================================

def fig_error_vs_time(df: pd.DataFrame):
    alpha_val = 1.0
    sub = df[np.isclose(df["alpha"], alpha_val)]
    if sub.empty:
        print("  Skipping fig_error_vs_time")
        return

    K0_vals = sorted(sub["K0"].unique())
    # pick a few representative K0
    K0_show = [K0_vals[i] for i in [0, len(K0_vals)//3, 2*len(K0_vals)//3, -1]]
    K0_show = sorted(set(K0_show))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(K0_show)))

    for i, K0 in enumerate(K0_show):
        ss = sub[np.isclose(sub["K0"], K0)].sort_values("t")
        axes[0].semilogy(ss["t"], ss["err_rho_vs_full"], "o-", color=cmap[i],
                        label=rf"$K_0={K0}$", markersize=4)
        axes[1].semilogy(ss["t"], ss["err_Z_frob"], "o-", color=cmap[i],
                        label=rf"$K_0={K0}$", markersize=4)

    axes[0].set_xlabel("$t$"); axes[0].set_ylabel(r"$\varepsilon_\rho$")
    axes[0].set_title("Density error vs time")
    axes[1].set_xlabel("$t$"); axes[1].set_ylabel(r"$\|\Delta Z\|_F$")
    axes[1].set_title(r"$\|\Delta Z\|_F$ vs time")
    for ax in axes:
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, "v1_error_vs_time")

    # LaTeX table
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        rf"\caption{{Temporal stability data ($\alpha={alpha_val}$, Figure~\ref{{fig:v1_error_vs_time}}).}}",
        r"\label{tab:v1_error_vs_time}",
        r"\small",
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"$K_0$ & $t$ & $\varepsilon_\rho$ & $\|\Delta Z\|_F$ & $\varepsilon_{E_\mathrm{LP}}$ \\",
        r"\midrule",
    ]
    for K0 in K0_show:
        ss = sub[np.isclose(sub["K0"], K0)].sort_values("t")
        for _, row in ss.iterrows():
            lines.append(
                f"{row['K0']:.1f} & {row['t']:.1f} & "
                f"{row['err_rho_vs_full']:.2e} & {row['err_Z_frob']:.2e} & "
                f"{row['err_E_LP']:.2e} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _save_table(lines, "v1_error_vs_time")


# ================================================================
# Fig 4: Error vs coupling strength alpha (at fixed K0, t)
# ================================================================

def fig_error_vs_alpha(df: pd.DataFrame):
    t_val = 0.5
    sub = df[np.isclose(df["t"], t_val)]
    if sub.empty:
        return

    K0_vals = sorted(sub["K0"].unique())
    K0_show = [K0_vals[i] for i in [len(K0_vals)//4, len(K0_vals)//2, -1]]
    K0_show = sorted(set(K0_show))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(K0_show)))

    for i, K0 in enumerate(K0_show):
        ss = sub[np.isclose(sub["K0"], K0)].sort_values("alpha")
        axes[0].semilogy(ss["alpha"], ss["err_rho_vs_full"], "o-", color=cmap[i],
                        label=rf"$K_0={K0}$", markersize=5)
        axes[1].semilogy(ss["alpha"], ss["leakage_apriori"], "s--", color=cmap[i],
                        label=rf"$K_0={K0}$ (leak)", markersize=5)
        axes[1].semilogy(ss["alpha"], ss["err_Z_frob"], "o-", color=cmap[i],
                        label=rf"$K_0={K0}$ ($\Delta Z$)", markersize=5)

    axes[0].set_xlabel(r"$\alpha$"); axes[0].set_ylabel(r"$\varepsilon_\rho$")
    axes[0].set_title("Density error vs coupling strength")
    axes[1].set_xlabel(r"$\alpha$"); axes[1].set_ylabel("Leakage / Error")
    axes[1].set_title("Leakage vs coupling strength")
    for ax in axes:
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, "v1_error_vs_alpha")

    # LaTeX table
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        rf"\caption{{Coupling strength data ($t={t_val}$, Figure~\ref{{fig:v1_error_vs_alpha}}).}}",
        r"\label{tab:v1_error_vs_alpha}",
        r"\small",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"$\alpha$ & $K_0$ & $\varepsilon_\rho$ & $\ell_\mathrm{rms}$ & $\|\Delta Z\|_F$ & $\varepsilon_{E_\mathrm{LP}}$ \\",
        r"\midrule",
    ]
    for K0 in K0_show:
        ss = sub[np.isclose(sub["K0"], K0)].sort_values("alpha")
        for _, row in ss.iterrows():
            lines.append(
                f"{row['alpha']:.1f} & {row['K0']:.1f} & "
                f"{row['err_rho_vs_full']:.2e} & {row['leakage_apriori']:.2e} & "
                f"{row['err_Z_frob']:.2e} & {row['err_E_LP']:.2e} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _save_table(lines, "v1_error_vs_alpha")


def main():
    if not RESULTS.exists():
        print(f"ERROR: {RESULTS} not found. Run run_v1_sweep.py first.")
        return 1

    df = load_data()
    print(f"Loaded {len(df)} rows from {RESULTS}")
    print(f"  alphas: {sorted(df['alpha'].unique())}")
    print(f"  K0s: {sorted(df['K0'].unique())}")
    print(f"  ts: {sorted(df['t'].unique())}")

    print("\n--- Generating figures ---")
    fig_error_vs_K0(df)
    fig_three_curves(df)
    fig_error_vs_time(df)
    fig_error_vs_alpha(df)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
