# exp02_all_noise_mae_vs_budget_nl0.05_ijcai.py
# Purpose:
#   Exp02 (nl=0.05): plot MAE vs budget for three noise types (iq_chain/chain/realistic_chain)
#   in a single 1x3 IJCAI-style figure (shared y-axis, shared legend on the right).
# Notes:
#   - Baselines are from:
#       exp02_summary_mean_abs_error_vs_budget_{noise}_nl0.05(2).csv
#   - OnlyAgent is only available for iq_chain from:
#       exp02_onlyagent_budget_sweep_mean_abs_error_iq_chain_nl0.05.csv
#   - In WSL/CLI, we force a non-interactive backend to avoid hanging on GUI/font issues.

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")  # helps in some WSL setups

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib
matplotlib.use("Agg")  # save files without GUI
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


# -------- paths (edit if needed) --------
BASELINE_PATHS = {
    "iq_chain": "exp02_summary_mean_abs_error_vs_budget_iq_chain_nl0.05(2).csv",
    "chain": "exp02_summary_mean_abs_error_vs_budget_chain_nl0.05(2).csv",
    "realistic_chain": "exp02_summary_mean_abs_error_vs_budget_realistic_chain_nl0.05(2).csv",
}
ONLYAGENT_IQ_PATH = "exp02_onlyagent_budget_sweep_mean_abs_error_iq_chain_nl0.05.csv"

OUT_PNG = "exp02_all_noise_mae_vs_budget_nl0.05_ijcai.png"
OUT_PDF = "exp02_all_noise_mae_vs_budget_nl0.05_ijcai.pdf"


def load_budget_table(csv_path: str):
    """Load a budget-sweep table: first column is method; other columns are budgets (ints)."""
    df = pd.read_csv(csv_path)
    id_col = df.columns[0]  # usually "method"
    budget_cols = [c for c in df.columns if c != id_col]
    budgets = np.array(sorted([int(c) for c in budget_cols]))
    labels = df[id_col].astype(str).tolist()
    series = {
        lab: df.loc[i, [str(b) for b in budgets]].astype(float).to_numpy()
        for i, lab in enumerate(labels)
    }
    return budgets, labels, series


def main():
    # IJCAI-like style
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
    })

    # Load baseline tables
    tables = {k: load_budget_table(v) for k, v in BASELINE_PATHS.items()}

    # Load OnlyAgent (iq_chain only)
    a_series, budgets_a = None, None
    try:
        df_a = pd.read_csv(ONLYAGENT_IQ_PATH)
        budgets_a = np.array(sorted([int(c) for c in df_a.columns if c != "method"]))
        a_series = df_a.loc[0, [str(b) for b in budgets_a]].astype(float).to_numpy()
    except Exception:
        a_series, budgets_a = None, None

    # Global y-limit
    global_ymax = 0.0
    for _, (budgets, _, series) in tables.items():
        global_ymax = max(global_ymax, max(float(np.nanmax(v)) for v in series.values()))
    if a_series is not None:
        global_ymax = max(global_ymax, float(np.nanmax(a_series)))

    # Stable markers across methods
    all_methods = []
    for _, (_, labels, _) in tables.items():
        for l in labels:
            if l not in all_methods:
                all_methods.append(l)
    if "OnlyAgent" not in all_methods:
        all_methods.append("OnlyAgent")

    marker_cycle = ["o", "s", "^", "v", "P", "X", "D", "<", ">"]
    mk = {lab: marker_cycle[i % len(marker_cycle)] for i, lab in enumerate(all_methods)}
    mk["OnlyAgent"] = "D"

    # Figure: 1x3 panels
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), sharey=True)

    panel_order = ["iq_chain", "chain", "realistic_chain"]
    panel_titles = {"iq_chain": "iq_chain", "chain": "chain", "realistic_chain": "realistic_chain"}
    panel_letters = ["(a)", "(b)", "(c)"]

    handles = {}

    for ax, key, letter in zip(axes, panel_order, panel_letters):
        budgets, labels, series = tables[key]

        # Plot baselines
        for lab in labels:
            line, = ax.plot(
                budgets, series[lab],
                marker=mk.get(lab, "o"),
                linewidth=1.0,
                markersize=3.0,
                alpha=0.75,
                label=lab
            )
            if lab not in handles:
                handles[lab] = line

        # Plot OnlyAgent only in iq_chain panel (if available)
        if key == "iq_chain" and a_series is not None:
            # align budgets if needed
            if not np.array_equal(budgets, budgets_a):
                y = np.interp(budgets, budgets_a, a_series)
            else:
                y = a_series
            line, = ax.plot(
                budgets, y,
                marker=mk["OnlyAgent"],
                linewidth=2.0,
                markersize=3.8,
                label="OnlyAgent"
            )
            if "OnlyAgent" not in handles:
                handles["OnlyAgent"] = line

        ax.set_title(f"{letter} {panel_titles[key]}", fontsize=8, pad=2)
        ax.set_xlabel("Budget (scan points)", fontsize=8, labelpad=2)

        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis="both", which="major", length=3.0, pad=1.2, labelsize=7)
        ax.tick_params(axis="both", which="minor", length=1.6)

        ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)

    axes[0].set_ylabel("MAE", fontsize=8, labelpad=2)

    # Consistent limits
    xmin = min(float(np.min(tables[k][0])) for k in panel_order)
    xmax = max(float(np.max(tables[k][0])) for k in panel_order)
    for ax in axes:
        ax.set_xlim(xmin - 20, xmax + 20)
        ax.set_ylim(0, global_ymax * 1.05)

    # Shared legend on the right (OnlyAgent first)
    legend_labels = list(handles.keys())
    if "OnlyAgent" in legend_labels:
        legend_labels.remove("OnlyAgent")
        legend_labels = ["OnlyAgent"] + sorted(legend_labels)
    else:
        legend_labels = sorted(legend_labels)

    legend_handles = [handles[l] for l in legend_labels]
    leg = fig.legend(
        legend_handles, legend_labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=6.5,
        frameon=True,
        fancybox=False,
        framealpha=0.9,
        borderpad=0.25,
        handlelength=1.6,
        handletextpad=0.4
    )
    leg.get_frame().set_linewidth(0.6)

    fig.tight_layout(pad=0.4, rect=(0, 0, 0.86, 1))

    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print("Saved:")
    print(" -", OUT_PNG)
    print(" -", OUT_PDF)


if __name__ == "__main__":
    main()
