# exp01_all_noise_agent1_vs_baselines_mae_vs_nl_ijcai.py
# Usage:
#   1) Put this script in the same folder as the CSVs (or edit the paths below)
#   2) python exp01_all_noise_agent1_vs_baselines_mae_vs_nl_ijcai.py
#
# Output:
#   - exp01_all_noise_agent1_vs_baselines_mae_vs_nl_ijcai.png  (600 dpi)
#   - exp01_all_noise_agent1_vs_baselines_mae_vs_nl_ijcai.pdf  (vector)

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


# -------- paths (edit if needed) --------
PATHS = {
    "iq_chain": "exp01_f5_baseline+agent__iq_chain__mean_abs_err_MHz(1).csv",
    "chain": "exp01_f5_baseline+agent__chain__mean_abs_err_MHz(1).csv",
    "realistic_chain": "exp01_f5_baseline+agent__realistic_chain__mean_abs_err_MHz(1).csv",
}

OUT_PNG = "exp01_all_noise_agent1_vs_baselines_mae_vs_nl_ijcai.png"
OUT_PDF = "exp01_all_noise_agent1_vs_baselines_mae_vs_nl_ijcai.pdf"


def load_table(csv_path: str):
    """Load one table: first col is method name; remaining cols are noise levels (nl)."""
    df = pd.read_csv(csv_path)
    id_col = df.columns[0]
    x_cols = [c for c in df.columns if c != id_col]

    # numeric order on x
    xs = np.array(sorted([float(c) for c in x_cols]))
    col_map = {float(c): c for c in x_cols}
    x_cols_ordered = [col_map[x] for x in xs]

    labels = df[id_col].astype(str).tolist()
    series = {lab: df.loc[i, x_cols_ordered].astype(float).to_numpy() for i, lab in enumerate(labels)}
    return xs, labels, series


def pick_agent_label(labels):
    """Robustly find Agent1 row name if variant exists."""
    for cand in labels:
        if cand.lower() in {"agent1", "a1", "onlyagent", "agent"}:
            return cand
    return "Agent1" if "Agent1" in labels else None


def main():
    # IJCAI-like styling
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "pdf.fonttype": 42,  # editable text in illustrator
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
    })

    # Load all three
    tables = {k: load_table(v) for k, v in PATHS.items()}

    # Global y-limit and x-range
    global_ymax = 0.0
    x_min, x_max = None, None
    for _, (xs, labels, series) in tables.items():
        global_ymax = max(global_ymax, max(float(np.nanmax(y)) for y in series.values()))
        x_min = float(np.min(xs)) if x_min is None else min(x_min, float(np.min(xs)))
        x_max = float(np.max(xs)) if x_max is None else max(x_max, float(np.max(xs)))

    # Stable marker assignment across panels
    all_labels = []
    for _, (_, labels, _) in tables.items():
        for l in labels:
            if l not in all_labels:
                all_labels.append(l)

    marker_cycle = ["o", "s", "^", "v", "P", "X", "D", "<", ">"]
    mk = {lab: marker_cycle[i % len(marker_cycle)] for i, lab in enumerate(all_labels)}
    mk["Agent1"] = "D"

    # Figure: 1x3 panels, shared y
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), sharey=True)
    panel_order = ["iq_chain", "chain", "realistic_chain"]
    panel_titles = {"iq_chain": "iq_chain", "chain": "chain", "realistic_chain": "realistic_chain"}
    panel_letters = ["(a)", "(b)", "(c)"]

    # Collect handles for a shared legend
    handles_for_legend = {}

    for ax, key, letter in zip(axes, panel_order, panel_letters):
        xs, labels, series = tables[key]
        agent_label = pick_agent_label(labels)
        baseline_labels = [l for l in labels if l != agent_label]

        # Baselines (fainter)
        for lab in baseline_labels:
            line, = ax.plot(
                xs, series[lab],
                marker=mk.get(lab, "o"),
                linewidth=1.0,
                markersize=3.0,
                alpha=0.65,
                label=lab
            )
            if lab not in handles_for_legend:
                handles_for_legend[lab] = line

        # Agent emphasized
        if agent_label is not None:
            line, = ax.plot(
                xs, series[agent_label],
                marker=mk.get("Agent1", "D"),
                linewidth=2.0,
                markersize=3.8,
                label="Agent1"
            )
            if "Agent1" not in handles_for_legend:
                handles_for_legend["Agent1"] = line

        ax.set_title(f"{letter} {panel_titles[key]}", fontsize=8, pad=2)
        ax.set_xlabel("Noise level (nl)", fontsize=8, labelpad=2)

        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis="both", which="major", length=3.0, pad=1.2, labelsize=7)
        ax.tick_params(axis="both", which="minor", length=1.6)

        ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, global_ymax * 1.05)

    axes[0].set_ylabel("MAE", fontsize=8, labelpad=2)

    # Shared legend at right: Agent1 first, others sorted
    legend_labels = list(handles_for_legend.keys())
    if "Agent1" in legend_labels:
        legend_labels.remove("Agent1")
        legend_labels = ["Agent1"] + sorted(legend_labels)
    else:
        legend_labels = sorted(legend_labels)

    legend_handles = [handles_for_legend[l] for l in legend_labels]
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
