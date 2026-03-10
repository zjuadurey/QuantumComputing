"""
plot_v1_all.py — Generate all 5 experiment figures for ShadowFluid paper.
Reads data from sweep_v1.csv.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from pathlib import Path

OUT = Path(__file__).resolve().parent
CSV = Path(__file__).resolve().parent.parent / "data/sweep_v1.csv"

# ═══════════════════════════════════════════════════════════════════════════
# Tunable parameters (centralized)
# ═══════════════════════════════════════════════════════════════════════════

# ── Figure size (width, height) ──────────────────────────────────────────
FIGSIZE_WIDE   = (7.2, 2.0)   # double-column 3-panel (Fig1, Fig5)
FIGSIZE_NARROW = (3.5, 2.0)   # single-column 2-panel (Fig3, Fig4)
FIGSIZE_BAR    = (3.5, 2.0)   # single-column bar chart (Fig2)

# ── Line width & marker size ──────────────────────────────────────────────
LW_WIDE    = 2.0    # double-column line width
LW_NARROW  = 1.8    # single-column line width
MS_WIDE    = 3.5      # double-column marker size
MS_NARROW  = 3      # single-column marker size

# ── Font ─────────────────────────────────────────────────────────────────
FONT_SIZE       = 10    # global font size
AXIS_LABEL_SIZE = 10    # axis label font size
TITLE_SIZE      = 11    # subplot title font size
LEGEND_SIZE     = 8     # legend font size
TICK_SIZE       = 9     # tick label font size

# ── Line colors (>=5 lines) ──────────────────────────────────────────────
LINE_COLORS  = ["#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd", "#d62728", "#e377c2"]
LINE_STYLES  = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
LINE_MARKERS = ["o", "s", "^", "*", "D", "p"]

# ── Line colors (<=4 lines) ──────────────────────────────────────────────
LINE4_COLORS  = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
LINE4_STYLES  = ["-", "--", "-.", ":"]
LINE4_MARKERS = ["o", "s", "^", "D"]

# ── Bar chart colors ───────────────────────────────────────────────────
BAR_COLORS  = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51"]
BAR_HATCHES = ["///", "\\\\\\", "xxx", "..."]
BAR_EDGE    = "#333333"
BAR_WIDTH   = 0.19      # width per bar group

# ── Grid ─────────────────────────────────────────────────────────────────
GRID_STYLE = "--"
GRID_ALPHA = 0.5
GRID_COLOR = "#999999"

# ═══════════════════════════════════════════════════════════════════════════
# Apply global style
# ═══════════════════════════════════════════════════════════════════════════
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": FONT_SIZE,
    "axes.labelsize": AXIS_LABEL_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelweight": "normal",
    "legend.fontsize": LEGEND_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "axes.grid": True,
    "grid.linestyle": GRID_STYLE,
    "grid.alpha": GRID_ALPHA,
    "grid.color": GRID_COLOR,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# ── Load data ────────────────────────────────────────────────────────────
NX = 5  # default grid size for figures

def load_tier1(nx=NX):
    """Load Tier-1 (J=1) data for given nx, clean alpha."""
    df = pd.read_csv(CSV)
    t1 = df[(df['J'] == 1) & (df['nx'] == nx)].copy()
    t1['alpha'] = t1['alpha'].round(1)
    valid_alphas = [round(0.1 * i, 1) for i in range(1, 11)]
    t1 = t1[t1['alpha'].isin(valid_alphas)]
    t1 = t1.drop_duplicates(subset=['alpha', 'K0', 't'], keep='first')
    return t1

def load_tier2(nx=NX):
    """Load Tier-2 (J>1) data for given nx."""
    df = pd.read_csv(CSV)
    return df[(df['J'] > 1) & (df['nx'] == nx)]

def get_slice(t1, alpha=None, t=None, K0=None):
    """Filter tier1 data by fixed parameters."""
    s = t1
    if alpha is not None: s = s[s['alpha'] == alpha]
    if t is not None:     s = s[s['t'] == t]
    if K0 is not None:    s = s[s['K0'] == K0]
    return s.sort_values(['alpha', 'K0', 't'])


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def _line(ax, x, y, idx, label, n_total=6, lw=None, ms=None):
    _lw = lw or LW_WIDE
    _ms = ms or MS_WIDE
    if n_total <= 4:
        c, ls, mk = LINE4_COLORS[idx], LINE4_STYLES[idx], LINE4_MARKERS[idx]
    else:
        c, ls, mk = LINE_COLORS[idx], LINE_STYLES[idx], LINE_MARKERS[idx]
    ax.plot(x, y, color=c, linestyle=ls, marker=mk,
            linewidth=_lw, markersize=_ms, markeredgecolor=c,
            markerfacecolor=c, label=label, zorder=3)


def _set_K0_ticks(ax, K0_vals):
    ax.set_xticks(K0_vals)
    ax.set_xticklabels([f"{int(v)}" for v in K0_vals])
    ax.set_xlim(K0_vals[0] - 0.5, K0_vals[-1] + 0.5)


def _compact_log_yaxis(ax):
    """Factor out common 10^n: ticks show mantissa, axis top shows ×10^n."""
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    # Get data range from all plotted lines
    all_y = []
    for line in ax.get_lines():
        yd = line.get_ydata()
        all_y.extend(yd[yd > 0])
    if not all_y:
        return
    ymin, ymax = min(all_y), max(all_y)
    span = np.log10(ymax / ymin) if ymin > 0 else 0

    if span <= 3:
        # Narrow range: linear scale + ScalarFormatter offset
        ax.set_yscale("linear")
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(fmt)
        # Re-set limits to data range with padding
        margin = (ymax - ymin) * 0.08
        ax.set_ylim(max(0, ymin - margin), ymax + margin)
    else:
        # Wide range: keep log, but use compact exponent-only labels
        from matplotlib.ticker import FuncFormatter
        def fmt_exp(x, pos):
            if x <= 0:
                return ""
            exp = np.log10(x)
            r = round(exp)
            if abs(exp - r) < 0.01:
                return rf"$10^{{{int(r)}}}$"
            return ""
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_exp))
        ax.yaxis.set_minor_formatter(NullFormatter())


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: error_vs_K0 (3-panel, figure*)
#   α ∈ {0.2, 0.4, 0.6, 0.8, 1.0}, t=0.5
# ═══════════════════════════════════════════════════════════════════════════
def fig_error_vs_K0():
    t1 = load_tier1()
    sel_alphas = [0.2, 0.4, 0.6, 0.8, 1.0]
    sub = get_slice(t1, t=0.5)

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_WIDE, sharey=False)
    titles = [
        r"Density error $\varepsilon_\rho$",
        r"Energy error $\varepsilon_{E_\mathrm{LP}}$",
        r"Leakage $\ell_\mathrm{rms}$",
    ]
    cols = ['err_rho_vs_full', 'err_E_LP', 'leakage_apriori']

    for ax, ylabel, col in zip(axes, titles, cols):
        for i, alpha in enumerate(sel_alphas):
            d = sub[sub['alpha'] == alpha].sort_values('K0')
            _line(ax, d['K0'].values, d[col].values, i,
                  rf"$\alpha\!=\!{alpha}$", lw=LW_WIDE, ms=MS_WIDE)
        ax.set_yscale("log")
        ax.set_xlabel(r"$K_0$")
        ax.set_ylabel(ylabel)
        _set_K0_ticks(ax, sub['K0'].unique())
        _compact_log_yaxis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(sel_alphas),
               fontsize=9, framealpha=0.9, handlelength=1.8, columnspacing=1.0,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(w_pad=1.0)
    fig.subplots_adjust(top=0.82)
    fig.savefig(OUT / "v1_error_vs_K0.pdf")
    print("Saved v1_error_vs_K0.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: three_curves bar chart
#   α=0.5, t=0.5
# ═══════════════════════════════════════════════════════════════════════════
def fig_three_curves():
    t1 = load_tier1()
    sub = get_slice(t1, alpha=0.5, t=0.5).sort_values('K0')
    K0s = sub['K0'].values

    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
    x = np.arange(len(K0s))

    metrics = [
        sub['bound_apriori'].values,
        sub['err_Z_frob'].values,
        sub['err_rho_vs_full'].values,
        sub['err_E_LP'].values,
    ]
    names = [
        r"Bound ($\ell_\mathrm{rms}$)",
        r"$\|\Delta Z\|_F$",
        r"$\varepsilon_\rho$",
        r"$\varepsilon_{E,LP}$",
    ]

    for i, (vals, name) in enumerate(zip(metrics, names)):
        offset = (i - 1.5) * BAR_WIDTH
        ax.bar(x + offset, vals, BAR_WIDTH,
               color=BAR_COLORS[i], hatch=BAR_HATCHES[i],
               edgecolor=BAR_EDGE, linewidth=0.6,
               label=name, zorder=3)

    ax.set_yscale("log")
    ax.set_xlabel(r"$K_0$")
    ax.set_ylabel(r"Error hierarchy")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(k)}" for k in K0s])
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               fontsize=8, framealpha=0.9, handlelength=1.5, columnspacing=0.8,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    fig.savefig(OUT / "v1_three_curves_alpha1_t0_5.pdf")
    print("Saved v1_three_curves_alpha1_t0_5.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: error_vs_time (2-panel)
#   α=0.5, K0 ∈ {2, 5, 8, 10}
# ═══════════════════════════════════════════════════════════════════════════
def fig_error_vs_time():
    t1 = load_tier1()
    sel_K0 = [2.0, 5.0, 8.0, 10.0]
    sub = get_slice(t1, alpha=0.5)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_NARROW)
    titles = [r"$\varepsilon_\rho(t)$", r"$\|\Delta Z(t)\|_F$"]
    cols = ['err_rho_vs_full', 'err_Z_frob']

    for ax, ylabel, col in zip(axes, titles, cols):
        for i, k0 in enumerate(sel_K0):
            d = sub[sub['K0'] == k0].sort_values('t')
            _line(ax, d['t'].values, d[col].values, i,
                  rf"$K_0\!=\!{int(k0)}$", n_total=4,
                  lw=LW_NARROW, ms=MS_NARROW)
        ax.set_yscale("log")
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(ylabel)
        ax.set_xticks([0.2, 0.5, 0.8])
        ax.set_xticklabels(["0.2", "0.5", "0.8"])
        ax.set_xlim(0.0, 1.1)
        _compact_log_yaxis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(sel_K0),
               fontsize=9, framealpha=0.9, handlelength=1.8, columnspacing=1.0,
               bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=1.0)
    fig.subplots_adjust(top=0.78)
    fig.savefig(OUT / "v1_error_vs_time.pdf")
    print("Saved v1_error_vs_time.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4: error_vs_alpha (2-panel)
#   K0 ∈ {3, 6, 10}, t=0.5
# ═══════════════════════════════════════════════════════════════════════════
def fig_error_vs_alpha():
    t1 = load_tier1()
    sel_K0 = [3.0, 6.0, 10.0]
    sub = get_slice(t1, t=0.5)
    alpha_vals = sorted(sub['alpha'].unique())

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_NARROW)

    # Left panel: ε_ρ vs α
    ax = axes[0]
    for i, k0 in enumerate(sel_K0):
        d = sub[sub['K0'] == k0].sort_values('alpha')
        _line(ax, d['alpha'].values, d['err_rho_vs_full'].values, i,
              rf"$K_0\!=\!{int(k0)}$", n_total=3, lw=LW_NARROW, ms=MS_NARROW)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\varepsilon_\rho$")
    ax.set_xticks([0.2, 0.5, 0.8])
    ax.set_xticklabels(["0.2", "0.5", "0.8"])
    ax.set_xlim(0.0, 1.1)
    _compact_log_yaxis(ax)

    # Right panel: ℓ_rms (dashed) + ||ΔZ||_F (solid) vs α
    MS_FILL = 5   # larger markers so filled vs hollow is visible
    ax = axes[1]
    for i, k0 in enumerate(sel_K0):
        d = sub[sub['K0'] == k0].sort_values('alpha')
        c = LINE4_COLORS[i]
        mk = LINE4_MARKERS[i]
        ls = LINE4_STYLES[i]
        ax.plot(d['alpha'].values, d['err_Z_frob'].values,
                color=c, linestyle=ls, marker=mk,
                linewidth=LW_NARROW, markersize=MS_FILL,
                markeredgecolor=c, markerfacecolor=c, zorder=3)
        ax.plot(d['alpha'].values, d['leakage_apriori'].values,
                color=c, linestyle=ls, marker=mk,
                linewidth=LW_NARROW, markersize=MS_FILL,
                markeredgecolor=c, markerfacecolor="white", zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\ell_\mathrm{rms}$  &  $\|\Delta Z\|_F$")
    _compact_log_yaxis(ax)

    # — Build unified legend for both panels (K0 lines + fill/hollow) —
    handles = []
    for i, k0 in enumerate(sel_K0):
        handles.append(Line2D([0], [0], color=LINE4_COLORS[i],
                              linestyle=LINE4_STYLES[i],
                              marker=LINE4_MARKERS[i],
                              markersize=5, label=rf"$K_0\!=\!{int(k0)}$"))
    handles.append(Line2D([0], [0], color="gray", marker="o", linestyle="",
                          markersize=6, markerfacecolor="gray",
                          label=r"$\|\Delta Z\|_F$ (filled)"))
    handles.append(Line2D([0], [0], color="gray", marker="o", linestyle="",
                          markersize=6, markerfacecolor="white",
                          label=r"$\ell_\mathrm{rms}$ (hollow)"))
    fig.legend(handles=handles, loc="upper center",
               ncol=len(sel_K0) + 2, fontsize=8, framealpha=0.9,
               handlelength=1.8, columnspacing=0.8,
               bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(w_pad=0.8)
    fig.subplots_adjust(top=0.80)
    fig.savefig(OUT / "v1_error_vs_alpha.pdf")
    print("Saved v1_error_vs_alpha.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5: tier2_multi (3-panel, figure*)
#   J ∈ {2, 4, 6, 8, 10}, t=0.5, + Tier-1 baselines
# ═══════════════════════════════════════════════════════════════════════════
def fig_tier2_multi():
    t1 = load_tier1()
    t2 = load_tier2()

    if len(t2) == 0:
        print("SKIP v1_tier2_multi.pdf — no Tier-2 data in CSV")
        return

    sel_J = [2, 4, 6, 8, 10]
    # Only use J values actually present
    sel_J = [j for j in sel_J if j in t2['J'].unique()]
    K0_vals = sorted(t2['K0'].unique())

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_WIDE, sharey=False)
    titles = [
        r"$\varepsilon_\rho$",
        r"$\varepsilon_{E_\mathrm{LP}}$",
        r"$\|\Delta Z\|_F$",
    ]
    cols = ['err_rho_vs_full', 'err_E_LP', 'err_Z_frob']

    multi_colors = ["#2ca02c", "#1f77b4", "#d62728", "#9467bd", "#e377c2"]
    multi_markers = ["o", "s", "^", "D", "p"]
    multi_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

    # Tier-1 baseline: α=0.5, t=0.5
    base = get_slice(t1, alpha=0.5, t=0.5).sort_values('K0')

    for ax, ylabel, col in zip(axes, titles, cols):
        for i, J in enumerate(sel_J):
            d = t2[t2['J'] == J].sort_values('K0')
            ax.plot(d['K0'].values, d[col].values,
                    color=multi_colors[i], linestyle=multi_styles[i],
                    marker=multi_markers[i],
                    linewidth=LW_WIDE, markersize=MS_WIDE,
                    markeredgecolor=multi_colors[i],
                    markerfacecolor=multi_colors[i],
                    label=rf"$J\!=\!{J}$",
                    zorder=3)
        # Baseline (dashed)
        ax.plot(base['K0'].values, base[col].values,
                color="#ff7f0e", linestyle="--", marker="*",
                linewidth=LW_WIDE, markersize=MS_WIDE - 1,
                markeredgecolor="#ff7f0e", markerfacecolor="white",
                label=r"$\alpha\!=\!0.5$", zorder=3)
        ax.set_yscale("log")
        ax.set_xlabel(r"$K_0$")
        ax.set_ylabel(ylabel)
        _set_K0_ticks(ax, np.array(K0_vals))
        _compact_log_yaxis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(sel_J) + 1,
               fontsize=9, framealpha=0.9, handlelength=1.8, columnspacing=1.0,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(w_pad=1.0)
    fig.subplots_adjust(top=0.82)
    fig.savefig(OUT / "v1_tier2_multi.pdf")
    print("Saved v1_tier2_multi.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    fig_error_vs_K0()
    fig_three_curves()
    fig_error_vs_time()
    fig_error_vs_alpha()
    fig_tier2_multi()
    print("\nAll 5 figures generated.")
