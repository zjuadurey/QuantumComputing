"""experiments/plot_advanced_gallery.py

Advanced and creative figure variants:
- Dual-axis plots
- Contour/heatmap advanced versions
- Comparative metrics with confidence
- Custom-styled panels
- KDD-specific narrative plots
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.gridspec import GridSpec
from scipy import stats, interpolate
import matplotlib.patches as mpatches

import plot_common as pc


def setup_kdd_style(fontsize: int = 10):
    """KDD conference style: technical, clear hierarchy."""
    plt.rcParams.update({
        "font.size": fontsize,
        "font.family": "sans-serif",
        "axes.labelsize": fontsize + 1,
        "axes.titlesize": fontsize + 2,
        "legend.fontsize": fontsize - 1,
        "xtick.labelsize": fontsize - 1,
        "ytick.labelsize": fontsize - 1,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "axes.linewidth": 0.8,
        "grid.alpha": 0.3,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })


# ============================================================================
# Type I: Dual-axis and Composite Plots
# ============================================================================

def plot_i1_error_vs_cost_dual_axis(rows, figdir):
    """I1: Dual-axis: error (log left) vs qubit cost efficiency (right)."""
    setup_kdd_style()

    t_eval = pc.max_t(rows)
    rows_t = pc.filter_close(rows, "t", t_eval)

    nx_vals = sorted([int(v) for v in pc.unique_sorted(rows_t, "nx")])
    K0_vals = sorted([float(v) for v in pc.unique_sorted(rows_t, "K0")])

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)

    colors = ["#0072B2", "#D55E00", "#009E73"]

    # Left axis: Error
    for i, nx in enumerate(nx_vals):
        errors = []
        for K0 in K0_vals:
            rs = [r for r in rows_t if r.get("nx") == nx and r.get("K0") == K0]
            mu, _, _ = pc.mean_std([r.get("err_rho_vs_full") for r in rs])
            errors.append(mu)

        ax1.semilogy(K0_vals, errors, 'o-', color=colors[i], linewidth=2.2,
                    markersize=6, label=f"nx={nx}", alpha=0.85)

    ax1.set_xlabel("Truncation Cutoff (K₀)", fontweight="bold", fontsize=12)
    ax1.set_ylabel("Density Error (log scale)", fontweight="bold", fontsize=12, color=colors[0])
    ax1.tick_params(axis='y', labelcolor=colors[0])
    ax1.grid(True, alpha=0.3, which="both")
    ax1.legend(loc="upper right", frameon=True, fontsize=10)

    # Right axis: Cost efficiency (modes kept per qubit)
    ax2 = ax1.twinx()
    nx_choice = nx_vals[0]
    M_vals = []
    q_shift_vals = []
    for K0 in K0_vals:
        rs = [r for r in rows_t if r.get("nx") == nx_choice and r.get("K0") == K0]
        m_mean, _, _ = pc.mean_std([r.get("M") for r in rs])
        q_shift_mean, _, _ = pc.mean_std([r.get("q_shift") for r in rs])
        M_vals.append(m_mean)
        q_shift_vals.append(q_shift_mean)

    efficiency = np.array(M_vals) / np.array(q_shift_vals)
    ax2.plot(K0_vals, efficiency, 's--', color="#E69F00", linewidth=2.5,
            markersize=7, label="Mode efficiency", alpha=0.75)
    ax2.set_ylabel("Modes per Qubit (M/q)", fontweight="bold", fontsize=12, color="#E69F00")
    ax2.tick_params(axis='y', labelcolor="#E69F00")

    fig.suptitle(f"Trade-off Landscape: Accuracy vs Quantum Efficiency (t={t_eval:g})",
                fontsize=13, fontweight="bold", y=0.98)
    out = figdir / "i1_dual_axis_error_efficiency.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")


def plot_i2_runtime_vs_error_bubble(rows, figdir):
    """I2: Bubble plot: x=error, y=speedup, size=system size."""
    setup_kdd_style()

    t_eval = pc.max_t(rows)
    rows_t = pc.filter_close(rows, "t", t_eval)

    nx_vals = sorted([int(v) for v in pc.unique_sorted(rows_t, "nx")])
    K0_vals = sorted([float(v) for v in pc.unique_sorted(rows_t, "K0")])

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)

    colors = ["#1b9e77", "#d95f02", "#7570b3"]

    for i, nx in enumerate(nx_vals):
        errors = []
        speedups = []
        sizes = []

        for K0 in K0_vals:
            rs = [r for r in rows_t if r.get("nx") == nx and r.get("K0") == K0]
            err_mu, _, _ = pc.mean_std([r.get("err_rho_vs_full") for r in rs])
            rt_full, _, _ = pc.mean_std([r.get("rt_baseline_full_s") for r in rs])
            rt_shadow, _, _ = pc.mean_std([r.get("rt_shadow_s") for r in rs])

            errors.append(err_mu)
            if rt_shadow > 0:
                speedups.append(rt_full / rt_shadow)
                sizes.append(nx * 50)  # Size proportional to system

        scatter = ax.scatter(errors, speedups, s=sizes, c=[colors[i]] * len(errors),
                           alpha=0.6, edgecolor="black", linewidth=0.8, label=f"nx={nx}")

    ax.set_xscale("log")
    ax.set_xlabel("Density Error (log scale)", fontweight="bold", fontsize=12)
    ax.set_ylabel("Speedup (Full/Shadow)", fontweight="bold", fontsize=12)
    ax.set_title("Error-Speedup Trade-off: Bubble Size = System Size",
                fontweight="bold", fontsize=12, pad=15)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", frameon=True, fontsize=11, scatterpoints=1)

    # Add annotation for best region
    ax.text(0.05, 0.95, "← Lower error\nHigher speedup →", transform=ax.transAxes,
           fontsize=10, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.1))

    out = figdir / "i2_bubble_error_vs_speedup.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")


# ============================================================================
# Type J: Contour and 3D-style projections
# ============================================================================

def plot_j1_contour_error_landscape(rows, figdir):
    """J1: Contour plot of error landscape across (K0, nx)."""
    setup_kdd_style()

    t_eval = pc.max_t(rows)
    rows_t = pc.filter_close(rows, "t", t_eval)

    nx_vals = sorted([int(v) for v in pc.unique_sorted(rows_t, "nx")])
    K0_vals = sorted([float(v) for v in pc.unique_sorted(rows_t, "K0")])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), constrained_layout=True)

    for ax_idx, (ax, key, title) in enumerate([
        (axes[0], "err_rho_vs_full", "Density Error"),
        (axes[1], "err_momentum_vs_full", "Momentum Error"),
    ]):
        # Create grid
        X, Y = np.meshgrid(K0_vals, nx_vals)
        Z = np.zeros_like(X, dtype=float)

        for i, nx in enumerate(nx_vals):
            for j, K0 in enumerate(K0_vals):
                rs = [r for r in rows_t if r.get("nx") == nx and r.get("K0") == K0]
                mu, _, _ = pc.mean_std([r.get(key) for r in rs])
                Z[i, j] = np.log10(mu) if mu > 0 else -20

        # Contour plot
        levels = np.linspace(Z.min(), Z.max(), 12)
        contour = ax.contourf(X, Y, Z, levels=levels, cmap="RdYlGn_r", alpha=0.8)
        contour_lines = ax.contour(X, Y, Z, levels=levels, colors="black", alpha=0.2, linewidths=0.5)
        ax.clabel(contour_lines, inline=True, fontsize=7, fmt="%1.1f")

        ax.scatter(X, Y, c=Z, s=100, marker='o', edgecolor="black", linewidth=0.8,
                  cmap="RdYlGn_r", vmin=Z.min(), vmax=Z.max())

        ax.set_xlabel("Cutoff (K₀)", fontweight="bold", fontsize=11)
        ax.set_ylabel("System Size (nx)", fontweight="bold", fontsize=11)
        ax.set_title(f"{title} Landscape", fontweight="bold", fontsize=11)

        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label("log₁₀(error)", fontweight="bold")

    fig.suptitle(f"Error Landscape: Contour Map (t={t_eval:g})",
                fontsize=13, fontweight="bold")
    out = figdir / "j1_contour_error_landscape.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")


def plot_j2_3d_projection_scatter(rows, figdir):
    """J2: Scatter in 3D space: K0 vs error vs speedup."""
    setup_kdd_style()

    t_eval = pc.max_t(rows)
    rows_t = pc.filter_close(rows, "t", t_eval)

    nx_vals = sorted([int(v) for v in pc.unique_sorted(rows_t, "nx")])
    K0_vals = sorted([float(v) for v in pc.unique_sorted(rows_t, "K0")])

    fig = plt.figure(figsize=(11, 5), constrained_layout=True)

    # Create two different projections
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    colors_nx = ["#0072B2", "#D55E00", "#009E73"]

    # Projection 1: K0 vs Error (colored by speedup)
    for i, nx in enumerate(nx_vals):
        K0_plot = []
        err_plot = []
        speedup_plot = []

        for K0 in K0_vals:
            rs = [r for r in rows_t if r.get("nx") == nx and r.get("K0") == K0]
            err, _, _ = pc.mean_std([r.get("err_rho_vs_full") for r in rs])
            rt_full, _, _ = pc.mean_std([r.get("rt_baseline_full_s") for r in rs])
            rt_shadow, _, _ = pc.mean_std([r.get("rt_shadow_s") for r in rs])

            K0_plot.append(K0)
            err_plot.append(err)
            if rt_shadow > 0:
                speedup_plot.append(rt_full / rt_shadow)

        scatter1 = ax1.scatter(K0_plot, err_plot, c=speedup_plot, s=100, alpha=0.7,
                             edgecolor="black", linewidth=0.8, cmap="viridis", label=f"nx={nx}")

    ax1.set_yscale("log")
    ax1.set_xlabel("Truncation Cutoff (K₀)", fontweight="bold", fontsize=11)
    ax1.set_ylabel("Density Error", fontweight="bold", fontsize=11)
    ax1.set_title("Projection 1: K₀ vs Error\n(color = speedup)", fontweight="bold", fontsize=11)
    ax1.grid(True, alpha=0.3, which="both")

    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label("Speedup", fontweight="bold")

    # Projection 2: Error vs Speedup (colored by K0)
    K0_norm = (np.array(K0_vals) - min(K0_vals)) / (max(K0_vals) - min(K0_vals))
    cmap_k0 = plt.colormaps["plasma"]

    for i, nx in enumerate(nx_vals):
        for j, K0 in enumerate(K0_vals):
            rs = [r for r in rows_t if r.get("nx") == nx and r.get("K0") == K0]
            err, _, _ = pc.mean_std([r.get("err_rho_vs_full") for r in rs])
            rt_full, _, _ = pc.mean_std([r.get("rt_baseline_full_s") for r in rs])
            rt_shadow, _, _ = pc.mean_std([r.get("rt_shadow_s") for r in rs])

            if rt_shadow > 0:
                speedup = rt_full / rt_shadow
                ax2.scatter(err, speedup, s=120, alpha=0.7, edgecolor="black", linewidth=0.8,
                           color=cmap_k0(K0_norm[j]), marker=["o", "s", "^"][i])

    ax2.set_xscale("log")
    ax2.set_xlabel("Density Error (log)", fontweight="bold", fontsize=11)
    ax2.set_ylabel("Speedup (Full/Shadow)", fontweight="bold", fontsize=11)
    ax2.set_title("Projection 2: Error vs Speedup\n(color = K₀)", fontweight="bold", fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("3D Trade-off Space: K₀, Error, Speedup",
                fontsize=13, fontweight="bold")
    out = figdir / "j2_3d_projection_scatter.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")


# ============================================================================
# Type K: Efficiency Ratio and Normalized Comparisons
# ============================================================================

def plot_k1_normalized_error_comparison(rows, figdir):
    """K1: Normalized error: shadow / low-pass / full (% relative to max)."""
    setup_kdd_style()

    t_eval = pc.max_t(rows)
    rows_t = pc.filter_close(rows, "t", t_eval)

    nx_choice = sorted([int(v) for v in pc.unique_sorted(rows_t, "nx")])[0]
    rows_nx = [r for r in rows_t if r.get("nx") == nx_choice]
    K0_vals = sorted([float(v) for v in pc.unique_sorted(rows_nx, "K0")])

    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5), constrained_layout=True)

    x = np.arange(len(K0_vals))
    width = 0.25

    shadow_norm = []
    lp_norm = []

    for K0 in K0_vals:
        rs = [r for r in rows_nx if r.get("K0") == K0]
        shadow_mu, _, _ = pc.mean_std([r.get("err_rho_vs_full") for r in rs])
        lp_mu, _, _ = pc.mean_std([r.get("err_rho_lp_vs_full") for r in rs])

        shadow_norm.append(shadow_mu)
        lp_norm.append(lp_mu)

    max_err = max(max(shadow_norm), max(lp_norm))

    ax.bar(x - width/2, np.array(shadow_norm) / max_err * 100, width,
          label="Shadow", color="#0072B2", alpha=0.85)
    ax.bar(x + width/2, np.array(lp_norm) / max_err * 100, width,
          label="Low-pass (baseline)", color="#D55E00", alpha=0.85)

    ax.set_ylabel("Normalized Error (% of max)", fontweight="bold", fontsize=12)
    ax.set_xlabel("Truncation Cutoff (K₀)", fontweight="bold", fontsize=12)
    ax.set_title("Error Normalization: Shadow vs Truncation Baseline",
                fontweight="bold", fontsize=12, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k:g}" for k in K0_vals])
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="best", frameon=True, fontsize=11)

    # Add 100% reference line
    ax.axhline(y=100, color="red", linestyle="--", linewidth=1.5, alpha=0.5, label="Baseline (100%)")

    out = figdir / "k1_normalized_error_bars.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")


def plot_k2_cost_vs_accuracy_with_pareto(rows, figdir):
    """K2: Pareto frontier with explicit cost-accuracy curves."""
    setup_kdd_style()

    t_eval = pc.max_t(rows)
    rows_t = pc.filter_close(rows, "t", t_eval)

    nx_vals = sorted([int(v) for v in pc.unique_sorted(rows_t, "nx")])
    K0_vals = sorted([float(v) for v in pc.unique_sorted(rows_t, "K0")])

    fig, ax = plt.subplots(1, 1, figsize=(10.5, 6), constrained_layout=True)

    colors_nx = ["#1b9e77", "#d95f02", "#7570b3"]

    for i, nx in enumerate(nx_vals):
        q_shifts = []
        accuracies = []  # 1 - error (accuracy)

        for K0 in K0_vals:
            rs = [r for r in rows_t if r.get("nx") == nx and r.get("K0") == K0]
            err, _, _ = pc.mean_std([r.get("err_rho_vs_full") for r in rs])
            q_shift, _, _ = pc.mean_std([r.get("q_shift") for r in rs])

            q_shifts.append(q_shift)
            accuracies.append(1.0 - err)  # Accuracy proxy

        # Plot with fill
        ax.fill_between(q_shifts, 0, accuracies, alpha=0.2, color=colors_nx[i])
        ax.plot(q_shifts, accuracies, 'o-', color=colors_nx[i], linewidth=2.5,
               markersize=7, label=f"nx={nx}", alpha=0.85)

    ax.set_xlabel("Quantum Cost (q_shift qubits)", fontweight="bold", fontsize=12)
    ax.set_ylabel("Accuracy (1 - error)", fontweight="bold", fontsize=12)
    ax.set_yscale("log")
    ax.set_title("Pareto Frontier: Quantum Cost vs Classical Accuracy",
                fontweight="bold", fontsize=12, pad=15)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", frameon=True, fontsize=11)

    # Annotate optimal region
    ax.text(0.98, 0.05, "← More qubits\nHigher accuracy →", transform=ax.transAxes,
           fontsize=10, ha="right", va="bottom",
           bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.2))

    out = figdir / "k2_pareto_cost_vs_accuracy.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")


# ============================================================================
# Type L: Time-dependent analysis
# ============================================================================

def plot_l1_error_growth_rate(rows, figdir):
    """L1: Error growth rate (d(error)/dt) vs K0."""
    setup_kdd_style()

    nx_choice = sorted([int(v) for v in pc.unique_sorted(rows, "nx")])[0]
    rows_nx = [r for r in rows if r.get("nx") == nx_choice]

    K0_vals = sorted([float(v) for v in pc.unique_sorted(rows_nx, "K0")])
    t_vals = sorted([float(v) for v in pc.unique_sorted(rows_nx, "t")])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)

    growth_rates = {K0: [] for K0 in K0_vals}

    for K0 in K0_vals:
        errors = []
        for t in t_vals:
            rs = [r for r in rows_nx if r.get("K0") == K0 and abs(float(r.get("t", 0)) - t) < 1e-10]
            mu, _, _ = pc.mean_std([r.get("err_rho_vs_full") for r in rs])
            errors.append(mu)

        # Compute growth rate (log derivative)
        if len(t_vals) > 1:
            log_errors = np.log10(np.array(errors))
            dt = t_vals[1] - t_vals[0]
            # Numerical gradient
            growth = np.gradient(log_errors, t_vals)
            growth_rates[K0] = growth

    # Left: Growth rate vs time
    colors = plt.cm.viridis(np.linspace(0, 1, len(K0_vals)))
    for i, K0 in enumerate(K0_vals):
        if len(growth_rates[K0]) > 0:
            axes[0].plot(t_vals, growth_rates[K0], 'o-', color=colors[i],
                        linewidth=2, markersize=5, label=f"K₀={K0:g}", alpha=0.8)

    axes[0].axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    axes[0].set_xlabel("Time t", fontweight="bold", fontsize=11)
    axes[0].set_ylabel("d(log₁₀ error)/dt", fontweight="bold", fontsize=11)
    axes[0].set_title("Error Growth Rate", fontweight="bold", fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", frameon=False, fontsize=9, ncol=2)

    # Right: Final growth rate vs K0
    final_growth_rates = []
    for K0 in K0_vals:
        if len(growth_rates[K0]) > 0:
            final_growth_rates.append(growth_rates[K0][-1])
        else:
            final_growth_rates.append(0)

    axes[1].bar(range(len(K0_vals)), final_growth_rates, color=colors, alpha=0.8, edgecolor="black")
    axes[1].axhline(y=0, color="black", linestyle="-", linewidth=1)
    axes[1].set_xticks(range(len(K0_vals)))
    axes[1].set_xticklabels([f"{k:g}" for k in K0_vals], rotation=45)
    axes[1].set_ylabel("Growth Rate at t_max", fontweight="bold", fontsize=11)
    axes[1].set_xlabel("Truncation Cutoff (K₀)", fontweight="bold", fontsize=11)
    axes[1].set_title("Asymptotic Growth Rate", fontweight="bold", fontsize=11)
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Error Temporal Dynamics (nx={nx_choice})",
                fontsize=13, fontweight="bold")
    out = figdir / "l1_error_growth_rate.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")


def plot_l2_time_evolution_matrix(rows, figdir):
    """L2: Error evolution as a (K0 vs t) heatmap."""
    setup_kdd_style()

    nx_choice = sorted([int(v) for v in pc.unique_sorted(rows, "nx")])[0]
    rows_nx = [r for r in rows if r.get("nx") == nx_choice]

    K0_vals = sorted([float(v) for v in pc.unique_sorted(rows_nx, "K0")])
    t_vals = sorted([float(v) for v in pc.unique_sorted(rows_nx, "t")])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)

    for ax_idx, (ax, key, title) in enumerate([
        (axes[0], "err_rho_vs_full", "Density Error"),
        (axes[1], "err_momentum_vs_full", "Momentum Error"),
    ]):
        Z = np.zeros((len(K0_vals), len(t_vals)))

        for i, K0 in enumerate(K0_vals):
            for j, t in enumerate(t_vals):
                rs = [r for r in rows_nx if r.get("K0") == K0 and abs(float(r.get("t", 0)) - t) < 1e-10]
                mu, _, _ = pc.mean_std([r.get(key) for r in rs])
                Z[i, j] = np.log10(mu) if mu > 0 else -20

        im = ax.imshow(Z, cmap="RdYlGn_r", aspect="auto", origin="lower",
                      extent=[t_vals[0], t_vals[-1], K0_vals[0], K0_vals[-1]])

        ax.set_xlabel("Time t", fontweight="bold", fontsize=11)
        ax.set_ylabel("Truncation Cutoff (K₀)", fontweight="bold", fontsize=11)
        ax.set_title(f"{title} Evolution", fontweight="bold", fontsize=11)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("log₁₀(error)", fontweight="bold")

    fig.suptitle(f"Error Temporal Map: K₀ vs Time (nx={nx_choice})",
                fontsize=13, fontweight="bold")
    out = figdir / "l2_time_evolution_heatmap.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")


# ============================================================================
# Type M: Summary/Dashboard views
# ============================================================================

def plot_m1_kdd_summary_dashboard(rows, figdir):
    """M1: KDD-style summary dashboard with 4-6 key metrics."""
    setup_kdd_style(fontsize=9)

    t_eval = pc.max_t(rows)
    rows_t = pc.filter_close(rows, "t", t_eval)

    nx_vals = sorted([int(v) for v in pc.unique_sorted(rows_t, "nx")])
    K0_vals = sorted([float(v) for v in pc.unique_sorted(rows_t, "K0")])

    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)

    # Panel A: Error vs K0
    ax_a = fig.add_subplot(gs[0, :2])
    for i, nx in enumerate(nx_vals):
        ys = []
        for K0 in K0_vals:
            rs = [r for r in rows_t if r.get("nx") == nx and r.get("K0") == K0]
            mu, _, _ = pc.mean_std([r.get("err_rho_vs_full") for r in rs])
            ys.append(mu)
        color = ["#1b9e77", "#d95f02", "#7570b3"][i]
        ax_a.semilogy(K0_vals, ys, 'o-', color=color, linewidth=1.8, markersize=4, label=f"nx={nx}")
    ax_a.set_ylabel("ρ Error", fontweight="bold", fontsize=9)
    ax_a.set_xlabel("K₀", fontweight="bold", fontsize=9)
    ax_a.set_title("A) Truncation Error", fontweight="bold", fontsize=9)
    ax_a.grid(True, alpha=0.25, which="both")
    ax_a.legend(loc="best", fontsize=8)

    # Panel B: Momentum error
    ax_b = fig.add_subplot(gs[0, 2])
    nx_choice = nx_vals[0]
    for K0 in K0_vals:
        rs = [r for r in rows_t if r.get("nx") == nx_choice and r.get("K0") == K0]
        mu_j, _, _ = pc.mean_std([r.get("err_momentum_vs_full") for r in rs])
        mu_rho, _, _ = pc.mean_std([r.get("err_rho_vs_full") for r in rs])
        ax_b.scatter(mu_rho, mu_j, s=80, alpha=0.7, edgecolor="black", linewidth=0.6)
    ax_b.set_xlabel("ρ Error", fontweight="bold", fontsize=9)
    ax_b.set_ylabel("J Error", fontweight="bold", fontsize=9)
    ax_b.set_title("B) Error Correlation", fontweight="bold", fontsize=9)
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.grid(True, alpha=0.25)

    # Panel C: Qubit cost
    ax_c = fig.add_subplot(gs[1, 0])
    q_base_vals = []
    q_shift_vals = []
    for K0 in K0_vals:
        rs = [r for r in rows_t if r.get("nx") == nx_choice and r.get("K0") == K0]
        q_base, _, _ = pc.mean_std([r.get("q_base") for r in rs])
        q_shift, _, _ = pc.mean_std([r.get("q_shift") for r in rs])
        q_base_vals.append(q_base)
        q_shift_vals.append(q_shift)
    x = np.arange(len(K0_vals))
    ax_c.bar(x - 0.2, q_base_vals, 0.4, label="Full", alpha=0.7, color="#0072B2")
    ax_c.bar(x + 0.2, q_shift_vals, 0.4, label="Shadow", alpha=0.7, color="#D55E00")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([f"{k:g}" for k in K0_vals], rotation=45, fontsize=8)
    ax_c.set_ylabel("Qubits", fontweight="bold", fontsize=9)
    ax_c.set_title("C) Resource Cost", fontweight="bold", fontsize=9)
    ax_c.legend(fontsize=8)
    ax_c.grid(True, alpha=0.25, axis="y")

    # Panel D: Runtime
    ax_d = fig.add_subplot(gs[1, 1])
    rt_full_vals = []
    rt_shadow_vals = []
    for K0 in K0_vals:
        rs = [r for r in rows_t if r.get("nx") == nx_choice and r.get("K0") == K0]
        rt_full, _, _ = pc.mean_std([r.get("rt_baseline_full_s") for r in rs])
        rt_shadow, _, _ = pc.mean_std([r.get("rt_shadow_s") for r in rs])
        rt_full_vals.append(rt_full)
        rt_shadow_vals.append(rt_shadow)
    ax_d.semilogy(K0_vals, rt_full_vals, 's-', color="#0072B2", linewidth=1.8, markersize=4, label="Full")
    ax_d.semilogy(K0_vals, rt_shadow_vals, 'o-', color="#D55E00", linewidth=1.8, markersize=4, label="Shadow")
    ax_d.set_xlabel("K₀", fontweight="bold", fontsize=9)
    ax_d.set_ylabel("Time (s)", fontweight="bold", fontsize=9)
    ax_d.set_title("D) Wall-clock Runtime", fontweight="bold", fontsize=9)
    ax_d.grid(True, alpha=0.25, which="both")
    ax_d.legend(fontsize=8)

    # Panel E: Speedup
    ax_e = fig.add_subplot(gs[1, 2])
    speedups = []
    for rt_f, rt_s in zip(rt_full_vals, rt_shadow_vals):
        if rt_s > 0:
            speedups.append(rt_f / rt_s)
    ax_e.plot(K0_vals, speedups, 'D-', color="#009E73", linewidth=2, markersize=5)
    ax_e.axhline(y=1, color="gray", linestyle=":", linewidth=1.5, alpha=0.5)
    ax_e.set_ylabel("Speedup", fontweight="bold", fontsize=9)
    ax_e.set_xlabel("K₀", fontweight="bold", fontsize=9)
    ax_e.set_title("E) Acceleration", fontweight="bold", fontsize=9)
    ax_e.grid(True, alpha=0.25)

    # Panel F: Shadow vs Low-pass
    ax_f = fig.add_subplot(gs[2, :])
    for i, nx in enumerate(nx_vals):
        shadow_ys = []
        lp_ys = []
        for K0 in K0_vals:
            rs = [r for r in rows_t if r.get("nx") == nx and r.get("K0") == K0]
            shadow_mu, _, _ = pc.mean_std([r.get("err_rho_vs_full") for r in rs])
            lp_mu, _, _ = pc.mean_std([r.get("err_rho_lp_vs_full") for r in rs])
            shadow_ys.append(shadow_mu)
            lp_ys.append(lp_mu)
        color = ["#1b9e77", "#d95f02", "#7570b3"][i]
        ax_f.semilogy(K0_vals, shadow_ys, 'o-', color=color, linewidth=1.5, markersize=4, alpha=0.7)
        ax_f.semilogy(K0_vals, lp_ys, 's--', color=color, linewidth=1.2, markersize=3, alpha=0.5)
    ax_f.set_xlabel("K₀", fontweight="bold", fontsize=9)
    ax_f.set_ylabel("ρ Error", fontweight="bold", fontsize=9)
    ax_f.set_title("F) Shadow Optimality (solid=shadow, dashed=low-pass baseline)",
                  fontweight="bold", fontsize=9)
    ax_f.grid(True, alpha=0.25, which="both")

    fig.suptitle("SHIFT-FLOW: KDD Summary Dashboard", fontweight="bold", fontsize=12, y=0.995)
    out = figdir / "m1_kdd_summary_dashboard.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Generate advanced figure variants.")
    ap.add_argument("--in", dest="inp", default="results/sweep.csv")
    ap.add_argument("--figdir", default="figs_advanced_gallery")
    args = ap.parse_args()

    rows = pc.load_sweep_csv(args.inp)
    if not rows:
        raise SystemExit(f"No rows found in {args.inp}")

    figdir = pc.ensure_figdir(args.figdir)

    print("\n" + "="*70)
    print("Advanced Figure Gallery: Advanced, Efficient, and Creative Variants")
    print("="*70 + "\n")

    # Type I: Dual-axis
    print("Generating Type I (Dual-axis and Composites)...")
    plot_i1_error_vs_cost_dual_axis(rows, figdir)
    plot_i2_runtime_vs_error_bubble(rows, figdir)

    # Type J: Contour/projections
    print("Generating Type J (Contour & 3D Projections)...")
    plot_j1_contour_error_landscape(rows, figdir)
    plot_j2_3d_projection_scatter(rows, figdir)

    # Type K: Normalized
    print("Generating Type K (Efficiency & Normalized)...")
    plot_k1_normalized_error_comparison(rows, figdir)
    plot_k2_cost_vs_accuracy_with_pareto(rows, figdir)

    # Type L: Time-dependent
    print("Generating Type L (Temporal Analysis)...")
    plot_l1_error_growth_rate(rows, figdir)
    plot_l2_time_evolution_matrix(rows, figdir)

    # Type M: Dashboards
    print("Generating Type M (Summary Dashboards)...")
    plot_m1_kdd_summary_dashboard(rows, figdir)

    print("\n" + "="*70)
    print(f"✓ Generated 11 advanced figures in: {figdir}/")
    print("="*70 + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
