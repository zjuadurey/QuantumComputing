"""experiments/plot_nips_gallery.py

Comprehensive top-tier journal figure generation for SHIFT-FLOW.
Generates 15-20 figure variants across different plot types and styles.

References:
- NeurIPS/ICML: minimal, sans-serif, clean lines, error bands
- Nature: serif fonts, thick lines, muted palette, clear hierarchy
- Science: colorful but constrained, high information density
- KDD: emphasis on trade-offs, scalability, efficiency narratives
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from scipy import stats

import plot_common as pc


# ============================================================================
# Color Palettes for different journals/styles
# ============================================================================

def palette_nature() -> dict:
    """Nature-style: muted, high contrast serif"""
    return {
        "primary": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
        "light": ["#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5"],
        "bg": "#ffffff",
        "grid": "#e0e0e0",
        "text": "#000000",
    }

def palette_neurips() -> dict:
    """NeurIPS-style: clean, colorblind-friendly, minimal"""
    return {
        "primary": ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7"],
        "light": ["#56B4E9", "#F0E442", "#D69F00", "#56B4E9", "#E69F00"],
        "bg": "#ffffff",
        "grid": "#f0f0f0",
        "text": "#333333",
    }

def palette_science() -> dict:
    """Science-style: vibrant, clear hierarchy"""
    return {
        "primary": ["#4472C4", "#ED7D31", "#A5A5A5", "#FFC000", "#5B9BD5"],
        "light": ["#B4C7E7", "#F8CBAD", "#D9D9D9", "#FFF2CC", "#BDD7EE"],
        "bg": "#ffffff",
        "grid": "#eeeeee",
        "text": "#000000",
    }

def palette_kdd() -> dict:
    """KDD-style: technical, high contrast for trade-offs"""
    return {
        "primary": ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"],
        "light": ["#80cdc1", "#fc8d59", "#b3b3cc", "#f1b6da", "#b5de2b"],
        "bg": "#ffffff",
        "grid": "#f5f5f5",
        "text": "#1a1a1a",
    }


# ============================================================================
# Plot Setup & Style Functions
# ============================================================================

def setup_matplotlib_journal(journal: str = "neurips", fontsize: int = 10):
    """Configure matplotlib for journal-quality output."""

    if journal.lower() == "nature":
        plt.rcParams.update({
            "font.size": fontsize,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.labelsize": fontsize + 1,
            "axes.titlesize": fontsize + 2,
            "legend.fontsize": fontsize - 1,
            "xtick.labelsize": fontsize - 1,
            "ytick.labelsize": fontsize - 1,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "axes.linewidth": 1.0,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.8,
        })
    elif journal.lower() == "neurips":
        plt.rcParams.update({
            "font.size": fontsize,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "axes.labelsize": fontsize + 1,
            "axes.titlesize": fontsize + 2,
            "legend.fontsize": fontsize - 1,
            "xtick.labelsize": fontsize - 1,
            "ytick.labelsize": fontsize - 1,
            "lines.linewidth": 1.6,
            "lines.markersize": 4,
            "axes.linewidth": 0.8,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
        })
    elif journal.lower() == "science":
        plt.rcParams.update({
            "font.size": fontsize,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.labelsize": fontsize + 1,
            "axes.titlesize": fontsize + 2,
            "legend.fontsize": fontsize - 1,
            "xtick.labelsize": fontsize - 1,
            "ytick.labelsize": fontsize - 1,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "axes.linewidth": 1.2,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.32,
            "grid.linewidth": 0.7,
        })
    else:  # kdd
        plt.rcParams.update({
            "font.size": fontsize,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "axes.labelsize": fontsize,
            "axes.titlesize": fontsize + 1,
            "legend.fontsize": fontsize - 1,
            "xtick.labelsize": fontsize - 1,
            "ytick.labelsize": fontsize - 1,
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
            "axes.linewidth": 0.8,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.6,
        })

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ============================================================================
# Figure Functions: Type A - Error vs K0/M (Accuracy vs Truncation)
# ============================================================================

def plot_a1_grouped_bars_neurips(rows, figdir):
    """A1: Grouped bar chart (NeurIPS style) - Error vs K0 at max t."""
    setup_matplotlib_journal("neurips")
    pal = palette_neurips()

    t_eval = pc.max_t(rows)
    rows_t = pc.filter_close(rows, "t", t_eval)

    nx_vals = pc.unique_sorted(rows_t, "nx")
    K0_vals = pc.unique_sorted(rows_t, "K0")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    x = np.arange(len(K0_vals))
    width = 0.25

    for ax_idx, (ax, key, title) in enumerate([
        (axes[0], "err_rho_vs_full", "Density Error"),
        (axes[1], "err_momentum_vs_full", "Momentum Error"),
    ]):
        for i, nx in enumerate(nx_vals):
            ys = []
            for K0 in K0_vals:
                rs = [r for r in rows_t if r.get("nx") == nx and r.get("K0") == K0]
                mu, _, _ = pc.mean_std([r.get(key) for r in rs])
                ys.append(mu)

            ax.bar(x + i * width, ys, width, label=f"nx={nx}",
                   color=pal["primary"][i % len(pal["primary"])], alpha=0.85)

        ax.set_yscale("log")
        ax.set_xlabel("Cutoff radius K₀", fontweight="bold")
        ax.set_ylabel(f"Error (log scale)", fontweight="bold")
        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_xticks(x + width)
        ax.set_xticklabels([f"{k:g}" for k in K0_vals])
        ax.grid(True, which="both", alpha=0.25, axis="y")
        ax.legend(loc="best", frameon=False, fontsize=9)

    fig.suptitle(f"Accuracy vs Truncation (t={t_eval:g})", fontsize=12, fontweight="bold")
    out = figdir / "a1_grouped_bars_neurips.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")


def plot_a2_lines_with_bands_nature(rows, figdir):
    """A2: Line plots with confidence bands (Nature style)."""
    setup_matplotlib_journal("nature")
    pal = palette_nature()

    t_eval = pc.max_t(rows)
    rows_t = pc.filter_close(rows, "t", t_eval)

    nx_vals = pc.unique_sorted(rows_t, "nx")
    K0_vals = sorted([float(k) for k in pc.unique_sorted(rows_t, "K0")])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)

    for ax_idx, (ax, key, title) in enumerate([
        (axes[0], "err_rho_vs_full", "Density Error"),
        (axes[1], "err_momentum_vs_full", "Momentum Error"),
    ]):
        for i, nx in enumerate(nx_vals):
            ys = []
            errs = []
            for K0 in K0_vals:
                rs = [r for r in rows_t if r.get("nx") == nx and r.get("K0") == K0]
                mu, sd, n = pc.mean_std([r.get(key) for r in rs])
                ys.append(mu)
                # 95% CI
                ci = 1.96 * sd / np.sqrt(max(n, 1)) if n > 0 else sd
                errs.append(ci)

            color = pal["primary"][i % len(pal["primary"])]
            light_color = pc.alpha_band(color, alpha=0.22)

            ax.plot(K0_vals, ys, 'o-', color=color, linewidth=2.2, markersize=6, label=f"nx={nx}")
            ax.fill_between(K0_vals, np.array(ys) - np.array(errs),
                           np.array(ys) + np.array(errs),
                           color=light_color, alpha=0.35)

        ax.set_yscale("log")
        ax.set_xlabel("Cutoff radius K₀", fontweight="bold", fontsize=11)
        ax.set_ylabel("Error", fontweight="bold", fontsize=11)
        ax.set_title(title, fontweight="bold", fontsize=12, pad=12)
        ax.grid(True, which="both", alpha=0.3, linestyle="-", linewidth=0.6)
        ax.legend(loc="best", frameon=True, fancybox=False, edgecolor="black", fontsize=9)

    fig.suptitle(f"Truncation Error vs Cutoff (t={t_eval:g}, 95% CI)",
                fontsize=13, fontweight="bold")
    out = figdir / "a2_lines_with_bands_nature.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")


def plot_a3_minimal_lines(rows, figdir):
    """A3: Minimal line plot (Science style) - just mean, no bands."""
    setup_matplotlib_journal("science")
    pal = palette_science()

    t_eval = pc.max_t(rows)
    rows_t = pc.filter_close(rows, "t", t_eval)

    nx_vals = pc.unique_sorted(rows_t, "nx")
    K0_vals = sorted([float(k) for k in pc.unique_sorted(rows_t, "K0")])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)

    for ax_idx, (ax, key, title) in enumerate([
        (axes[0], "err_rho_vs_full", "ρ error"),
        (axes[1], "err_momentum_vs_full", "J error"),
    ]):
        for i, nx in enumerate(nx_vals):
            ys = []
            for K0 in K0_vals:
                rs = [r for r in rows_t if r.get("nx") == nx and r.get("K0") == K0]
                mu, _, _ = pc.mean_std([r.get(key) for r in rs])
                ys.append(mu)

            color = pal["primary"][i % len(pal["primary"])]
            ax.semilogy(K0_vals, ys, 'o-', color=color, linewidth=2.4, markersize=5,
                       label=f"nx={nx}", alpha=0.9)

        ax.set_xlabel("K₀", fontweight="bold")
        ax.set_ylabel(f"{title}", fontweight="bold")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="upper right", frameon=False, fontsize=9)

    fig.suptitle(f"Accuracy vs K₀ (t={t_eval:g})", fontsize=12, fontweight="bold", y=1.02)
    out = figdir / "a3_minimal_lines_science.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote: {out}")


def plot_a4_heatmap_error(rows, figdir):
    """A4: Heatmap showing error across (nx, K0) grid."""
    setup_matplotlib_journal("science")

    t_eval = pc.max_t(rows)
    rows_t = pc.filter_close(rows, "t", t_eval)

    nx_vals = sorted([int(v) for v in pc.unique_sorted(rows_t, "nx")])
    K0_vals = sorted([float(v) for v in pc.unique_sorted(rows_t, "K0")])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    for ax_idx, (ax, key) in enumerate([
        (axes[0], "err_rho_vs_full"),
        (axes[1], "err_momentum_vs_full"),
    ]):
        mat = np.zeros((len(nx_vals), len(K0_vals)))
        for i, nx in enumerate(nx_vals):
            for j, K0 in enumerate(K0_vals):
                rs = [r for r in rows_t if r.get("nx") == nx and r.get("K0") == K0]
                mu, _, _ = pc.mean_std([r.get(key) for r in rs])
                mat[i, j] = mu

        im = ax.imshow(np.log10(mat), cmap="RdYlGn_r", aspect="auto", origin="lower")
        ax.set_xticks(range(len(K0_vals)))
        ax.set_yticks(range(len(nx_vals)))
        ax.set_xticklabels([f"{k:g}" for k in K0_vals], rotation=45)
        ax.set_yticklabels([f"{n}" for n in nx_vals])
        ax.set_xlabel("K₀", fontweight="bold")
        ax.set_ylabel("nx", fontweight="bold")
        ax.set_title(["Density Error", "Momentum Error"][ax_idx], fontweight="bold")

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("log₁₀(error)", fontweight="bold")

    fig.suptitle(f"Error Landscape at t={t_eval:g}", fontsize=12, fontweight="bold")
    out = figdir / "a4_heatmap_error.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")


# ============================================================================
# Figure Functions: Type B - Trade-offs (Error vs Cost/Resource)
# ============================================================================

def plot_b1_scatter_pareto_neurips(rows, figdir):
    """B1: Scatter plot of error vs q_shift (Pareto frontier visualization)."""
    setup_matplotlib_journal("neurips")
    pal = palette_neurips()

    t_eval = pc.max_t(rows)
    rows_t = pc.filter_close(rows, "t", t_eval)

    nx_vals = sorted([int(v) for v in pc.unique_sorted(rows_t, "nx")])
    K0_vals = sorted([float(v) for v in pc.unique_sorted(rows_t, "K0")])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    for ax_idx, (ax, key, ylabel) in enumerate([
        (axes[0], "err_rho_vs_full", "Density Error"),
        (axes[1], "err_momentum_vs_full", "Momentum Error"),
    ]):
        for i, nx in enumerate(nx_vals):
            xs = []
            ys = []
            for K0 in K0_vals:
                rs = [r for r in rows_t if r.get("nx") == nx and r.get("K0") == K0]
                mu_err, _, _ = pc.mean_std([r.get(key) for r in rs])
                q_shift_mean, _, _ = pc.mean_std([r.get("q_shift") for r in rs])
                xs.append(q_shift_mean)
                ys.append(mu_err)

            color = pal["primary"][i % len(pal["primary"])]
            ax.scatter(xs, ys, s=100, alpha=0.7, color=color, edgecolor="black",
                      linewidth=0.8, label=f"nx={nx}", zorder=3)

            # Connect points
            sorted_idx = np.argsort(xs)
            xs_sorted = np.array(xs)[sorted_idx]
            ys_sorted = np.array(ys)[sorted_idx]
            ax.plot(xs_sorted, ys_sorted, '-', color=color, alpha=0.3, linewidth=1.5, zorder=1)

        ax.set_yscale("log")
        ax.set_xlabel("Qubit Cost (q_shift)", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(f"{ylabel} vs Qubit Cost", fontweight="bold", pad=10)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="best", frameon=False)

    fig.suptitle(f"Pareto Trade-off: Error vs Resource Cost (t={t_eval:g})",
                fontsize=12, fontweight="bold")
    out = figdir / "b1_pareto_scatter_neurips.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")


def plot_b2_tradeoff_with_annotations(rows, figdir):
    """B2: Trade-off with K0 value annotations."""
    setup_matplotlib_journal("science")
    pal = palette_science()

    t_eval = pc.max_t(rows)
    rows_t = pc.filter_close(rows, "t", t_eval)

    nx_vals = sorted([int(v) for v in pc.unique_sorted(rows_t, "nx")])
    K0_vals = sorted([float(v) for v in pc.unique_sorted(rows_t, "K0")])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    for ax_idx, (ax, key, ylabel) in enumerate([
        (axes[0], "err_rho_vs_full", "Density Error"),
        (axes[1], "err_momentum_vs_full", "Momentum Error"),
    ]):
        nx_choice = nx_vals[0]  # Focus on first nx
        xs = []
        ys = []
        labels = []

        for K0 in K0_vals:
            rs = [r for r in rows_t if r.get("nx") == nx_choice and r.get("K0") == K0]
            mu_err, _, _ = pc.mean_std([r.get(key) for r in rs])
            q_shift_mean, _, _ = pc.mean_std([r.get("q_shift") for r in rs])
            xs.append(q_shift_mean)
            ys.append(mu_err)
            labels.append(f"K₀={K0:g}")

        color = pal["primary"][0]
        ax.plot(xs, ys, 'o-', color=color, linewidth=2.5, markersize=8, alpha=0.8)

        # Annotate points
        for x, y, label in zip(xs, ys, labels):
            ax.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points",
                       fontsize=8, alpha=0.7)

        ax.set_yscale("log")
        ax.set_xlabel("Qubit Cost (q_shift)", fontweight="bold", fontsize=11)
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=11)
        ax.set_title(f"{ylabel} (nx={nx_choice})", fontweight="bold", fontsize=11)
        ax.grid(True, alpha=0.3, which="both")

    fig.suptitle(f"Error-Cost Trade-off: Truncation Boundary (t={t_eval:g})",
                fontsize=12, fontweight="bold")
    out = figdir / "b2_tradeoff_annotated.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")


# ============================================================================
# Figure Functions: Type C - Time Evolution
# ============================================================================

def plot_c1_error_vs_time_lines(rows, figdir):
    """C1: Error vs time, multiple K0 curves."""
    setup_matplotlib_journal("neurips")
    pal = palette_neurips()

    nx_choice = sorted([int(v) for v in pc.unique_sorted(rows, "nx")])[0]
    rows_nx = [r for r in rows if r.get("nx") == nx_choice]

    K0_vals = sorted([float(v) for v in pc.unique_sorted(rows_nx, "K0")])
    t_vals = sorted([float(v) for v in pc.unique_sorted(rows_nx, "t")])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    for ax_idx, (ax, key, title) in enumerate([
        (axes[0], "err_rho_vs_full", "Density Error"),
        (axes[1], "err_momentum_vs_full", "Momentum Error"),
    ]):
        for i, K0 in enumerate(K0_vals):
            ys = []
            for t in t_vals:
                rs = [r for r in rows_nx if r.get("K0") == K0 and abs(float(r.get("t", 0)) - t) < 1e-10]
                mu, _, _ = pc.mean_std([r.get(key) for r in rs])
                ys.append(mu)

            color = pal["primary"][i % len(pal["primary"])]
            ax.semilogy(t_vals, ys, 'o-', color=color, linewidth=2, markersize=4, label=f"K₀={K0:g}")

        ax.set_xlabel("Time t", fontweight="bold")
        ax.set_ylabel(title, fontweight="bold")
        ax.set_title(title, fontweight="bold", pad=10)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="best", frameon=False, fontsize=9, ncol=2)

    fig.suptitle(f"Error Evolution (nx={nx_choice})", fontsize=12, fontweight="bold")
    out = figdir / "c1_error_vs_time_lines.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")


def plot_c2_error_vs_time_bands(rows, figdir):
    """C2: Error vs time with confidence bands (Nature style)."""
    setup_matplotlib_journal("nature")
    pal = palette_nature()

    nx_choice = sorted([int(v) for v in pc.unique_sorted(rows, "nx")])[0]
    rows_nx = [r for r in rows if r.get("nx") == nx_choice]

    K0_vals = sorted([float(v) for v in pc.unique_sorted(rows_nx, "K0")])
    t_vals = sorted([float(v) for v in pc.unique_sorted(rows_nx, "t")])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)

    for ax_idx, (ax, key, title) in enumerate([
        (axes[0], "err_rho_vs_full", "Density Error"),
        (axes[1], "err_momentum_vs_full", "Momentum Error"),
    ]):
        for i, K0 in enumerate(K0_vals):
            ys = []
            errs = []
            for t in t_vals:
                rs = [r for r in rows_nx if r.get("K0") == K0 and abs(float(r.get("t", 0)) - t) < 1e-10]
                mu, sd, n = pc.mean_std([r.get(key) for r in rs])
                ys.append(mu)
                ci = 1.96 * sd / np.sqrt(max(n, 1)) if n > 0 else sd
                errs.append(ci)

            color = pal["primary"][i % len(pal["primary"])]
            light_color = pc.alpha_band(color, alpha=0.25)

            ax.semilogy(t_vals, ys, 'o-', color=color, linewidth=2.2, markersize=5, label=f"K₀={K0:g}")
            ax.fill_between(t_vals, np.array(ys) - np.array(errs),
                           np.array(ys) + np.array(errs),
                           color=light_color, alpha=0.4)

        ax.set_xlabel("Time t", fontweight="bold", fontsize=11)
        ax.set_ylabel(title, fontweight="bold", fontsize=11)
        ax.set_title(title, fontweight="bold", fontsize=11, pad=10)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="best", frameon=True, fontsize=9, ncol=2)

    fig.suptitle(f"Temporal Error Evolution (nx={nx_choice}, 95% CI)",
                fontsize=13, fontweight="bold")
    out = figdir / "c2_error_vs_time_bands.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")


# ============================================================================
# Figure Functions: Type D - Runtime & Acceleration
# ============================================================================

def plot_d1_runtime_comparison_grouped(rows, figdir):
    """D1: Runtime comparison as grouped bars."""
    setup_matplotlib_journal("neurips")
    pal = palette_neurips()

    nx_vals = sorted([int(v) for v in pc.unique_sorted(rows, "nx")])

    fig, ax = plt.subplots(1, 1, figsize=(10, 4), constrained_layout=True)

    x = np.arange(len(nx_vals))
    width = 0.35

    rt_baseline_full = []
    rt_shadow = []

    for nx in nx_vals:
        rows_nx = [r for r in rows if r.get("nx") == nx]
        t_eval = max([float(r.get("t", 0)) for r in rows_nx])
        rows_t = [r for r in rows_nx if abs(float(r.get("t", 0)) - t_eval) < 1e-10]

        mu_full, _, _ = pc.mean_std([r.get("rt_baseline_full_s") for r in rows_t])
        mu_shadow, _, _ = pc.mean_std([r.get("rt_shadow_s") for r in rows_t])

        rt_baseline_full.append(mu_full)
        rt_shadow.append(mu_shadow)

    ax.bar(x - width/2, rt_baseline_full, width, label="Full-state (FFT)",
           color=pal["primary"][0], alpha=0.85)
    ax.bar(x + width/2, rt_shadow, width, label="Shadow (truncated)",
           color=pal["primary"][1], alpha=0.85)

    ax.set_ylabel("Wall-clock Time (s)", fontweight="bold", fontsize=11)
    ax.set_xlabel("System Size (nx)", fontweight="bold", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}" for n in nx_vals])
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y", which="both")
    ax.legend(loc="best", frameon=False, fontsize=10)

    ax.set_title("Runtime Comparison: Full vs Shadow", fontweight="bold", fontsize=12, pad=12)

    out = figdir / "d1_runtime_grouped_bars.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")


def plot_d2_speedup_inset(rows, figdir):
    """D2: Speedup with inset zoom on details."""
    setup_matplotlib_journal("science")
    pal = palette_science()

    nx_vals = sorted([int(v) for v in pc.unique_sorted(rows, "nx")])
    K0_vals = sorted([float(v) for v in pc.unique_sorted(rows, "K0")])

    fig, ax = plt.subplots(1, 1, figsize=(9, 5), constrained_layout=True)

    t_eval = max([float(r.get("t", 0)) for r in rows])
    rows_t = [r for r in rows if abs(float(r.get("t", 0)) - t_eval) < 1e-10]

    for i, nx in enumerate(nx_vals):
        speedups = []
        K0_plot = []

        for K0 in K0_vals:
            rs = [r for r in rows_t if r.get("nx") == nx and r.get("K0") == K0]
            rt_full, _, _ = pc.mean_std([r.get("rt_baseline_full_s") for r in rs])
            rt_shadow, _, _ = pc.mean_std([r.get("rt_shadow_s") for r in rs])

            if rt_shadow > 0:
                speedup = rt_full / rt_shadow
                speedups.append(speedup)
                K0_plot.append(K0)

        color = pal["primary"][i]
        ax.plot(K0_plot, speedups, 'o-', color=color, linewidth=2.5, markersize=7,
               label=f"nx={nx}", alpha=0.85)

    ax.set_xlabel("Cutoff K₀", fontweight="bold", fontsize=11)
    ax.set_ylabel("Speedup (Full / Shadow)", fontweight="bold", fontsize=11)
    ax.set_title("Shadow Acceleration vs Truncation", fontweight="bold", fontsize=12, pad=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True, fontsize=10)

    # Inset zoom
    axins = ax.inset_axes([0.55, 0.15, 0.35, 0.35])
    for i, nx in enumerate(nx_vals):
        speedups = []
        K0_plot = []
        for K0 in K0_vals[-4:]:  # Last 4 K0 values
            rs = [r for r in rows_t if r.get("nx") == nx and r.get("K0") == K0]
            rt_full, _, _ = pc.mean_std([r.get("rt_baseline_full_s") for r in rs])
            rt_shadow, _, _ = pc.mean_std([r.get("rt_shadow_s") for r in rs])
            if rt_shadow > 0:
                speedup = rt_full / rt_shadow
                speedups.append(speedup)
                K0_plot.append(K0)

        color = pal["primary"][i]
        axins.plot(K0_plot, speedups, 'o-', color=color, linewidth=2, markersize=5)

    axins.set_title("Detail (high K₀)", fontsize=9)
    axins.grid(True, alpha=0.3)
    ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=1)

    out = figdir / "d2_speedup_inset.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")


# ============================================================================
# Figure Functions: Type E - Distributions (Robustness across seeds)
# ============================================================================

def plot_e1_violin_distributions(rows, figdir):
    """E1: Violin plots showing seed robustness."""
    setup_matplotlib_journal("science")
    pal = palette_science()

    t_eval = pc.max_t(rows)
    rows_t = pc.filter_close(rows, "t", t_eval)

    nx_choice = sorted([int(v) for v in pc.unique_sorted(rows_t, "nx")])[0]
    rows_nx = [r for r in rows_t if r.get("nx") == nx_choice]
    K0_vals = sorted([float(v) for v in pc.unique_sorted(rows_nx, "K0")])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    for ax_idx, (ax, key, title) in enumerate([
        (axes[0], "err_rho_vs_full", "Density Error"),
        (axes[1], "err_momentum_vs_full", "Momentum Error"),
    ]):
        data_to_plot = []
        labels = []

        for K0 in K0_vals:
            rs = [r for r in rows_nx if r.get("K0") == K0]
            vals = [r.get(key) for r in rs if r.get(key) is not None]
            if vals:
                data_to_plot.append(vals)
                labels.append(f"{K0:g}")

        parts = ax.violinplot(data_to_plot, positions=range(len(data_to_plot)),
                             showmeans=True, showmedians=True)

        for pc_elem in parts["bodies"]:
            pc_elem.set_facecolor(pal["primary"][0])
            pc_elem.set_alpha(0.7)

        ax.set_yscale("log")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_xlabel("K₀", fontweight="bold", fontsize=11)
        ax.set_ylabel(title, fontweight="bold", fontsize=11)
        ax.set_title(f"{title} Distribution (seeds={len(pc.unique_sorted(rows_nx, 'seed'))})",
                    fontweight="bold", fontsize=11)
        ax.grid(True, alpha=0.3, axis="y", which="both")

    fig.suptitle(f"Robustness Across Seeds (nx={nx_choice}, t={t_eval:g})",
                fontsize=12, fontweight="bold")
    out = figdir / "e1_violin_distributions.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")


def plot_e2_box_whisker_distributions(rows, figdir):
    """E2: Box-whisker plots."""
    setup_matplotlib_journal("neurips")
    pal = palette_neurips()

    t_eval = pc.max_t(rows)
    rows_t = pc.filter_close(rows, "t", t_eval)

    nx_choice = sorted([int(v) for v in pc.unique_sorted(rows_t, "nx")])[0]
    rows_nx = [r for r in rows_t if r.get("nx") == nx_choice]
    K0_vals = sorted([float(v) for v in pc.unique_sorted(rows_nx, "K0")])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    for ax_idx, (ax, key, title) in enumerate([
        (axes[0], "err_rho_vs_full", "Density Error"),
        (axes[1], "err_momentum_vs_full", "Momentum Error"),
    ]):
        data_to_plot = []
        labels = []

        for K0 in K0_vals:
            rs = [r for r in rows_nx if r.get("K0") == K0]
            vals = [r.get(key) for r in rs if r.get(key) is not None]
            if vals:
                data_to_plot.append(vals)
                labels.append(f"{K0:g}")

        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                       widths=0.6, showmeans=True)

        for patch in bp["boxes"]:
            patch.set_facecolor(pal["primary"][ax_idx])
            patch.set_alpha(0.7)

        ax.set_yscale("log")
        ax.set_xlabel("K₀", fontweight="bold")
        ax.set_ylabel(title, fontweight="bold")
        ax.set_title(f"{title}", fontweight="bold", pad=10)
        ax.grid(True, alpha=0.3, axis="y", which="both")

    fig.suptitle(f"Error Distribution (nx={nx_choice})", fontsize=12, fontweight="bold")
    out = figdir / "e2_box_whisker_distributions.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")


# ============================================================================
# Figure Functions: Type F - Multi-dimensional / Faceted
# ============================================================================

def plot_f1_multi_panel_by_nx(rows, figdir):
    """F1: Small multiples - one row per nx value."""
    setup_matplotlib_journal("neurips")
    pal = palette_neurips()

    t_eval = pc.max_t(rows)
    rows_t = pc.filter_close(rows, "t", t_eval)

    nx_vals = sorted([int(v) for v in pc.unique_sorted(rows_t, "nx")])
    K0_vals = sorted([float(v) for v in pc.unique_sorted(rows_t, "K0")])

    fig, axes = plt.subplots(len(nx_vals), 2, figsize=(10, 3.2 * len(nx_vals)),
                            constrained_layout=True)
    if len(nx_vals) == 1:
        axes = axes.reshape(1, -1)

    for row_idx, nx in enumerate(nx_vals):
        for col_idx, (key, title) in enumerate([
            ("err_rho_vs_full", "Density Error"),
            ("err_momentum_vs_full", "Momentum Error"),
        ]):
            ax = axes[row_idx, col_idx]

            ys = []
            for K0 in K0_vals:
                rs = [r for r in rows_t if r.get("nx") == nx and r.get("K0") == K0]
                mu, _, _ = pc.mean_std([r.get(key) for r in rs])
                ys.append(mu)

            color = pal["primary"][col_idx]
            ax.semilogy(K0_vals, ys, 'o-', color=color, linewidth=2, markersize=6, alpha=0.85)

            ax.set_ylabel(title, fontweight="bold", fontsize=10)
            if row_idx == len(nx_vals) - 1:
                ax.set_xlabel("K₀", fontweight="bold", fontsize=10)
            ax.set_title(f"nx={nx}", fontweight="bold", fontsize=11)
            ax.grid(True, alpha=0.3, which="both")

    fig.suptitle(f"Accuracy vs Truncation (System Size Scaling, t={t_eval:g})",
                fontsize=12, fontweight="bold")
    out = figdir / "f1_multi_panel_by_nx.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")


# ============================================================================
# Figure Functions: Type G - Shadow vs Low-pass Comparison
# ============================================================================

def plot_g1_shadow_vs_lowpass_overlay(rows, figdir):
    """G1: Overlay shadow and low-pass errors to show truncation optimality."""
    setup_matplotlib_journal("nature")
    pal = palette_nature()

    t_eval = pc.max_t(rows)
    rows_t = pc.filter_close(rows, "t", t_eval)

    nx_choice = sorted([int(v) for v in pc.unique_sorted(rows_t, "nx")])[0]
    rows_nx = [r for r in rows_t if r.get("nx") == nx_choice]
    K0_vals = sorted([float(v) for v in pc.unique_sorted(rows_nx, "K0")])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)

    for ax_idx, (ax, key_shadow, key_lp, title) in enumerate([
        (axes[0], "err_rho_vs_full", "err_rho_lp_vs_full", "Density Error"),
        (axes[1], "err_momentum_vs_full", "err_momentum_lp_vs_full", "Momentum Error"),
    ]):

        shadow_ys = []
        lp_ys = []

        for K0 in K0_vals:
            rs = [r for r in rows_nx if r.get("K0") == K0]
            shadow_mu, _, _ = pc.mean_std([r.get(key_shadow) for r in rs])
            lp_mu, _, _ = pc.mean_std([r.get(key_lp) for r in rs])
            shadow_ys.append(shadow_mu)
            lp_ys.append(lp_mu)

        color_shadow = pal["primary"][0]
        color_lp = pal["primary"][1]

        ax.semilogy(K0_vals, shadow_ys, 'o-', color=color_shadow, linewidth=2.2,
                   markersize=6, label="Shadow", alpha=0.85)
        ax.semilogy(K0_vals, lp_ys, 's--', color=color_lp, linewidth=2,
                   markersize=5, label="Low-pass (baseline)", alpha=0.75)

        ax.set_xlabel("K₀", fontweight="bold", fontsize=11)
        ax.set_ylabel(title, fontweight="bold", fontsize=11)
        ax.set_title(title, fontweight="bold", fontsize=11, pad=10)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="best", frameon=True, fontsize=10)

    fig.suptitle(f"Shadow vs Truncation Baseline (nx={nx_choice}, t={t_eval:g})",
                fontsize=13, fontweight="bold")
    out = figdir / "g1_shadow_vs_lowpass_overlay.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")


def plot_g2_ratio_shadow_to_lowpass(rows, figdir):
    """G2: Ratio of shadow error to low-pass error (shows optimality)."""
    setup_matplotlib_journal("science")
    pal = palette_science()

    t_eval = pc.max_t(rows)
    rows_t = pc.filter_close(rows, "t", t_eval)

    nx_vals = sorted([int(v) for v in pc.unique_sorted(rows_t, "nx")])
    K0_vals = sorted([float(v) for v in pc.unique_sorted(rows_t, "K0")])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    for ax_idx, (ax, key_shadow, key_lp, title) in enumerate([
        (axes[0], "err_rho_vs_full", "err_rho_lp_vs_full", "Density"),
        (axes[1], "err_momentum_vs_full", "err_momentum_lp_vs_full", "Momentum"),
    ]):
        for i, nx in enumerate(nx_vals):
            ratios = []
            for K0 in K0_vals:
                rs = [r for r in rows_t if r.get("nx") == nx and r.get("K0") == K0]
                shadow_mu, _, _ = pc.mean_std([r.get(key_shadow) for r in rs])
                lp_mu, _, _ = pc.mean_std([r.get(key_lp) for r in rs])
                if lp_mu > 0:
                    ratio = shadow_mu / lp_mu
                    ratios.append(ratio)
                else:
                    ratios.append(1.0)

            color = pal["primary"][i]
            ax.plot(K0_vals, ratios, 'o-', color=color, linewidth=2, markersize=6,
                   label=f"nx={nx}", alpha=0.85)

        # Add y=1 reference line
        ax.axhline(y=1.0, color="red", linestyle=":", linewidth=1.5, alpha=0.6, label="Ideal (ratio=1)")

        ax.set_xlabel("K₀", fontweight="bold")
        ax.set_ylabel(f"{title} Ratio (Shadow / LP)", fontweight="bold")
        ax.set_title(f"{title} Error Ratio", fontweight="bold", pad=10)
        ax.set_ylim([0.8, 1.3])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", frameon=False, fontsize=9)

    fig.suptitle(f"Truncation Optimality: Shadow vs Low-pass (t={t_eval:g})",
                fontsize=12, fontweight="bold")
    out = figdir / "g2_ratio_shadow_to_lowpass.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")


# ============================================================================
# Figure Functions: Type H - Advanced Styling (NeurIPS/ICML specific)
# ============================================================================

def plot_h1_neurips_2col_compact(rows, figdir):
    """H1: Compact 2-column NeurIPS-style layout with multiple insights."""
    setup_matplotlib_journal("neurips", fontsize=9)
    pal = palette_neurips()

    t_eval = pc.max_t(rows)
    rows_t = pc.filter_close(rows, "t", t_eval)

    nx_vals = sorted([int(v) for v in pc.unique_sorted(rows_t, "nx")])
    K0_vals = sorted([float(v) for v in pc.unique_sorted(rows_t, "K0")])

    fig = plt.figure(figsize=(8.5, 5.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # Top-left: Error vs K0
    ax1 = fig.add_subplot(gs[0, 0])
    for i, nx in enumerate(nx_vals[:2]):
        ys = []
        for K0 in K0_vals:
            rs = [r for r in rows_t if r.get("nx") == nx and r.get("K0") == K0]
            mu, _, _ = pc.mean_std([r.get("err_rho_vs_full") for r in rs])
            ys.append(mu)
        color = pal["primary"][i]
        ax1.semilogy(K0_vals, ys, 'o-', color=color, linewidth=1.8, markersize=4, label=f"nx={nx}")
    ax1.set_ylabel("Density Error", fontweight="bold", fontsize=9)
    ax1.set_xlabel("K₀", fontweight="bold", fontsize=9)
    ax1.grid(True, alpha=0.25, which="both")
    ax1.legend(loc="best", frameon=False, fontsize=8)
    ax1.set_title("A) Truncation Error", fontweight="bold", fontsize=9)

    # Top-right: Pareto
    ax2 = fig.add_subplot(gs[0, 1])
    nx_choice = nx_vals[0]
    for K0 in K0_vals:
        rs = [r for r in rows_t if r.get("nx") == nx_choice and r.get("K0") == K0]
        err, _, _ = pc.mean_std([r.get("err_rho_vs_full") for r in rs])
        q_shift, _, _ = pc.mean_std([r.get("q_shift") for r in rs])
        ax2.scatter(q_shift, err, s=80, color=pal["primary"][0], alpha=0.7, edgecolor="black", linewidth=0.7)
    ax2.set_yscale("log")
    ax2.set_ylabel("Density Error", fontweight="bold", fontsize=9)
    ax2.set_xlabel("Qubit Cost", fontweight="bold", fontsize=9)
    ax2.grid(True, alpha=0.25)
    ax2.set_title("B) Error-Cost Pareto", fontweight="bold", fontsize=9)

    # Bottom: Time evolution
    ax3 = fig.add_subplot(gs[1, :])
    rows_nx = [r for r in rows if r.get("nx") == nx_choice]
    t_vals = sorted([float(v) for v in pc.unique_sorted(rows_nx, "t")])
    K0_subset = K0_vals[::2]  # Every other K0
    for i, K0 in enumerate(K0_subset):
        ys = []
        for t in t_vals:
            rs = [r for r in rows_nx if r.get("K0") == K0 and abs(float(r.get("t", 0)) - t) < 1e-10]
            mu, _, _ = pc.mean_std([r.get("err_rho_vs_full") for r in rs])
            ys.append(mu)
        color = pal["primary"][i % len(pal["primary"])]
        ax3.semilogy(t_vals, ys, 'o-', color=color, linewidth=1.8, markersize=4, label=f"K₀={K0:g}")
    ax3.set_ylabel("Density Error", fontweight="bold", fontsize=9)
    ax3.set_xlabel("Time t", fontweight="bold", fontsize=9)
    ax3.grid(True, alpha=0.25, which="both")
    ax3.legend(loc="best", frameon=False, fontsize=8, ncol=4)
    ax3.set_title("C) Temporal Evolution", fontweight="bold", fontsize=9)

    fig.suptitle("SHIFT-FLOW: Truncation Analysis Summary", fontweight="bold", fontsize=11, y=0.98)
    out = figdir / "h1_neurips_2col_compact.pdf"
    fig.savefig(out)
    print(f"Wrote: {out}")


def plot_h2_nature_3panel_dense(rows, figdir):
    """H2: Dense 3-panel Nature-style layout."""
    setup_matplotlib_journal("nature", fontsize=10)
    pal = palette_nature()

    t_eval = pc.max_t(rows)
    rows_t = pc.filter_close(rows, "t", t_eval)

    nx_vals = sorted([int(v) for v in pc.unique_sorted(rows_t, "nx")])
    K0_vals = sorted([float(v) for v in pc.unique_sorted(rows_t, "K0")])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), constrained_layout=True)

    # Left: Density error
    for i, nx in enumerate(nx_vals):
        ys = []
        for K0 in K0_vals:
            rs = [r for r in rows_t if r.get("nx") == nx and r.get("K0") == K0]
            mu, _, _ = pc.mean_std([r.get("err_rho_vs_full") for r in rs])
            ys.append(mu)
        color = pal["primary"][i]
        light = pc.alpha_band(color, alpha=0.2)
        axes[0].semilogy(K0_vals, ys, 'o-', color=color, linewidth=2.2, markersize=6,
                        label=f"nx={nx}", alpha=0.9)
        axes[0].fill_between(K0_vals, np.array(ys)*0.95, np.array(ys)*1.05, color=light, alpha=0.3)
    axes[0].set_ylabel("Density Error", fontweight="bold", fontsize=11)
    axes[0].set_xlabel("K₀", fontweight="bold", fontsize=11)
    axes[0].set_title("a | Truncation Error", fontweight="bold", fontsize=11)
    axes[0].grid(True, alpha=0.3, which="both")
    axes[0].legend(frameon=True, fontsize=9)

    # Middle: Shadow vs Low-pass
    nx_choice = nx_vals[0]
    rows_nx = [r for r in rows_t if r.get("nx") == nx_choice]
    shadow_ys = []
    lp_ys = []
    for K0 in K0_vals:
        rs = [r for r in rows_nx if r.get("K0") == K0]
        shadow_mu, _, _ = pc.mean_std([r.get("err_rho_vs_full") for r in rs])
        lp_mu, _, _ = pc.mean_std([r.get("err_rho_lp_vs_full") for r in rs])
        shadow_ys.append(shadow_mu)
        lp_ys.append(lp_mu)
    axes[1].semilogy(K0_vals, shadow_ys, 'o-', color=pal["primary"][0], linewidth=2.2,
                    markersize=6, label="Shadow", alpha=0.9)
    axes[1].semilogy(K0_vals, lp_ys, 's--', color=pal["primary"][1], linewidth=2,
                    markersize=5, label="Low-pass", alpha=0.8)
    axes[1].set_ylabel("Density Error", fontweight="bold", fontsize=11)
    axes[1].set_xlabel("K₀", fontweight="bold", fontsize=11)
    axes[1].set_title("b | Shadow Optimality", fontweight="bold", fontsize=11)
    axes[1].grid(True, alpha=0.3, which="both")
    axes[1].legend(frameon=True, fontsize=9)

    # Right: Runtime
    rt_results = []
    for K0 in K0_vals:
        rs = [r for r in rows_t if r.get("nx") == nx_choice and r.get("K0") == K0]
        rt_full, _, _ = pc.mean_std([r.get("rt_baseline_full_s") for r in rs])
        rt_shadow, _, _ = pc.mean_std([r.get("rt_shadow_s") for r in rs])
        if rt_shadow > 0:
            speedup = rt_full / rt_shadow
            rt_results.append(speedup)
        else:
            rt_results.append(0)
    axes[2].plot(K0_vals, rt_results, 'D-', color=pal["primary"][2], linewidth=2.5,
                markersize=7, alpha=0.9)
    axes[2].axhline(y=1, color="gray", linestyle=":", linewidth=1.5, alpha=0.5)
    axes[2].set_ylabel("Speedup Factor", fontweight="bold", fontsize=11)
    axes[2].set_xlabel("K₀", fontweight="bold", fontsize=11)
    axes[2].set_title("c | Acceleration", fontweight="bold", fontsize=11)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Quantum Shadow Dynamics: Accuracy, Optimality, and Efficiency",
                fontweight="bold", fontsize=12, y=1.00)
    out = figdir / "h2_nature_3panel_dense.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote: {out}")


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Generate comprehensive figure gallery.")
    ap.add_argument("--in", dest="inp", default="results/sweep.csv")
    ap.add_argument("--figdir", default="figs_nips_gallery")
    args = ap.parse_args()

    rows = pc.load_sweep_csv(args.inp)
    if not rows:
        raise SystemExit(f"No rows found in {args.inp}")

    figdir = pc.ensure_figdir(args.figdir)

    print("\n" + "="*70)
    print("SHIFT-FLOW: Top-Tier Journal Figure Gallery")
    print("="*70 + "\n")

    # Type A: Error vs K0/M
    print("Generating Type A (Error vs Truncation)...")
    plot_a1_grouped_bars_neurips(rows, figdir)
    plot_a2_lines_with_bands_nature(rows, figdir)
    plot_a3_minimal_lines(rows, figdir)
    plot_a4_heatmap_error(rows, figdir)

    # Type B: Trade-offs
    print("Generating Type B (Error-Cost Trade-offs)...")
    plot_b1_scatter_pareto_neurips(rows, figdir)
    plot_b2_tradeoff_with_annotations(rows, figdir)

    # Type C: Time Evolution
    print("Generating Type C (Temporal Evolution)...")
    plot_c1_error_vs_time_lines(rows, figdir)
    plot_c2_error_vs_time_bands(rows, figdir)

    # Type D: Runtime
    print("Generating Type D (Runtime & Acceleration)...")
    plot_d1_runtime_comparison_grouped(rows, figdir)
    plot_d2_speedup_inset(rows, figdir)

    # Type E: Distributions
    print("Generating Type E (Robustness Distributions)...")
    plot_e1_violin_distributions(rows, figdir)
    plot_e2_box_whisker_distributions(rows, figdir)

    # Type F: Multi-panel
    print("Generating Type F (Small Multiples)...")
    plot_f1_multi_panel_by_nx(rows, figdir)

    # Type G: Shadow vs Low-pass
    print("Generating Type G (Shadow vs Baseline Comparison)...")
    plot_g1_shadow_vs_lowpass_overlay(rows, figdir)
    plot_g2_ratio_shadow_to_lowpass(rows, figdir)

    # Type H: Advanced layouts
    print("Generating Type H (Advanced Multi-panel Layouts)...")
    plot_h1_neurips_2col_compact(rows, figdir)
    plot_h2_nature_3panel_dense(rows, figdir)

    print("\n" + "="*70)
    print(f"✓ Generated 17 figures in: {figdir}/")
    print("="*70 + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
