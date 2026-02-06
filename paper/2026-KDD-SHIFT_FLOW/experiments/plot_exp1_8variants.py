"""experiments/plot_exp1_8variants.py

Exp1: show how shadow accuracy improves as more low-frequency modes are kept.

This script generates 8 alternative figure designs for Exp1, aligned with the
"standard academic chart library" in experiments/plot.md.

Output files go to --figdir (default: figs_exp1_8/) with names exp1_var*.pdf.
"""

from __future__ import annotations

import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

import plot_common as pc


EPS = 1e-18


def _pos(x: float) -> float:
    return max(float(x), EPS)


def _style_metric(metric: str):
    if metric == "rho":
        return r"$\varepsilon_\rho$", "err_rho_vs_full", "err_rho_lp_vs_full"
    if metric == "mom":
        return r"$\varepsilon_{\mathbf{J}}$", "err_momentum_vs_full", "err_momentum_lp_vs_full"
    raise ValueError(metric)


def _aggregate(rows_t: list[dict], nx_vals: list[int], K0_vals: list[float]):
    """Return dicts of mean/std for required keys at each (nx, K0)."""
    keys = [
        "err_rho_vs_full",
        "err_momentum_vs_full",
        "err_rho_lp_vs_full",
        "err_momentum_lp_vs_full",
        "rt_total_s",
    ]
    stats: dict[tuple[int, float], dict[str, tuple[float, float, int]]] = {}
    for nx in nx_vals:
        for K0 in K0_vals:
            rs = [r for r in rows_t if int(r.get("nx")) == int(nx) and abs(float(r.get("K0")) - float(K0)) <= 1e-12]
            d: dict[str, tuple[float, float, int]] = {}
            for k in keys:
                mu, sd, n = pc.mean_std([r.get(k) for r in rs])
                d[k] = (mu, sd, n)
            stats[(int(nx), float(K0))] = d
    return stats


def _K0_to_M(rows_t: list[dict]) -> dict[float, int]:
    out: dict[float, int] = {}
    for r in rows_t:
        k0 = r.get("K0")
        m = r.get("M")
        if k0 is None or m is None:
            continue
        out[float(k0)] = int(m)
    return out


def _var1_lines_bands_vs_M(figdir, t_eval, nx_vals, Ms, K0s_sorted, stats):
    """#6: line plots with confidence bands vs M."""
    colors = pc.color_cycle(len(nx_vals))
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9), constrained_layout=True)
    for ax, metric in zip(axes, ["rho", "mom"]):
        ylab, key_sh, _key_lp = _style_metric(metric)
        for color, nx in zip(colors, nx_vals):
            mu = [stats[(nx, K0)][key_sh][0] for K0 in K0s_sorted]
            sd = [stats[(nx, K0)][key_sh][1] for K0 in K0s_sorted]
            y = np.asarray(mu, dtype=float)
            s = np.asarray(sd, dtype=float)
            lo = np.maximum(y - s, EPS)
            hi = np.maximum(y + s, EPS)
            ax.plot(Ms, y, marker="o", color=color, label=f"nx={nx}")
            ax.fill_between(Ms, lo, hi, color=pc.alpha_band(color, 0.18), linewidth=0)
        ax.set_yscale("log")
        ax.set_xlabel("M (# retained k-modes)")
        ax.set_title(f"{ylab} (shadow vs full)")
        ax.grid(True, which="both", alpha=0.28)
    axes[0].legend(loc="best", frameon=True)
    fig.suptitle(f"Exp1 (var1): Accuracy improves with M (t={t_eval:g})")
    out = figdir / "exp1_var1_lines_bands_vs_M.pdf"
    fig.savefig(out)
    return out


def _var2_lines_bands_vs_K0(figdir, t_eval, nx_vals, K0_vals, K0_to_M, stats):
    """#6: line plots with confidence bands vs K0."""
    colors = pc.color_cycle(len(nx_vals))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9), constrained_layout=True)
    for ax, metric in zip(axes, ["rho", "mom"]):
        ylab, key_sh, _key_lp = _style_metric(metric)
        for color, nx in zip(colors, nx_vals):
            mu = [stats[(nx, K0)][key_sh][0] for K0 in K0_vals]
            sd = [stats[(nx, K0)][key_sh][1] for K0 in K0_vals]
            y = np.asarray(mu, dtype=float)
            s = np.asarray(sd, dtype=float)
            lo = np.maximum(y - s, EPS)
            hi = np.maximum(y + s, EPS)
            ax.plot(K0_vals, y, marker="o", color=color, label=f"nx={nx}")
            ax.fill_between(K0_vals, lo, hi, color=pc.alpha_band(color, 0.18), linewidth=0)
        ax.set_yscale("log")
        ax.set_xlabel("K0 (cutoff radius)")
        ax.set_title(f"{ylab} (shadow vs full)")
        ax.grid(True, which="both", alpha=0.28)

        # ticks with M annotation
        labels = [f"{k0:g}\n(M={K0_to_M.get(float(k0), 0)})" for k0 in K0_vals]
        ax.set_xticks(K0_vals)
        ax.set_xticklabels(labels)
    axes[0].legend(loc="best", frameon=True)
    fig.suptitle(f"Exp1 (var2): Accuracy vs K0 with uncertainty (t={t_eval:g})")
    out = figdir / "exp1_var2_lines_bands_vs_K0.pdf"
    fig.savefig(out)
    return out


def _var3_facets_nx_two_metrics(figdir, t_eval, nx_vals, Ms, K0s_sorted, stats):
    """#19: facet grid by nx, show rho+mom together per panel."""
    c_rho, c_mom = pc.color_cycle(2)
    fig, axes = plt.subplots(1, len(nx_vals), figsize=(12.0, 3.6), constrained_layout=True, sharey=True)
    if len(nx_vals) == 1:
        axes = [axes]

    for ax, nx in zip(axes, nx_vals):
        for metric, color, ls in [("rho", c_rho, "-"), ("mom", c_mom, "--")]:
            ylab, key_sh, _key_lp = _style_metric(metric)
            mu = [stats[(nx, K0)][key_sh][0] for K0 in K0s_sorted]
            sd = [stats[(nx, K0)][key_sh][1] for K0 in K0s_sorted]
            y = np.asarray(mu, dtype=float)
            s = np.asarray(sd, dtype=float)
            lo = np.maximum(y - s, EPS)
            hi = np.maximum(y + s, EPS)
            ax.plot(Ms, y, marker="o", color=color, ls=ls, label=ylab)
            ax.fill_between(Ms, lo, hi, color=pc.alpha_band(color, 0.16), linewidth=0)

        ax.set_title(f"nx={nx}")
        ax.set_xlabel("M")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.28)

    axes[0].set_ylabel("error (vs full)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=True)
    fig.suptitle(f"Exp1 (var3): Facets by nx (t={t_eval:g})")
    out = figdir / "exp1_var3_facets_by_nx_two_metrics.pdf"
    fig.savefig(out)
    return out


def _var4_grouped_bars_errorbars(figdir, t_eval, nx_vals, K0_vals, K0_to_M, stats):
    """#1: grouped bar chart with error bars."""
    colors = pc.color_cycle(len(nx_vals))
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.9), constrained_layout=True)
    x = np.arange(len(K0_vals), dtype=float)
    width = 0.75 / max(len(nx_vals), 1)

    for ax, metric in zip(axes, ["rho", "mom"]):
        ylab, key_sh, _key_lp = _style_metric(metric)
        for i, (color, nx) in enumerate(zip(colors, nx_vals)):
            mu = [stats[(nx, K0)][key_sh][0] for K0 in K0_vals]
            sd = [stats[(nx, K0)][key_sh][1] for K0 in K0_vals]
            xi = x - 0.375 + (i + 0.5) * width
            ax.bar(xi, mu, width=width, color=color, alpha=0.88, label=f"nx={nx}")
            ax.errorbar(xi, mu, yerr=sd, fmt="none", ecolor="k", elinewidth=0.8, capsize=2, alpha=0.7)
        ax.set_yscale("log")
        ax.set_title(f"{ylab} (shadow vs full)")
        ax.set_xlabel("K0")
        ax.grid(True, which="both", alpha=0.28)
        labels = [f"{k0:g}\n(M={K0_to_M.get(float(k0), 0)})" for k0 in K0_vals]
        ax.set_xticks(x)
        ax.set_xticklabels(labels)

    axes[0].legend(loc="best", frameon=True)
    fig.suptitle(f"Exp1 (var4): Grouped bars with std error bars (t={t_eval:g})")
    out = figdir / "exp1_var4_grouped_bars_errorbars.pdf"
    fig.savefig(out)
    return out


def _var5_scatter_fit_loglog(figdir, t_eval, nx_vals, Ms, K0s_sorted, stats):
    """#8: scatter with log-log fit."""
    colors = pc.color_cycle(len(nx_vals))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9), constrained_layout=True)

    for ax, metric in zip(axes, ["rho", "mom"]):
        ylab, key_sh, _key_lp = _style_metric(metric)
        for color, nx in zip(colors, nx_vals):
            y = np.asarray([_pos(stats[(nx, K0)][key_sh][0]) for K0 in K0s_sorted], dtype=float)
            x = np.asarray([float(m) for m in Ms], dtype=float)
            ax.scatter(x, y, color=color, alpha=0.9, label=f"nx={nx}")

            # fit in log-log
            lx = np.log10(x)
            ly = np.log10(y)
            a, b = np.polyfit(lx, ly, deg=1)
            yfit = 10 ** (a + b * lx)
            ax.plot(x, yfit, color=color, ls=":", alpha=0.9)
            ax.text(x[-1], yfit[-1], f"{b:.2f}", color=color, fontsize=8, ha="left", va="center")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("M (log)")
        ax.set_title(f"{ylab} (fit slope shown)" )
        ax.grid(True, which="both", alpha=0.28)

    axes[0].legend(loc="best", frameon=True)
    fig.suptitle(f"Exp1 (var5): Scatter + log-log fit (t={t_eval:g})")
    out = figdir / "exp1_var5_scatter_fit_loglog.pdf"
    fig.savefig(out)
    return out


def _var6_heatmap_nx_K0(figdir, t_eval, nx_vals, K0_vals, stats):
    """#11: heatmap overview."""
    def mat(key: str) -> np.ndarray:
        out = np.full((len(nx_vals), len(K0_vals)), np.nan, dtype=float)
        for i, nx in enumerate(nx_vals):
            for j, K0 in enumerate(K0_vals):
                out[i, j] = _pos(stats[(nx, K0)][key][0])
        return out

    rho = np.log10(mat("err_rho_vs_full"))
    mom = np.log10(mat("err_momentum_vs_full"))
    vmin = float(np.nanmin([np.nanmin(rho), np.nanmin(mom)]))
    vmax = float(np.nanmax([np.nanmax(rho), np.nanmax(mom)]))

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9), constrained_layout=True)
    im0 = axes[0].imshow(rho, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0].set_title(r"$\log_{10}\,\varepsilon_\rho$ (vs full)")
    axes[1].imshow(mom, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
    axes[1].set_title(r"$\log_{10}\,\varepsilon_{\mathbf{J}}$ (vs full)")

    for ax in axes:
        ax.set_xlabel("K0")
        ax.set_ylabel("nx")
        ax.set_xticks(np.arange(len(K0_vals)))
        ax.set_xticklabels([f"{k0:g}" for k0 in K0_vals])
        ax.set_yticks(np.arange(len(nx_vals)))
        ax.set_yticklabels([str(nx) for nx in nx_vals])

    cbar = fig.colorbar(im0, ax=axes, fraction=0.046, pad=0.02)
    cbar.set_label("log10 error")
    fig.suptitle(f"Exp1 (var6): Heatmap overview at t={t_eval:g}")
    out = figdir / "exp1_var6_heatmap_nx_K0.pdf"
    fig.savefig(out)
    return out


def _var7_shadow_vs_lp_overlay(figdir, t_eval, nx_vals, Ms, K0s_sorted, stats):
    """Composite: show shadow vs full and low-pass vs full together (solid vs dashed)."""
    colors = pc.color_cycle(len(nx_vals))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.9), constrained_layout=True)
    for ax, metric in zip(axes, ["rho", "mom"]):
        ylab, key_sh, key_lp = _style_metric(metric)
        for color, nx in zip(colors, nx_vals):
            y_sh = np.asarray([_pos(stats[(nx, K0)][key_sh][0]) for K0 in K0s_sorted], dtype=float)
            s_sh = np.asarray([stats[(nx, K0)][key_sh][1] for K0 in K0s_sorted], dtype=float)
            lo = np.maximum(y_sh - s_sh, EPS)
            hi = np.maximum(y_sh + s_sh, EPS)
            y_lp = np.asarray([_pos(stats[(nx, K0)][key_lp][0]) for K0 in K0s_sorted], dtype=float)
            ax.plot(Ms, y_sh, marker="o", color=color, ls="-", label=f"nx={nx}")
            ax.fill_between(Ms, lo, hi, color=pc.alpha_band(color, 0.14), linewidth=0)
            ax.plot(Ms, y_lp, color=color, ls="--", alpha=0.9)
        ax.set_yscale("log")
        ax.set_xlabel("M")
        ax.set_title(f"{ylab}: shadow(solid) vs low-pass(dashed)")
        ax.grid(True, which="both", alpha=0.28)

    # add line-style legend
    from matplotlib.lines import Line2D

    style_legend = [
        Line2D([0], [0], color="k", lw=2.0, ls="-", label="shadow vs full"),
        Line2D([0], [0], color="k", lw=2.0, ls="--", label="low-pass vs full"),
    ]
    axes[0].legend(loc="best", frameon=True)
    fig.legend(handles=style_legend, loc="upper right", frameon=True)
    fig.suptitle(f"Exp1 (var7): Shadow achieves truncation-optimal error (t={t_eval:g})")
    out = figdir / "exp1_var7_shadow_vs_lp_overlay.pdf"
    fig.savefig(out)
    return out


def _var8_ratio_shadow_to_lp(figdir, t_eval, rows_t, nx_vals, Ms, K0s_sorted):
    """Ratio plot: err_shadow_vs_full / err_lp_vs_full (should be ~1)."""

    def ratio_stats(nx: int, K0: float, key_sh: str, key_lp: str):
        rs = [r for r in rows_t if int(r.get("nx")) == int(nx) and abs(float(r.get("K0")) - float(K0)) <= 1e-12]
        ratios = []
        for r in rs:
            a = float(r.get(key_sh))
            b = float(r.get(key_lp))
            if b == 0.0:
                continue
            ratios.append(a / b)
        mu, sd, n = pc.mean_std(ratios)
        return mu, sd, n

    colors = pc.color_cycle(len(nx_vals))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), constrained_layout=True)
    all_vals = []

    for ax, metric in zip(axes, ["rho", "mom"]):
        ylab, key_sh, key_lp = _style_metric(metric)
        for color, nx in zip(colors, nx_vals):
            mu = []
            sd = []
            for K0 in K0s_sorted:
                m, s, _n = ratio_stats(nx, K0, key_sh, key_lp)
                mu.append(m)
                sd.append(s)
            y = np.asarray(mu, dtype=float)
            s = np.asarray(sd, dtype=float)
            ax.plot(Ms, y, marker="o", color=color, label=f"nx={nx}")
            ax.fill_between(Ms, y - s, y + s, color=pc.alpha_band(color, 0.14), linewidth=0)
            all_vals.extend(list(y - s))
            all_vals.extend(list(y + s))

        ax.axhline(1.0, color="k", lw=1.2, ls=":", alpha=0.8)
        ax.set_xlabel("M")
        ax.set_title(f"ratio {ylab}: shadow/full divided by low-pass/full")
        ax.grid(True, which="both", alpha=0.28)

    # y-limits around 1 if possible
    finite = [v for v in all_vals if math.isfinite(float(v))]
    if finite:
        lo = float(min(finite))
        hi = float(max(finite))
        pad = 0.08 * (hi - lo + 1e-12)
        for ax in axes:
            ax.set_ylim(lo - pad, hi + pad)

    axes[0].legend(loc="best", frameon=True)
    fig.suptitle(f"Exp1 (var8): Shadow adds ~no extra error beyond low-pass (t={t_eval:g})")
    out = figdir / "exp1_var8_ratio_shadow_to_lowpass.pdf"
    fig.savefig(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/sweep.csv")
    ap.add_argument("--figdir", default="figs_exp1_8")
    ap.add_argument("--t", type=float, default=None, help="evaluation time (default: max t in file)")
    args = ap.parse_args()

    rows = pc.load_sweep_csv(args.inp)
    if not rows:
        raise SystemExit(f"No rows found in {args.inp}")

    t_eval = pc.max_t(rows) if args.t is None else float(args.t)
    rows_t = pc.filter_close(rows, "t", t_eval)
    if not rows_t:
        raise SystemExit(f"No rows at t={t_eval} in {args.inp}")

    pc.apply_mpl_style()
    figdir = pc.ensure_figdir(args.figdir)

    nx_vals = sorted(int(x) for x in pc.unique_sorted(rows_t, "nx"))
    K0_vals = sorted(float(x) for x in pc.unique_sorted(rows_t, "K0"))
    K0_to_M = _K0_to_M(rows_t)

    # sort K0 by M (ascending)
    pairs = sorted((int(K0_to_M[float(k0)]), float(k0)) for k0 in K0_vals)
    Ms = [m for (m, _k0) in pairs]
    K0s_sorted = [k0 for (_m, k0) in pairs]

    stats = _aggregate(rows_t, nx_vals, K0_vals)

    outs = []
    outs.append(_var1_lines_bands_vs_M(figdir, t_eval, nx_vals, Ms, K0s_sorted, stats))
    outs.append(_var2_lines_bands_vs_K0(figdir, t_eval, nx_vals, K0_vals, K0_to_M, stats))
    outs.append(_var3_facets_nx_two_metrics(figdir, t_eval, nx_vals, Ms, K0s_sorted, stats))
    outs.append(_var4_grouped_bars_errorbars(figdir, t_eval, nx_vals, K0_vals, K0_to_M, stats))
    outs.append(_var5_scatter_fit_loglog(figdir, t_eval, nx_vals, Ms, K0s_sorted, stats))
    outs.append(_var6_heatmap_nx_K0(figdir, t_eval, nx_vals, K0_vals, stats))
    outs.append(_var7_shadow_vs_lp_overlay(figdir, t_eval, nx_vals, Ms, K0s_sorted, stats))
    outs.append(_var8_ratio_shadow_to_lp(figdir, t_eval, rows_t, nx_vals, Ms, K0s_sorted))

    for o in outs:
        print(f"Wrote: {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
