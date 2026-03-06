"""experiments/plot_kdd_gallery.py

Generate a gallery of KDD/NeurIPS/ICML/Nature-style figures from:
  - results/sweep.csv
  - results/full_aer_times.csv

The goal is to showcase SHIFT-FLOW shadow advantages:
- accuracy vs truncation (vs full)
- cost proxies (q_shift vs q_base)
- Aer evolution-time speedups (full vs shadow)

Outputs are written to a dedicated folder (default: figs_kdd_gallery/).

Run:
  python3 experiments/plot_kdd_gallery.py \
    --sweep results/sweep.csv \
    --full results/full_aer_times.csv \
    --outdir figs_kdd_gallery
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_common as pc


EPS = 1e-18


def apply_gallery_style() -> None:
    pc.apply_mpl_style()
    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.fancybox": False,
        }
    )


def nx_colors(nx_vals: list[int]) -> dict[int, str]:
    # restrained, Nature-like accents
    base = [
        "#1C2C3A",  # ink navy
        "#2E6F6D",  # muted teal
        "#6B4F3A",  # umber
        "#516B7A",  # steel
        "#3E6259",  # muted green
        "#4B5563",  # slate
    ]
    return {int(nx): base[i % len(base)] for i, nx in enumerate(sorted(nx_vals))}


def _t_key(t: float) -> str:
    return f"{float(t):.12g}"


def _finite(vals):
    out = []
    for x in vals:
        if x is None:
            continue
        try:
            v = float(x)
        except Exception:
            continue
        if math.isfinite(v):
            out.append(v)
    return out


def pct25_50_75(vals) -> tuple[float, float, float, int]:
    v = np.asarray(_finite(vals), dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    return (
        float(np.percentile(v, 25)),
        float(np.percentile(v, 50)),
        float(np.percentile(v, 75)),
        int(v.size),
    )


def mean_std(vals) -> tuple[float, float, int]:
    mu, sd, n = pc.mean_std(vals)
    return float(mu), float(sd), int(n)


def build_full_time_map(full_rows: list[dict], metric: str = "run") -> tuple[dict[tuple[int, int, str], float], dict[int, list[float]]]:
    if metric not in {"run", "total"}:
        raise ValueError(metric)
    key_full = "rt_full_aer_run_s"
    key_tr = "rt_full_aer_transpile_s"

    m: dict[tuple[int, int, str], float] = {}
    by_nx: dict[int, list[float]] = {}
    for r in full_rows:
        nx = r.get("nx")
        seed = r.get("seed")
        t = r.get("t")
        if nx is None or seed is None or t is None:
            continue
        v = r.get(key_full)
        if v is None:
            continue
        v = float(v)
        if metric == "total":
            v_tr = r.get(key_tr)
            if v_tr is not None and math.isfinite(float(v_tr)):
                v = v + float(v_tr)
        if not math.isfinite(v):
            continue
        k = (int(nx), int(seed), _t_key(float(t)))
        m[k] = v
        by_nx.setdefault(int(nx), []).append(v)
    return m, by_nx


def compute_speedups(
    sweep_rows: list[dict],
    full_time_map: dict[tuple[int, int, str], float],
    *,
    metric: str = "run",
) -> list[dict]:
    if metric not in {"run", "total"}:
        raise ValueError(metric)
    key_shadow = "rt_shadow_aer_run_s"
    key_tr = "rt_shadow_aer_transpile_s"

    out = []
    for r in sweep_rows:
        if r.get("shadow_backend") != "qiskit_aer":
            continue
        nx = r.get("nx")
        seed = r.get("seed")
        t = r.get("t")
        if nx is None or seed is None or t is None:
            continue
        v = r.get(key_shadow)
        if v is None:
            continue
        v = float(v)
        if metric == "total":
            v_tr = r.get(key_tr)
            if v_tr is not None and math.isfinite(float(v_tr)):
                v = v + float(v_tr)
        if not math.isfinite(v) or v <= 0.0:
            continue

        fk = (int(nx), int(seed), _t_key(float(t)))
        full_v = full_time_map.get(fk)
        if full_v is None or (not math.isfinite(float(full_v))) or float(full_v) <= 0.0:
            continue

        rr = dict(r)
        rr["shadow_time_aer"] = float(v)
        rr["full_time_aer"] = float(full_v)
        rr["speedup"] = float(full_v) / float(v)
        out.append(rr)
    return out


def select_K0_subset(rows: list[dict], nx_focus: int, n: int = 4) -> list[float]:
    rs = [r for r in rows if int(r.get("nx")) == int(nx_focus)]
    K0_vals = sorted({float(r.get("K0")) for r in rs if r.get("K0") is not None})
    if len(K0_vals) <= n:
        return K0_vals
    # pick roughly evenly spaced K0s
    idx = np.linspace(0, len(K0_vals) - 1, n).round().astype(int)
    return [K0_vals[i] for i in idx]


def K0_to_M_at_t(rows_t: list[dict]) -> dict[float, int]:
    m = {}
    for r in rows_t:
        K0 = r.get("K0")
        M = r.get("M")
        if K0 is None or M is None:
            continue
        m[float(K0)] = int(M)
    return m


def save(fig, outdir: Path, name: str) -> Path:
    out = outdir / name
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="results/sweep.csv")
    ap.add_argument("--full", default="results/full_aer_times.csv")
    ap.add_argument("--outdir", default="figs_kdd_gallery")
    ap.add_argument("--t-eval", type=float, default=None, help="default: max t in sweep")
    ap.add_argument("--nx-focus", type=int, default=6)
    ap.add_argument("--speedup-metric", choices=["run", "total"], default="run")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sweep_rows = pc.load_sweep_csv(args.sweep)
    full_rows = pc.load_sweep_csv(args.full)
    if not sweep_rows:
        raise SystemExit(f"No rows in {args.sweep}")
    if not full_rows:
        raise SystemExit(f"No rows in {args.full}")

    t_eval = pc.max_t(sweep_rows) if args.t_eval is None else float(args.t_eval)
    rows_t = pc.filter_close(sweep_rows, "t", t_eval)
    if not rows_t:
        raise SystemExit(f"No rows at t={t_eval} in {args.sweep}")

    # Required columns sanity
    needed = ["err_rho_vs_full", "err_momentum_vs_full", "err_rho_lp_vs_full", "err_momentum_lp_vs_full"]
    for k in needed:
        if k not in rows_t[0]:
            raise SystemExit(f"Missing column {k} in sweep")

    nx_vals = sorted({int(r.get("nx")) for r in rows_t if r.get("nx") is not None})
    K0_vals = sorted({float(r.get("K0")) for r in rows_t if r.get("K0") is not None})
    K0_to_M = K0_to_M_at_t(rows_t)
    K0_sorted_by_M = sorted(K0_vals, key=lambda k0: K0_to_M.get(float(k0), 0))
    Ms = [int(K0_to_M[float(k0)]) for k0 in K0_sorted_by_M]

    cmap_nx = nx_colors(nx_vals)
    apply_gallery_style()

    # ---------- aggregate stats at t_eval ----------
    stats = {}
    for nx in nx_vals:
        for K0 in K0_sorted_by_M:
            rs = [r for r in rows_t if int(r.get("nx")) == nx and abs(float(r.get("K0")) - float(K0)) <= 1e-12]
            stats[(nx, K0)] = {
                "rho": mean_std([r.get("err_rho_vs_full") for r in rs]),
                "mom": mean_std([r.get("err_momentum_vs_full") for r in rs]),
                "rho_lp": mean_std([r.get("err_rho_lp_vs_full") for r in rs]),
                "mom_lp": mean_std([r.get("err_momentum_lp_vs_full") for r in rs]),
            }

    # ---------- build speedup rows ----------
    full_map, full_by_nx = build_full_time_map(full_rows, metric=args.speedup_metric)
    speed_rows = compute_speedups(sweep_rows, full_map, metric=args.speedup_metric)
    speed_rows_t = pc.filter_close(speed_rows, "t", t_eval)

    # group speedups by (nx,K0)
    sp = {}
    for nx in nx_vals:
        for K0 in K0_sorted_by_M:
            rs = [r for r in speed_rows_t if int(r.get("nx")) == nx and abs(float(r.get("K0")) - float(K0)) <= 1e-12]
            sp[(nx, K0)] = {
                "speedup": pct25_50_75([r.get("speedup") for r in rs]),
                "shadow_time": pct25_50_75([r.get("shadow_time_aer") for r in rs]),
            }
    full_stat = {nx: pct25_50_75(full_by_nx.get(nx, [])) for nx in nx_vals}

    outs: list[Path] = []

    # ========================
    # (01) Grouped bars (errors vs M)
    # ========================
    x = np.arange(len(Ms), dtype=float)
    width = 0.75 / max(len(nx_vals), 1)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.9), constrained_layout=True)
    for ax, metric in zip(axes, ["rho", "mom"]):
        for i, nx in enumerate(nx_vals):
            color = cmap_nx[nx]
            mu = [stats[(nx, K0)][metric][0] for K0 in K0_sorted_by_M]
            sd = [stats[(nx, K0)][metric][1] for K0 in K0_sorted_by_M]
            xi = x - 0.375 + (i + 0.5) * width
            ax.bar(
                xi,
                mu,
                width=width,
                color=color,
                alpha=0.9,
                edgecolor=(0, 0, 0, 0.25),
                linewidth=0.6,
                label=f"nx={nx}",
                zorder=2,
            )
            ax.errorbar(xi, mu, yerr=sd, fmt="none", ecolor=(0, 0, 0, 0.55), elinewidth=0.8, capsize=2, zorder=3)
        ax.set_yscale("log")
        ax.set_xlabel("M (retained modes)")
        ax.set_title(r"$\varepsilon_\rho$" if metric == "rho" else r"$\varepsilon_{\mathbf{J}}$")
        ax.set_xticks(x)
        ax.set_xticklabels([str(m) for m in Ms])
        ax.grid(True, which="both", alpha=0.18)
    axes[0].legend(loc="best", frameon=True)
    fig.suptitle(f"Accuracy vs M (shadow vs full) at t={t_eval:g}")
    outs.append(save(fig, outdir, "kdd_01_errors_grouped_bars_vs_M.pdf"))

    # ========================
    # (02) Lines + bands (errors vs M)
    # ========================
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.8), constrained_layout=True)
    for ax, metric in zip(axes, ["rho", "mom"]):
        for nx in nx_vals:
            color = cmap_nx[nx]
            mu = np.asarray([stats[(nx, K0)][metric][0] for K0 in K0_sorted_by_M], dtype=float)
            sd = np.asarray([stats[(nx, K0)][metric][1] for K0 in K0_sorted_by_M], dtype=float)
            lo = np.maximum(mu - sd, EPS)
            hi = np.maximum(mu + sd, EPS)
            ax.plot(Ms, mu, marker="o", color=color, label=f"nx={nx}")
            ax.fill_between(Ms, lo, hi, color=pc.alpha_band(color, 0.16), linewidth=0)
        ax.set_yscale("log")
        ax.set_xlabel("M")
        ax.set_title(r"$\varepsilon_\rho$" if metric == "rho" else r"$\varepsilon_{\mathbf{J}}$")
        ax.grid(True, which="both", alpha=0.18)
    axes[0].legend(loc="best", frameon=True)
    fig.suptitle(f"Accuracy with uncertainty (t={t_eval:g})")
    outs.append(save(fig, outdir, "kdd_02_errors_lines_bands_vs_M.pdf"))

    # ========================
    # (03) Shadow vs LP overlay (truncation-optimality)
    # ========================
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.8), constrained_layout=True)
    for ax, metric, key_lp in [
        (axes[0], "rho", "rho_lp"),
        (axes[1], "mom", "mom_lp"),
    ]:
        for nx in nx_vals:
            color = cmap_nx[nx]
            mu_sh = np.asarray([stats[(nx, K0)][metric][0] for K0 in K0_sorted_by_M], dtype=float)
            mu_lp = np.asarray([stats[(nx, K0)][key_lp][0] for K0 in K0_sorted_by_M], dtype=float)
            ax.plot(Ms, mu_sh, marker="o", color=color, ls="-", label=f"shadow nx={nx}")
            ax.plot(Ms, mu_lp, color=color, ls="--", alpha=0.9, label=f"low-pass nx={nx}")
        ax.set_yscale("log")
        ax.set_xlabel("M")
        ax.set_title(r"$\varepsilon_\rho$" if metric == "rho" else r"$\varepsilon_{\mathbf{J}}$")
        ax.grid(True, which="both", alpha=0.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[: len(nx_vals)], [f"nx={nx}" for nx in nx_vals], loc="upper right", frameon=True)
    fig.suptitle(f"Shadow matches truncation baseline (solid=shadow, dashed=low-pass) at t={t_eval:g}")
    outs.append(save(fig, outdir, "kdd_03_shadow_vs_lowpass_overlay.pdf"))

    # ========================
    # (04) Ratio plot: shadow/full divided by low-pass/full
    # ========================
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6), constrained_layout=True)
    for ax, metric, key_lp in [
        (axes[0], "rho", "rho_lp"),
        (axes[1], "mom", "mom_lp"),
    ]:
        all_vals = []
        for nx in nx_vals:
            color = cmap_nx[nx]
            r_mu = []
            r_sd = []
            for K0 in K0_sorted_by_M:
                mu_sh, sd_sh, _n = stats[(nx, K0)][metric]
                mu_lp, sd_lp, _n = stats[(nx, K0)][key_lp]
                if mu_lp <= 0:
                    r_mu.append(float("nan"))
                    r_sd.append(float("nan"))
                else:
                    r_mu.append(mu_sh / mu_lp)
                    # conservative band from relative stds
                    rel = 0.0
                    if mu_sh > 0:
                        rel += (sd_sh / mu_sh) ** 2
                    if mu_lp > 0:
                        rel += (sd_lp / mu_lp) ** 2
                    r_sd.append(abs(mu_sh / mu_lp) * math.sqrt(rel))
            y = np.asarray(r_mu, dtype=float)
            s = np.asarray(r_sd, dtype=float)
            ax.plot(Ms, y, marker="o", color=color, label=f"nx={nx}")
            ax.fill_between(Ms, y - s, y + s, color=pc.alpha_band(color, 0.12), linewidth=0)
            all_vals.extend(list(y - s))
            all_vals.extend(list(y + s))

        ax.axhline(1.0, color="k", lw=1.1, ls=":", alpha=0.75)
        ax.set_xlabel("M")
        ax.set_title("ratio" if metric == "rho" else "ratio")
        ax.grid(True, alpha=0.18)
        finite = [v for v in all_vals if math.isfinite(float(v))]
        if finite:
            lo = float(min(finite))
            hi = float(max(finite))
            pad = 0.08 * (hi - lo + 1e-12)
            ax.set_ylim(lo - pad, hi + pad)
    axes[0].set_title(r"$\varepsilon_\rho^{sh}/\varepsilon_\rho^{lp}$")
    axes[1].set_title(r"$\varepsilon_{\mathbf{J}}^{sh}/\varepsilon_{\mathbf{J}}^{lp}$")
    axes[0].legend(loc="best", frameon=True)
    fig.suptitle(f"Shadow adds ~no extra error beyond truncation (t={t_eval:g})")
    outs.append(save(fig, outdir, "kdd_04_ratio_shadow_to_lowpass.pdf"))

    # ========================
    # (05) Error vs q_shift, bubble=size shadow time (Aer)
    # ========================
    # Aggregate per (nx,K0)
    pts = []
    for nx in nx_vals:
        for K0 in K0_sorted_by_M:
            # combined error
            er = stats[(nx, K0)]["rho"][0]
            em = stats[(nx, K0)]["mom"][0]
            y = max(float(er), float(em))
            qsh = None
            for r in rows_t:
                if int(r.get("nx")) == nx and abs(float(r.get("K0")) - float(K0)) <= 1e-12:
                    qsh = int(r.get("q_shift"))
                    break
            if qsh is None:
                continue
            sh_med = sp[(nx, K0)]["shadow_time"][1]
            if not math.isfinite(float(sh_med)):
                continue
            pts.append((nx, K0, int(qsh), int(K0_to_M[K0]), y, sh_med))

    fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.2), constrained_layout=True)
    times = np.asarray([p[5] for p in pts], dtype=float)
    tmin, tmax = float(np.min(times)), float(np.max(times))
    sizes = 50.0 + 300.0 * (times - tmin) / (tmax - tmin + 1e-12)
    for (nx, _K0, qsh, M, y, _t), s in zip(pts, sizes):
        ax.scatter(
            [qsh],
            [_finite([y])[0] if _finite([y]) else y],
            s=float(s),
            color=cmap_nx[int(nx)],
            alpha=0.78,
            edgecolor=(0, 0, 0, 0.25),
            linewidth=0.6,
        )
    for nx in nx_vals:
        ax.scatter([], [], s=80, color=cmap_nx[nx], label=f"nx={nx}")
    ax.set_yscale("log")
    ax.set_xlabel(r"$q_{shift}$")
    ax.set_ylabel(r"combined error  max($\varepsilon_\rho$, $\varepsilon_{\mathbf{J}}$)")
    ax.set_title(f"Trade-off: error vs q_shift (bubble=size Aer shadow time)\n(t={t_eval:g})")
    ax.grid(True, which="both", alpha=0.18)
    ax.legend(loc="best", frameon=True)
    outs.append(save(fig, outdir, "kdd_05_pareto_error_vs_qshift_bubble_time.pdf"))

    # ========================
    # (06) Error vs speedup (Aer)
    # ========================
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.2), constrained_layout=True)
    for nx in nx_vals:
        xs = []
        ys = []
        for K0 in K0_sorted_by_M:
            er = stats[(nx, K0)]["rho"][0]
            em = stats[(nx, K0)]["mom"][0]
            xval = max(float(er), float(em))
            yval = sp[(nx, K0)]["speedup"][1]
            if math.isfinite(xval) and math.isfinite(yval):
                xs.append(max(xval, EPS))
                ys.append(yval)
        ax.plot(xs, ys, marker="o", color=cmap_nx[nx], label=f"nx={nx}")
    ax.set_xscale("log")
    ax.set_xlabel(r"combined error  max($\varepsilon_\rho$, $\varepsilon_{\mathbf{J}}$)")
    ax.set_ylabel("speedup (full/shadow)")
    ax.set_title(f"Accuracy-speedup trade-off (Aer, {args.speedup_metric})\n(t={t_eval:g})")
    ax.grid(True, which="both", alpha=0.18)
    ax.legend(loc="best", frameon=True)
    outs.append(save(fig, outdir, "kdd_06_tradeoff_error_vs_speedup.pdf"))

    # ========================
    # (07) Speedup vs M (linear + inset)
    # ========================
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.9), constrained_layout=True)
    all_hi = []
    for nx in nx_vals:
        med = [sp[(nx, K0)]["speedup"][1] for K0 in K0_sorted_by_M]
        lo = [sp[(nx, K0)]["speedup"][0] for K0 in K0_sorted_by_M]
        hi = [sp[(nx, K0)]["speedup"][2] for K0 in K0_sorted_by_M]
        ax.plot(Ms, med, marker="o", color=cmap_nx[nx], label=f"nx={nx}")
        ax.fill_between(Ms, lo, hi, color=pc.alpha_band(cmap_nx[nx], 0.14), linewidth=0)
        all_hi.extend(hi)
    ax.axhline(1.0, color="k", lw=1.0, ls=":", alpha=0.7)
    ymax = max([v for v in all_hi if math.isfinite(float(v))] + [1.0])
    ax.set_ylim(0.0, ymax * 1.08)
    ax.set_xlabel("M")
    ax.set_ylabel("speedup (full/shadow)")
    ax.set_title(f"Aer speedup vs M ({args.speedup_metric}, t={t_eval:g})")
    ax.grid(True, alpha=0.18)
    ax.legend(loc="upper left", frameon=True)

    # inset zoom
    try:
        from mpl_toolkits.axes_grid1.inset_locator import mark_inset

        if ymax >= 10.0:
            axins = ax.inset_axes([0.32, 0.22, 0.50, 0.40])
            for nx in nx_vals:
                med = [sp[(nx, K0)]["speedup"][1] for K0 in K0_sorted_by_M]
                lo = [sp[(nx, K0)]["speedup"][0] for K0 in K0_sorted_by_M]
                hi = [sp[(nx, K0)]["speedup"][2] for K0 in K0_sorted_by_M]
                axins.plot(Ms, med, marker="o", ms=3, lw=1.4, color=cmap_nx[nx])
                axins.fill_between(Ms, lo, hi, color=pc.alpha_band(cmap_nx[nx], 0.10), linewidth=0)
            axins.axhline(1.0, color="k", lw=0.9, ls=":", alpha=0.7)
            axins.set_xlim(min(Ms), max(Ms))
            axins.set_ylim(0.8, 6.0)
            axins.grid(True, alpha=0.18)
            axins.set_xticks([])
            axins.set_yticks([1, 2, 4, 6])
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec=(0, 0, 0, 0.25), lw=0.8)
    except Exception:
        pass

    outs.append(save(fig, outdir, "kdd_07_speedup_vs_M_inset.pdf"))

    # ========================
    # (08) Aer runtime vs qubit proxy (q)
    # ========================
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.0), constrained_layout=True)
    # full points
    for nx in nx_vals:
        q = None
        for r in rows_t:
            if int(r.get("nx")) == nx:
                q = int(r.get("q_base"))
                break
        if q is None:
            continue
        lo, med, hi, _n = full_stat[nx]
        ax.scatter([q], [med], s=120, marker="s", color=cmap_nx[nx], alpha=0.85, edgecolor=(0, 0, 0, 0.25), linewidth=0.6)
        ax.vlines([q], [lo], [hi], color=pc.alpha_band(cmap_nx[nx], 0.65), lw=2.2)

    # shadow points
    for nx in nx_vals:
        qsh_list = []
        t_list = []
        for K0 in K0_sorted_by_M:
            # find q_shift from any row
            qsh = None
            for r in rows_t:
                if int(r.get("nx")) == nx and abs(float(r.get("K0")) - float(K0)) <= 1e-12:
                    qsh = int(r.get("q_shift"))
                    break
            if qsh is None:
                continue
            qsh_list.append(qsh)
            t_list.append(sp[(nx, K0)]["shadow_time"][1])
        ax.plot(qsh_list, t_list, marker="o", color=cmap_nx[nx], label=f"nx={nx}")

    ax.set_yscale("log")
    ax.set_xlabel("qubit proxy q")
    ax.set_ylabel(f"Aer time (s) [{args.speedup_metric}]")
    ax.set_title("Aer evolution time vs qubit proxy (full squares; shadow circles)")
    ax.grid(True, which="both", alpha=0.18)
    ax.legend(loc="best", frameon=True)
    outs.append(save(fig, outdir, "kdd_08_aer_time_vs_qubit_proxy.pdf"))

    # ========================
    # (09) Heatmap: log10 combined error vs (nx, M)
    # ========================
    nx_idx = {nx: i for i, nx in enumerate(nx_vals)}
    M_vals = Ms
    comb_err = np.full((len(nx_vals), len(M_vals)), np.nan, dtype=float)
    comb_sp = np.full((len(nx_vals), len(M_vals)), np.nan, dtype=float)
    for i, nx in enumerate(nx_vals):
        for j, K0 in enumerate(K0_sorted_by_M):
            er = stats[(nx, K0)]["rho"][0]
            em = stats[(nx, K0)]["mom"][0]
            comb_err[i, j] = math.log10(max(max(float(er), float(em)), EPS))
            comb_sp[i, j] = float(sp[(nx, K0)]["speedup"][1])

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), constrained_layout=True)
    vmin = float(np.nanmin(comb_err))
    vmax = float(np.nanmax(comb_err))
    im0 = axes[0].imshow(comb_err, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="cividis")
    axes[0].set_title(r"$\log_{10}$ combined error (vs full)")
    axes[0].set_xlabel("M")
    axes[0].set_ylabel("nx")
    axes[0].set_xticks(np.arange(len(M_vals)))
    axes[0].set_xticklabels([str(m) for m in M_vals])
    axes[0].set_yticks(np.arange(len(nx_vals)))
    axes[0].set_yticklabels([str(nx) for nx in nx_vals])

    # speedup heatmap
    sp_log = np.log10(np.maximum(comb_sp, 1.0))
    vmin2 = float(np.nanmin(sp_log))
    vmax2 = float(np.nanmax(sp_log))
    im1 = axes[1].imshow(sp_log, aspect="auto", origin="lower", vmin=vmin2, vmax=vmax2, cmap="viridis")
    axes[1].set_title(r"$\log_{10}$ speedup (full/shadow)")
    axes[1].set_xlabel("M")
    axes[1].set_ylabel("nx")
    axes[1].set_xticks(np.arange(len(M_vals)))
    axes[1].set_xticklabels([str(m) for m in M_vals])
    axes[1].set_yticks(np.arange(len(nx_vals)))
    axes[1].set_yticklabels([str(nx) for nx in nx_vals])

    c0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)
    c0.set_label("log10 error")
    c1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)
    c1.set_label("log10 speedup")
    fig.suptitle(f"Overview at t={t_eval:g}")
    outs.append(save(fig, outdir, "kdd_09_heatmaps_error_and_speedup.pdf"))

    # ========================
    # (10) Error vs time (nx_focus)
    # ========================
    nx_focus = int(args.nx_focus)
    K0_subset = select_K0_subset(sweep_rows, nx_focus, n=4)
    t_vals = sorted({float(r.get("t")) for r in sweep_rows if r.get("t") is not None})
    colors_k0 = plt.cm.cividis(np.linspace(0.18, 0.88, len(K0_subset)))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.8), constrained_layout=True)
    for ax, key, title in [
        (axes[0], "err_rho_vs_full", r"$\varepsilon_\rho(t)$"),
        (axes[1], "err_momentum_vs_full", r"$\varepsilon_{\mathbf{J}}(t)$"),
    ]:
        for col, K0 in zip(colors_k0, K0_subset):
            mu = []
            sd = []
            for t in t_vals:
                rs = [
                    r
                    for r in sweep_rows
                    if int(r.get("nx")) == nx_focus
                    and abs(float(r.get("K0")) - float(K0)) <= 1e-12
                    and abs(float(r.get("t")) - float(t)) <= 1e-12
                ]
                m, s, _n = mean_std([r.get(key) for r in rs])
                mu.append(m)
                sd.append(s)
            y = np.asarray(mu, dtype=float)
            s = np.asarray(sd, dtype=float)
            lo = np.maximum(y - s, EPS)
            hi = np.maximum(y + s, EPS)
            ax.plot(t_vals, y, color=col, lw=2.0, label=f"K0={K0:g}")
            ax.fill_between(t_vals, lo, hi, color=(col[0], col[1], col[2], 0.14), linewidth=0)
        ax.set_yscale("log")
        ax.set_xlabel("t")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.18)
    axes[0].legend(loc="best", frameon=True)
    fig.suptitle(f"Error vs time (nx={nx_focus})")
    outs.append(save(fig, outdir, f"kdd_10_error_vs_time_nx{nx_focus}.pdf"))

    # ========================
    # (11) Speedup vs time (nx_focus)
    # ========================
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.9), constrained_layout=True)
    for col, K0 in zip(colors_k0, K0_subset):
        mu = []
        lo = []
        hi = []
        for t in t_vals:
            rs = [
                r
                for r in speed_rows
                if int(r.get("nx")) == nx_focus
                and abs(float(r.get("K0")) - float(K0)) <= 1e-12
                and abs(float(r.get("t")) - float(t)) <= 1e-12
            ]
            q1, med, q3, _n = pct25_50_75([r.get("speedup") for r in rs])
            mu.append(med)
            lo.append(q1)
            hi.append(q3)
        ax.plot(t_vals, mu, color=col, lw=2.0, label=f"K0={K0:g}")
        ax.fill_between(t_vals, lo, hi, color=(col[0], col[1], col[2], 0.14), linewidth=0)
    ax.axhline(1.0, color="k", lw=1.0, ls=":", alpha=0.7)
    ax.set_xlabel("t")
    ax.set_ylabel("speedup (full/shadow)")
    ax.set_title(f"Aer speedup vs time (nx={nx_focus})")
    ax.grid(True, alpha=0.18)
    ax.legend(loc="best", frameon=True)
    outs.append(save(fig, outdir, f"kdd_11_speedup_vs_time_nx{nx_focus}.pdf"))

    # ========================
    # (12) Seed distribution (violin) at nx_focus, t_eval
    # ========================
    rs_focus = [r for r in rows_t if int(r.get("nx")) == nx_focus]
    if rs_focus:
        # by K0 (ordered by M)
        data_err = []
        data_spd = []
        labels = []
        for K0 in K0_sorted_by_M:
            rr = [r for r in rs_focus if abs(float(r.get("K0")) - float(K0)) <= 1e-12]
            if not rr:
                continue
            comb = [max(float(r.get("err_rho_vs_full")), float(r.get("err_momentum_vs_full"))) for r in rr]
            data_err.append([math.log10(max(v, EPS)) for v in comb])

            rr_sp = [r for r in speed_rows_t if int(r.get("nx")) == nx_focus and abs(float(r.get("K0")) - float(K0)) <= 1e-12]
            spv = [float(r.get("speedup")) for r in rr_sp if r.get("speedup") is not None]
            data_spd.append(spv)

            labels.append(str(int(K0_to_M[float(K0)])))

        fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.8), constrained_layout=True)
        parts = axes[0].violinplot(data_err, showmeans=True, showmedians=False, showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor("#1C2C3A")
            b.set_alpha(0.55)
        axes[0].set_title(r"seed dist: $\log_{10}$ combined error")
        axes[0].set_xticks(np.arange(1, len(labels) + 1))
        axes[0].set_xticklabels(labels)
        axes[0].set_xlabel("M")
        axes[0].grid(True, alpha=0.18)

        parts = axes[1].violinplot(data_spd, showmeans=True, showmedians=False, showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor("#2E6F6D")
            b.set_alpha(0.55)
        axes[1].set_title("seed dist: speedup")
        axes[1].set_xticks(np.arange(1, len(labels) + 1))
        axes[1].set_xticklabels(labels)
        axes[1].set_xlabel("M")
        axes[1].grid(True, alpha=0.18)

        fig.suptitle(f"Seed distributions (nx={nx_focus}, t={t_eval:g})")
        outs.append(save(fig, outdir, f"kdd_12_seed_distributions_violin_nx{nx_focus}.pdf"))

    # ========================
    # (13) Runtime breakdown (shadow) vs M at nx_focus
    # ========================
    rs_sp = [r for r in sweep_rows if int(r.get("nx")) == nx_focus and abs(float(r.get("t")) - t_eval) <= 1e-12]
    if rs_sp:
        # median components per K0
        comp = {
            "aer_transpile": [],
            "aer_run": [],
            "post": [],
            "metrics": [],
        }
        for K0 in K0_sorted_by_M:
            rr = [r for r in rs_sp if abs(float(r.get("K0")) - float(K0)) <= 1e-12]
            comp["aer_transpile"].append(np.median(_finite([r.get("rt_shadow_aer_transpile_s") for r in rr])))
            comp["aer_run"].append(np.median(_finite([r.get("rt_shadow_aer_run_s") for r in rr])))
            comp["post"].append(np.median(_finite([r.get("rt_shadow_post_s") for r in rr])))
            comp["metrics"].append(np.median(_finite([r.get("rt_metrics_s") for r in rr])))

        fig, ax = plt.subplots(1, 1, figsize=(9.2, 3.9), constrained_layout=True)
        x = np.arange(len(Ms))
        bottom = np.zeros_like(x, dtype=float)
        cols = ["#BCCAD2", "#516B7A", "#2E6F6D", "#6B4F3A"]
        for (name, col) in zip(["aer_transpile", "aer_run", "post", "metrics"], cols):
            y = np.asarray(comp[name], dtype=float)
            ax.bar(x, y, bottom=bottom, width=0.72, color=col, alpha=0.9, label=name)
            bottom = bottom + y
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([str(m) for m in Ms])
        ax.set_xlabel("M")
        ax.set_ylabel("time per point (s)")
        ax.set_title(f"Shadow pipeline breakdown (nx={nx_focus}, t={t_eval:g})")
        ax.grid(True, which="both", alpha=0.18)
        ax.legend(loc="best", frameon=True, ncol=2)
        outs.append(save(fig, outdir, f"kdd_13_shadow_runtime_breakdown_nx{nx_focus}.pdf"))

    # ========================
    # (14) Summary panel: error + speedup vs M
    # ========================
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.2), constrained_layout=True)
    for j, metric in enumerate(["rho", "mom"]):
        ax = axes[0, j]
        for nx in nx_vals:
            color = cmap_nx[nx]
            mu = np.asarray([stats[(nx, K0)][metric][0] for K0 in K0_sorted_by_M], dtype=float)
            ax.plot(Ms, mu, marker="o", color=color, label=f"nx={nx}")
        ax.set_yscale("log")
        ax.set_xlabel("M")
        ax.set_title(r"$\varepsilon_\rho$" if metric == "rho" else r"$\varepsilon_{\mathbf{J}}$")
        ax.grid(True, which="both", alpha=0.18)

    ax = axes[1, 0]
    for nx in nx_vals:
        med = [sp[(nx, K0)]["speedup"][1] for K0 in K0_sorted_by_M]
        ax.plot(Ms, med, marker="o", color=cmap_nx[nx], label=f"nx={nx}")
    ax.axhline(1.0, color="k", lw=1.0, ls=":", alpha=0.7)
    ax.set_xlabel("M")
    ax.set_ylabel("speedup")
    ax.set_title("Aer speedup")
    ax.grid(True, alpha=0.18)
    ax.legend(loc="best", frameon=True)

    ax = axes[1, 1]
    for nx in nx_vals:
        sh_med = [sp[(nx, K0)]["shadow_time"][1] for K0 in K0_sorted_by_M]
        ax.plot(Ms, sh_med, marker="o", color=cmap_nx[nx], label=f"shadow nx={nx}")
        f_med = full_stat[nx][1]
        ax.axhline(f_med, color=cmap_nx[nx], ls="--", alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("M")
    ax.set_ylabel(f"Aer time (s) [{args.speedup_metric}]")
    ax.set_title("Aer time (shadow curves; full dashed)" )
    ax.grid(True, which="both", alpha=0.18)

    fig.suptitle(f"KDD gallery summary (t={t_eval:g})")
    outs.append(save(fig, outdir, "kdd_14_summary_panel.pdf"))

    # write an index
    idx = outdir / "INDEX.md"
    with idx.open("w") as f:
        f.write("# KDD Gallery Figure Index\n\n")
        for p in outs:
            f.write(f"- {p.name}\n")
    print(f"Wrote: {idx}")
    for p in outs:
        print(f"Wrote: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
