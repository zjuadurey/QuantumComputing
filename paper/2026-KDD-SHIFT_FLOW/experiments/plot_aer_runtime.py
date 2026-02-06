"""experiments/plot_aer_runtime.py

Plot Aer timing comparison: shadow vs full evolution.

Inputs:
- Shadow sweep CSV produced with: `--shadow-backend aer`
- Full Aer timing CSV produced with: `--record-full-aer-times`

This script focuses on Aer *run* time by default (not transpile time).
"""

from __future__ import annotations

import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

import plot_common as pc


def _t_key(t: float) -> str:
    return f"{float(t):.12g}"


def _finite(xs):
    out = []
    for x in xs:
        if x is None:
            continue
        try:
            v = float(x)
        except Exception:
            continue
        if math.isfinite(v):
            out.append(v)
    return out


def _pcts(vals: list[float]) -> tuple[float, float, float, int]:
    v = np.asarray(_finite(vals), dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    return (
        float(np.percentile(v, 25)),
        float(np.percentile(v, 50)),
        float(np.percentile(v, 75)),
        int(v.size),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shadow", default="results/sweep.csv", help="shadow sweep CSV (shadow_backend=qiskit_aer)")
    ap.add_argument("--full", default="results/full_aer_times.csv", help="full Aer timing CSV")
    ap.add_argument("--figdir", default="figs")
    ap.add_argument("--out", default="exp_aer_runtime_shadow_vs_full.pdf")
    ap.add_argument("--metric", choices=["run", "total"], default="run")
    ap.add_argument("--t", type=float, default=None, help="optional: select a single t")
    args = ap.parse_args()

    shadow_rows = pc.load_sweep_csv(args.shadow)
    full_rows = pc.load_sweep_csv(args.full)

    if args.t is not None:
        shadow_rows = pc.filter_close(shadow_rows, "t", float(args.t))
        full_rows = pc.filter_close(full_rows, "t", float(args.t))

    if not shadow_rows:
        raise SystemExit(f"No shadow rows found in {args.shadow}")
    if not full_rows:
        raise SystemExit(f"No full rows found in {args.full}")

    # Choose timing columns
    if args.metric == "run":
        key_shadow = "rt_shadow_aer_run_s"
        key_full = "rt_full_aer_run_s"
    else:
        key_shadow = "rt_shadow_aer_run_s"  # plus transpile
        key_full = "rt_full_aer_run_s"

    # --- build full timing map keyed by (nx, seed, t) ---
    full_map: dict[tuple[int, int, str], float] = {}
    full_by_nx: dict[int, list[float]] = {}
    for r in full_rows:
        nx = r.get("nx")
        seed = r.get("seed")
        t = r.get("t")
        if nx is None or seed is None or t is None:
            continue

        v_run = r.get(key_full)
        v_tr = r.get("rt_full_aer_transpile_s")
        if v_run is None:
            continue

        v = float(v_run)
        if args.metric == "total" and v_tr is not None and math.isfinite(float(v_tr)):
            v = v + float(v_tr)

        if not math.isfinite(v):
            continue
        k = (int(nx), int(seed), _t_key(float(t)))
        full_map[k] = v
        full_by_nx.setdefault(int(nx), []).append(v)

    if not full_map:
        raise SystemExit(
            f"No finite full Aer times in {args.full}. "
            "Did you run: python3 experiments/run_sweep.py --record-full-aer-times ... ?"
        )

    # --- shadow groups ---
    shadow_groups: dict[tuple[int, float], dict[str, object]] = {}
    missing_full = 0
    total_shadow = 0
    for r in shadow_rows:
        if r.get("shadow_backend") != "qiskit_aer":
            continue
        nx = r.get("nx")
        seed = r.get("seed")
        t = r.get("t")
        K0 = r.get("K0")
        M = r.get("M")
        if nx is None or seed is None or t is None or K0 is None or M is None:
            continue

        v_run = r.get(key_shadow)
        v_tr = r.get("rt_shadow_aer_transpile_s")
        if v_run is None:
            continue

        v = float(v_run)
        if args.metric == "total" and v_tr is not None and math.isfinite(float(v_tr)):
            v = v + float(v_tr)
        if not math.isfinite(v):
            continue

        total_shadow += 1
        gk = (int(nx), float(K0))
        g = shadow_groups.setdefault(
            gk,
            {
                "nx": int(nx),
                "K0": float(K0),
                "M": int(M),
                "shadow": [],
                "ratio": [],
            },
        )
        g["shadow"].append(v)

        fk = (int(nx), int(seed), _t_key(float(t)))
        if fk in full_map:
            g["ratio"].append(full_map[fk] / v)
        else:
            missing_full += 1

    if not shadow_groups:
        raise SystemExit(
            f"No shadow Aer rows found in {args.shadow}. "
            "Did you run: python3 experiments/run_sweep.py --shadow-backend aer ... ?"
        )

    # --- prep plot data ---
    pc.apply_mpl_style()
    figdir = pc.ensure_figdir(args.figdir)

    nx_vals = sorted({k[0] for k in shadow_groups.keys()})
    colors = pc.color_cycle(len(nx_vals))

    # full per nx stats
    full_stat = {nx: _pcts(full_by_nx.get(nx, [])) for nx in nx_vals}
    missing_full_nx = [nx for nx in nx_vals if not math.isfinite(full_stat[nx][1])]
    if missing_full_nx:
        raise SystemExit(f"Missing full Aer times for nx={missing_full_nx} in {args.full}")

    # sort points by M within each nx
    by_nx: dict[int, list[tuple[int, float, tuple[float, float, float, int], tuple[float, float, float, int]]]] = {}
    for (nx, K0), g in shadow_groups.items():
        M = int(g["M"])
        sh = _pcts(g["shadow"])
        ra = _pcts(g["ratio"])
        by_nx.setdefault(int(nx), []).append((M, float(K0), sh, ra))
    for nx in by_nx:
        by_nx[nx].sort(key=lambda t: t[0])

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9), constrained_layout=True)

    # Panel A: Aer runtime
    ax = axes[0]
    for color, nx in zip(colors, nx_vals):
        pts = by_nx.get(nx, [])
        Ms = [p[0] for p in pts]
        med = [p[2][1] for p in pts]
        lo = [p[2][0] for p in pts]
        hi = [p[2][2] for p in pts]
        ax.plot(Ms, med, marker="o", color=color, label=f"shadow nx={nx}")
        ax.fill_between(Ms, lo, hi, color=pc.alpha_band(color, 0.16), linewidth=0)

        f_lo, f_med, f_hi, _n = full_stat[nx]
        ax.axhline(f_med, color=color, ls="--", alpha=0.85, label=f"full nx={nx}")
        ax.axhspan(f_lo, f_hi, color=pc.alpha_band(color, 0.08), linewidth=0)

    ax.set_yscale("log")
    ax.set_xlabel("M (# retained modes)")
    ax.set_ylabel(f"Aer time (s) [{args.metric}]")
    ax.set_title("Aer evolution time: shadow vs full")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", frameon=True, fontsize=8, ncol=2)

    # Panel B: speedup
    ax = axes[1]
    all_lo = []
    all_hi = []
    for color, nx in zip(colors, nx_vals):
        pts = by_nx.get(nx, [])
        Ms = [p[0] for p in pts]
        med = [p[3][1] for p in pts]
        lo = [p[3][0] for p in pts]
        hi = [p[3][2] for p in pts]
        ax.plot(Ms, med, marker="o", color=color, label=f"nx={nx}")
        ax.fill_between(Ms, lo, hi, color=pc.alpha_band(color, 0.16), linewidth=0)
        all_lo.extend(lo)
        all_hi.extend(hi)

    # A linear y-axis is more interpretable for speedups in this range.
    ax.axhline(1.0, color="k", lw=1.1, ls=":", alpha=0.75)
    ymax = max([v for v in all_hi if math.isfinite(float(v))] + [1.0])
    ax.set_ylim(0.0, ymax * 1.08)
    ax.set_xlabel("M (# retained modes)")
    ax.set_ylabel("speedup  (full / shadow)")
    ax.set_title("Speedup (linear)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", frameon=True)

    # Optional inset zoom to make small speedups visible alongside large ones.
    try:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

        if ymax >= 10.0:
            # Place the inset lower so it doesn't cover the high-speedup curve (nx=7).
            try:
                axins = ax.inset_axes([0.20, 0.25, 0.50, 0.30])
            except Exception:
                axins = inset_axes(ax, width="46%", height="40%", loc="center right", borderpad=1.1)
            M_all: list[float] = []
            for color, nx in zip(colors, nx_vals):
                pts = by_nx.get(nx, [])
                Ms = [p[0] for p in pts]
                med = [p[3][1] for p in pts]
                lo = [p[3][0] for p in pts]
                hi = [p[3][2] for p in pts]
                M_all.extend([float(m) for m in Ms])
                axins.plot(Ms, med, marker="o", ms=3, lw=1.4, color=color)
                axins.fill_between(Ms, lo, hi, color=pc.alpha_band(color, 0.12), linewidth=0)

            axins.axhline(1.0, color="k", lw=0.9, ls=":", alpha=0.75)
            if M_all:
                axins.set_xlim(min(M_all), max(M_all))
            axins.set_ylim(0.8, 6.0)
            axins.grid(True, alpha=0.25)
            axins.set_xticks([])
            axins.set_yticks([1, 2, 4, 6])
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec=(0, 0, 0, 0.25), lw=0.8)
    except Exception:
        # If inset toolkit isn't available, keep the main plot only.
        pass

    t_note = "all t" if args.t is None else f"t={args.t:g}"
    fig.suptitle(f"Aer timing comparison ({t_note})")

    out = figdir / args.out
    fig.savefig(out)
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
