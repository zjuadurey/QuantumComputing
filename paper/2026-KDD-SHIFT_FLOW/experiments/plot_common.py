"""experiments/plot_common.py

Small helpers for reading sweep CSVs and aggregating.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


INT_FIELDS = {
    "nx",
    "N",
    "M",
    "seed",
    "shift_x",
    "shift_y",
    "k0_1_y",
    "k0_1_x",
    "k0_2_y",
    "k0_2_x",
    "q_base",
    "q_shift",
}


def load_sweep_csv(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for raw in r:
            row: dict[str, Any] = {}
            for k, v in raw.items():
                if v is None:
                    row[k] = None
                    continue
                s = v.strip()
                if s == "":
                    row[k] = None
                    continue
                if k in INT_FIELDS:
                    try:
                        row[k] = int(s)
                        continue
                    except Exception:
                        pass
                try:
                    row[k] = float(s)
                except Exception:
                    row[k] = s
            rows.append(row)
    return rows


def mean_std(values: Iterable[float]) -> tuple[float, float, int]:
    xs = [float(x) for x in values if x is not None and np.isfinite(float(x))]
    if not xs:
        return float("nan"), float("nan"), 0
    arr = np.asarray(xs, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0)), int(arr.size)


def filter_close(rows: list[dict[str, Any]], key: str, value: float, tol: float = 1e-12) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        x = r.get(key, None)
        if x is None:
            continue
        try:
            if abs(float(x) - float(value)) <= tol:
                out.append(r)
        except Exception:
            continue
    return out


def unique_sorted(rows: list[dict[str, Any]], key: str) -> list[Any]:
    vals = set()
    for r in rows:
        v = r.get(key, None)
        if v is None:
            continue
        vals.add(v)
    try:
        return sorted(vals)
    except Exception:
        return list(vals)


def groupby(rows: list[dict[str, Any]], keys: list[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    g: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for r in rows:
        k = tuple(r.get(name, None) for name in keys)
        g.setdefault(k, []).append(r)
    return g


def max_t(rows: list[dict[str, Any]]) -> float:
    ts = [float(r["t"]) for r in rows if r.get("t", None) is not None]
    return float(max(ts)) if ts else float("nan")


def apply_mpl_style() -> None:
    import matplotlib.pyplot as plt

    # Paper-style defaults (journal/conference friendly)
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        # matplotlib without seaborn styles
        pass

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "axes.linewidth": 0.8,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "dejavuserif",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ensure_figdir(figdir: str | Path) -> Path:
    p = Path(figdir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def color_cycle(n: int) -> list[str]:
    # Okabe-Ito colorblind-friendly palette (subset)
    base = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7"]
    if n <= len(base):
        return base[:n]
    out = []
    for i in range(n):
        out.append(base[i % len(base)])
    return out


def alpha_band(color: str, alpha: float = 0.18) -> tuple[float, float, float, float]:
    """Convert a hex color to RGBA with the given alpha."""
    import matplotlib.colors as mcolors

    r, g, b, _ = mcolors.to_rgba(color)
    return (r, g, b, float(alpha))


# ─────────────────────────────────────────────────────────────
# Paper-ready visual style  (NeurIPS / ICML / KDD)
# ─────────────────────────────────────────────────────────────
PALETTE_PASTEL = ["#7CB8D6", "#F2A65A", "#82C9A0", "#C9A0DC", "#F2857D"]
SPINE_COLOR = "#404040"
TEXT_COLOR = "#2F2F2F"


def apply_paper_rcparams() -> None:
    """Set global rcParams for paper-ready figures (call once per script)."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8,
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def set_paper_style(ax) -> None:
    """Apply clean axes: left+bottom spines only, no grid, inward ticks."""
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(1.2)
        ax.spines[sp].set_color(SPINE_COLOR)
    ax.tick_params(
        axis="both", which="major", direction="in",
        length=4, width=1.0, color=SPINE_COLOR, labelcolor=TEXT_COLOR,
    )
    ax.tick_params(
        axis="both", which="minor", direction="in",
        length=2, width=0.8, color=SPINE_COLOR,
    )
    ax.grid(False)


def smooth_errorband(
    xs,
    ys,
    yerr,
    *,
    log_scale: bool = False,
    n_fine: int = 200,
    sigma: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate + light smooth → (xs_fine, ys_smooth, lo, hi).

    Uses scipy PCHIP + gaussian_filter1d if available; falls back to
    numpy linear interpolation + box smooth otherwise.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    yerr = np.asarray(yerr, dtype=float)

    xs_fine = np.linspace(xs.min(), xs.max(), n_fine)

    if log_scale:
        work_y = np.log10(np.maximum(ys, 1e-30))
        work_lo = np.log10(np.maximum(ys - yerr, ys * 0.01))
        work_hi = np.log10(ys + yerr)
    else:
        work_y, work_lo, work_hi = ys, ys - yerr, ys + yerr

    try:
        from scipy.interpolate import PchipInterpolator
        from scipy.ndimage import gaussian_filter1d as gf1d

        def _interp(v):
            out = PchipInterpolator(xs, v)(xs_fine)
            return gf1d(out, sigma) if sigma > 0 else out
    except ImportError:
        def _interp(v):
            out = np.interp(xs_fine, xs, v)
            if sigma > 0:
                w = max(1, int(sigma * 3))
                k = np.ones(2 * w + 1) / (2 * w + 1)
                out = np.convolve(out, k, mode="same")
            return out

    sy, slo, shi = _interp(work_y), _interp(work_lo), _interp(work_hi)

    if log_scale:
        return xs_fine, 10**sy, 10**slo, 10**shi
    return xs_fine, sy, slo, shi
