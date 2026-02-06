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
