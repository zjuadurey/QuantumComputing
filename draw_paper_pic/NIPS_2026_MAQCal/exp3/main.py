from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path(".").resolve()

# =========================================================
# Inputs: budget-scaling MAE under three noise families
# =========================================================
PATHS_BY_PANEL = {
    "IQ Chain": "exp02_summary_mean_abs_error_vs_budget_iq_chain_nl0.05.csv",
    "Chain": "exp02_summary_mean_abs_error_vs_budget_chain_nl0.05.csv",
    "Realistic Chain": "exp02_summary_mean_abs_error_vs_budget_realistic_chain_nl0.05.csv",
}

OUT_BASE = "Figure_budget_scaling_mae_neurips_style"

# =========================================================
# Figure geometry: same style as the previous figure
# =========================================================
FIG_W_IN = 9.0
FIG_H_IN = 2.40

AX_SIDE_IN = 1.58
AX_CENTERS_X = [0.15, 0.40, 0.65]
AX_CENTER_Y = 0.60

LEG_ANCHOR_X = 0.77
LEG_ANCHOR_Y = AX_CENTER_Y

# =========================================================
# Method style
# =========================================================
METHOD_ORDER = [
    "NE24",
    "QA-Google",
    "QE-IBM",
    "Agent1",
]

MARKERS = ["o", "s", "^", "*", "D", "P", "X", "+"]

TITLE_MAP = {
    "IQ Chain": "(a) IQ chain",
    "Chain": "(b) chain",
    "Realistic Chain": "(c) realistic chain",
}

# Our method: solid; baselines: dashed
SOLID_METHODS = {"Agent1"}

BASELINE_LINEWIDTH = 1.8
MAIN_LINEWIDTH = 2.2

# Fixed matplotlib default colors, preserving method-color order
COLORS = {
    "NE24": "C0",
    "QA-Google": "C1",
    "QE-IBM": "C2",
    "Agent1": "C3",
}

plt.rcParams.update(
    {
        "font.size": 9.5,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11.0,
        "axes.titleweight": "bold",
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "lines.linewidth": BASELINE_LINEWIDTH,
        "lines.markersize": 4.8,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 3.8,
        "ytick.major.size": 3.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
    }
)


def load_wide_csv(path: Path):
    """
    CSV format:
        rows    = method
        columns = budget values, e.g. 201, 301, ..., 1001
    """
    df = pd.read_csv(path)

    x_cols = [c for c in df.columns if c != "method"]
    x = sorted(float(c) for c in x_cols)
    col_of = {float(c): c for c in x_cols}

    series = {}
    for _, row in df.iterrows():
        method = str(row["method"])
        series[method] = [float(row[col_of[v]]) for v in x]

    return x, series


def ordered_methods(methods: list[str]) -> list[str]:
    ordered = [m for m in METHOD_ORDER if m in methods]
    ordered += [m for m in methods if m not in ordered]
    return ordered


def choose_ticks(x: list[float], max_ticks: int = 5) -> list[float]:
    """
    Pick up to max_ticks ticks evenly across x while keeping endpoints.
    """
    if len(x) <= max_ticks:
        return x

    n = len(x)
    raw = [i * (n - 1) / (max_ticks - 1) for i in range(max_ticks)]

    idxs = []
    for v in raw:
        j = int(round(v))
        if j not in idxs:
            idxs.append(j)

    if 0 not in idxs:
        idxs = [0] + idxs
    if (n - 1) not in idxs:
        idxs = idxs + [n - 1]

    idxs = sorted(set(idxs))
    return [x[i] for i in idxs]


def is_almost_int(v: float, eps: float = 1e-9) -> bool:
    return abs(v - round(v)) < eps


def get_linestyle(method: str) -> str:
    return "-" if method in SOLID_METHODS else "--"


def get_linewidth(method: str) -> float:
    return MAIN_LINEWIDTH if method in SOLID_METHODS else BASELINE_LINEWIDTH


def axes_rect(center_x: float, center_y: float, side_in: float):
    width_frac = side_in / FIG_W_IN
    height_frac = side_in / FIG_H_IN

    return [
        center_x - width_frac / 2,
        center_y - height_frac / 2,
        width_frac,
        height_frac,
    ]


def add_square_axes(fig, center_x: float, center_y: float, side_in: float):
    return fig.add_axes(axes_rect(center_x, center_y, side_in))


def plot():
    panels = list(PATHS_BY_PANEL.keys())

    loaded = {}
    for panel in panels:
        loaded[panel] = load_wide_csv(BASE_DIR / PATHS_BY_PANEL[panel])

    methods = ordered_methods(list(loaded[panels[0]][1].keys()))

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
    axes = [add_square_axes(fig, cx, AX_CENTER_Y, AX_SIDE_IN) for cx in AX_CENTERS_X]

    for ax_idx, (ax, panel) in enumerate(zip(axes, panels)):
        x, series = loaded[panel]

        for i, method in enumerate(methods):
            if method not in series:
                continue

            ax.plot(
                x,
                series[method],
                marker=MARKERS[i % len(MARKERS)],
                linestyle=get_linestyle(method),
                linewidth=get_linewidth(method),
                color=COLORS.get(method, None),
                label=method,
            )

        ax.set_title(TITLE_MAP[panel], pad=6)
        ax.set_ylabel("MAE" if ax_idx == 0 else "")

        ax.grid(True, linestyle="--", alpha=0.4)

        ticks = choose_ticks(x, max_ticks=5)
        ax.set_xticks(ticks)

        if all(is_almost_int(t) for t in ticks):
            ax.set_xticklabels([str(int(round(t))) for t in ticks])

        xr = max(x) - min(x)
        pad = 0.03 * xr if xr > 0 else 1.0
        ax.set_xlim(min(x) - pad, max(x) + pad)

        # Keep every subplot square.
        ax.set_box_aspect(1)

    axes[1].set_xlabel("Budget", labelpad=4)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(LEG_ANCHOR_X, LEG_ANCHOR_Y),
        bbox_transform=fig.transFigure,
        ncol=1,
        frameon=True,
        handlelength=2.8,
        handletextpad=0.65,
        borderpad=0.45,
        labelspacing=0.56,
    )

    out_png = BASE_DIR / f"{OUT_BASE}.png"
    out_pdf = BASE_DIR / f"{OUT_BASE}.pdf"
    out_svg = BASE_DIR / f"{OUT_BASE}.svg"

    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.015)

    plt.close(fig)

    return out_png, out_pdf, out_svg


def main():
    for path in plot():
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()