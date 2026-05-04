from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path(".").resolve()

# =========================================================
# Inputs: original three CSV files
# =========================================================
PATHS_BY_PANEL = {
    "Fail rate": "exp04_chain_failrate_over_ideal.csv",
    "MAE overall": "exp04_chain_mean_over_db.csv",
    "MAE only fail": "exp04_chain_mean_over_db_given_over.csv",
}

OUT_BASE = "Figure7_chain_neurips_subset_rightlegend_square"

X_KEEP = [0.00, 0.01, 0.03, 0.05, 0.07, 0.09, 0.10]
X_TICK_LABELS = ["0", "1", "3", "5", "7", "9", "10"]

FIG_W_IN = 9.0
FIG_H_IN = 2.40

AX_SIDE_IN = 1.58
AX_CENTERS_X = [0.15, 0.40, 0.65]
AX_CENTER_Y = 0.60

LEG_ANCHOR_X = 0.77
LEG_ANCHOR_Y = AX_CENTER_Y

METHOD_ORDER = [
    "NE24_v1",
    "NE24_v2",
    "QA-Google_v1",
    "QA-Google_v2",
    "QE-IBM_v1",
    "QE-IBM_v2",
    "Agent2",
]

MARKERS = ["o", "s", "^", "v", "D", "P", "*", "X", "+"]

TITLE_MAP = {
    "Fail rate": "(a) fail rate",
    "MAE overall": "(b) MAE overall (dB)",
    "MAE only fail": "(c) MAE only fail (dB)",
}

SOLID_METHODS = {"Agent2"}

BASELINE_LINEWIDTH = 1.8
MAIN_LINEWIDTH = 2.2

COLORS = {
    "NE24_v1": "C0",
    "NE24_v2": "C1",
    "QA-Google_v1": "C2",
    "QA-Google_v2": "C3",
    "QE-IBM_v1": "C4",
    "QE-IBM_v2": "C5",
    "Agent2": "C6",
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


def ordered_methods(methods: list[str]) -> list[str]:
    ordered = [m for m in METHOD_ORDER if m in methods]
    ordered += [m for m in methods if m not in ordered]
    return ordered


def load_wide_csv_and_filter(path: Path, x_keep: list[float], tol: float = 1e-10):
    df = pd.read_csv(path)

    x_cols = [c for c in df.columns if c != "method"]
    parsed_cols = [(float(c), c) for c in x_cols]

    selected_cols = []
    for target in x_keep:
        dist, col = min((abs(x_val - target), col) for x_val, col in parsed_cols)
        if dist > tol:
            raise ValueError(f"{path.name} 缺少横轴点 {target}; 当前列为: {x_cols}")
        selected_cols.append(col)

    series = {
        str(row["method"]): [float(row[col]) for col in selected_cols]
        for _, row in df.iterrows()
    }

    return list(x_keep), series


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
    panels = ["Fail rate", "MAE overall", "MAE only fail"]

    loaded = {}
    for panel in panels:
        csv_path = BASE_DIR / PATHS_BY_PANEL[panel]
        loaded[panel] = load_wide_csv_and_filter(csv_path, X_KEEP)

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
                linestyle="-" if method in SOLID_METHODS else "--",
                linewidth=MAIN_LINEWIDTH if method in SOLID_METHODS else BASELINE_LINEWIDTH,
                color=COLORS.get(method, None),
                label=method,
            )

        ax.set_title(TITLE_MAP[panel], pad=6)

        if ax_idx == 0:
            ax.set_ylabel("Error rate")
        else:
            ax.set_ylabel("")

        ax.grid(True, linestyle="--", alpha=0.4)

        ax.set_xticks(X_KEEP)
        ax.set_xticklabels(X_TICK_LABELS)

        xr = max(X_KEEP) - min(X_KEEP)
        ax.set_xlim(min(X_KEEP) - 0.03 * xr, max(X_KEEP) + 0.03 * xr)

        ax.set_box_aspect(1)

    axes[1].set_xlabel("Noise level", labelpad=4)

    axes[2].annotate(
        r"$(\times 10^{-2})$",
        xy=(X_KEEP[-1], 0),
        xycoords=("data", "axes fraction"),
        xytext=(8, -10),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=9.5,
        annotation_clip=False,
    )

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