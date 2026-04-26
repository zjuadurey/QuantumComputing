#!/usr/bin/env python
"""Plot intervention effects on logical failure rate."""

from __future__ import annotations

import argparse
import html
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from failureops.io_utils import ensure_parent_dir, read_csv_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/results/attribution_summary.csv")
    parser.add_argument("--output", default="figures/intervention_delta_lfr.png")
    parser.add_argument("--title", default="FailureOps intervention sensitivity")
    args = parser.parse_args()

    rows = read_csv_rows(args.input)
    try:
        _plot_with_matplotlib(rows, args.output, args.title)
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        _plot_with_svg_convert(rows, args.output, args.title)
    print(f"wrote plot to {args.output}")


def _plot_with_matplotlib(rows: list[dict[str, str]], output: str, title: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/failureops-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    interventions = [row["intervention"] for row in rows]
    deltas = [float(row["absolute_delta_lfr"]) for row in rows]
    colors = ["#2f7f7f" if value <= 0 else "#b65f35" for value in deltas]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(interventions, deltas, color=colors)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_ylabel("Absolute delta LFR")
    ax.set_xlabel("Intervention")
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()

    ensure_parent_dir(output)
    fig.savefig(output, dpi=180)


def _plot_with_svg_convert(rows: list[dict[str, str]], output: str, title: str) -> None:
    convert = shutil.which("convert")
    if convert is None:
        raise RuntimeError(
            "plotting requires either matplotlib in the Python environment "
            "or ImageMagick's convert command"
        )

    svg = _build_svg_bar_chart(rows, title)
    ensure_parent_dir(output)
    with tempfile.NamedTemporaryFile("w", suffix=".svg", delete=False) as handle:
        handle.write(svg)
        svg_path = handle.name
    try:
        subprocess.run([convert, svg_path, output], check=True)
    finally:
        Path(svg_path).unlink(missing_ok=True)


def _build_svg_bar_chart(rows: list[dict[str, str]], title: str) -> str:
    width = 1100
    height = 620
    margin_left = 88
    margin_right = 40
    margin_top = 55
    margin_bottom = 170
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    interventions = [row["intervention"] for row in rows]
    deltas = [float(row["absolute_delta_lfr"]) for row in rows]
    ymin = min(0.0, min(deltas))
    ymax = max(0.0, max(deltas))
    if ymin == ymax:
        ymin -= 1.0
        ymax += 1.0
    padding = (ymax - ymin) * 0.12
    ymin -= padding
    ymax += padding

    def y_pos(value: float) -> float:
        return margin_top + (ymax - value) / (ymax - ymin) * plot_height

    zero_y = y_pos(0.0)
    slot = plot_width / max(1, len(rows))
    bar_width = slot * 0.62

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="550" y="28" text-anchor="middle" font-family="DejaVu Sans, Arial, sans-serif" font-size="22">{html.escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{zero_y:.2f}" x2="{width - margin_right}" y2="{zero_y:.2f}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#333" stroke-width="1"/>',
        f'<text x="22" y="{margin_top + plot_height / 2:.2f}" transform="rotate(-90 22 {margin_top + plot_height / 2:.2f})" text-anchor="middle" font-family="DejaVu Sans, Arial, sans-serif" font-size="15">Absolute delta LFR</text>',
        f'<text x="590" y="{height - 16}" text-anchor="middle" font-family="DejaVu Sans, Arial, sans-serif" font-size="15">Intervention</text>',
    ]

    for tick in _ticks(ymin, ymax):
        y = y_pos(tick)
        parts.append(
            f'<line x1="{margin_left - 5}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#ddd" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{margin_left - 10}" y="{y + 5:.2f}" text-anchor="end" font-family="DejaVu Sans, Arial, sans-serif" font-size="12">{tick:.2f}</text>'
        )

    for index, (intervention, delta) in enumerate(zip(interventions, deltas)):
        center_x = margin_left + slot * index + slot / 2
        bar_x = center_x - bar_width / 2
        bar_y = min(y_pos(delta), zero_y)
        bar_h = abs(zero_y - y_pos(delta))
        color = "#2f7f7f" if delta <= 0 else "#b65f35"
        parts.append(
            f'<rect x="{bar_x:.2f}" y="{bar_y:.2f}" width="{bar_width:.2f}" height="{bar_h:.2f}" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{center_x:.2f}" y="{bar_y - 6:.2f}" text-anchor="middle" font-family="DejaVu Sans, Arial, sans-serif" font-size="11">{delta:.3f}</text>'
        )
        parts.append(
            f'<text x="{center_x:.2f}" y="{height - margin_bottom + 22}" text-anchor="middle" font-family="DejaVu Sans, Arial, sans-serif" font-size="13">{index + 1}</text>'
        )

    legend_y = height - margin_bottom + 52
    legend_x = margin_left
    column_width = 470
    for index, intervention in enumerate(interventions):
        column = 0 if index < 5 else 1
        row = index if index < 5 else index - 5
        x = legend_x + column * column_width
        y = legend_y + row * 22
        safe_label = html.escape(intervention)
        parts.append(
            f'<text x="{x}" y="{y}" font-family="DejaVu Sans, Arial, sans-serif" font-size="13">{index + 1}. {safe_label}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def _ticks(ymin: float, ymax: float) -> list[float]:
    step = (ymax - ymin) / 5
    return [ymin + step * index for index in range(6)]


if __name__ == "__main__":
    main()
