"""Aggregate metric JSON files into a comparison table and simple plots."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build comparison tables/plots from result directories")
    p.add_argument("--result-dirs", required=True, help="comma-separated list of output directories")
    p.add_argument("--out-dir", default=str(ROOT / "results" / "comparison_tables"))
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for result_dir in [Path(x.strip()) for x in args.result_dirs.split(",") if x.strip()]:
        metric_path = result_dir / "test_metrics.json"
        if not metric_path.exists():
            continue
        metrics = json.loads(metric_path.read_text())
        rows.append(
            {
                "run_label": result_dir.name,
                "method": metrics.get("method", result_dir.name),
                "density_rel_l2": metrics.get("density_rel_l2"),
                "lowfreq_spectral_rel_l2": metrics.get("lowfreq_spectral_rel_l2"),
                "lowpass_energy_rel_l2": metrics.get("lowpass_energy_rel_l2"),
                "commutator_leakage": metrics.get("commutator_leakage"),
                "runtime_seconds": metrics.get("runtime_seconds"),
                "parameter_count": metrics.get("parameter_count"),
            }
        )

    if not rows:
        raise FileNotFoundError("No test_metrics.json files found in the requested result directories")

    rows = sorted(rows, key=lambda row: float(row.get("density_rel_l2", float("inf"))))
    table_path = out_dir / "main_comparison_table.csv"
    with table_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plt.figure(figsize=(6.0, 4.0))
    plt.bar([row["run_label"] for row in rows], [row["density_rel_l2"] for row in rows])
    plt.ylabel("Density relative L2")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "density_rel_l2_bar.png", dpi=180)
    plt.close()

    print(f"Wrote table: {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
