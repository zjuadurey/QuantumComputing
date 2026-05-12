from __future__ import annotations

import argparse
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

from shiftflow.bench.runner import run_evaluation  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate long-rollout stability and plot error-vs-time curves")
    p.add_argument("--config", required=True)
    args = p.parse_args(argv)

    metrics = run_evaluation(args.config, split_column="split_id", split_values=["test"], output_name="long_rollout_metrics.json")
    out_dir = Path(metrics["output_dir"])
    curve_path = out_dir / "long_rollout_curves.png"

    plt.figure(figsize=(5.5, 3.8))
    if "density_rel_l2_curve" in metrics:
        plt.plot(metrics["density_rel_l2_curve"], label="density rel L2", linewidth=2)
    if "lowfreq_spectral_rel_l2_curve" in metrics:
        plt.plot(metrics["lowfreq_spectral_rel_l2_curve"], label="spectral rel L2", linewidth=2)
    plt.xlabel("Rollout step")
    plt.ylabel("Error")
    plt.title("Long-rollout error curve")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(curve_path, dpi=180)
    plt.close()

    (out_dir / "long_rollout_curves.json").write_text(json.dumps({
        "density_rel_l2_curve": metrics.get("density_rel_l2_curve", []),
        "lowfreq_spectral_rel_l2_curve": metrics.get("lowfreq_spectral_rel_l2_curve", []),
    }, indent=2))
    print(f"Wrote long-rollout curves: {curve_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
