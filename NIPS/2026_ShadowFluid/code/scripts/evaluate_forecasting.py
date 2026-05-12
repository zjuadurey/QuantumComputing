from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shiftflow.bench.runner import run_evaluation  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate in-distribution forecasting")
    p.add_argument("--config", required=True)
    args = p.parse_args(argv)
    run_evaluation(args.config, split_column="split_id", split_values=["test"], output_name="forecasting_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
