from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shiftflow.bench.runner import run_evaluation  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate cross-Hamiltonian generalization")
    p.add_argument("--config", required=True)
    p.add_argument("--split-column", default="split_ood_alpha")
    p.add_argument("--split-values", default="test")
    args = p.parse_args(argv)
    split_values = [x.strip() for x in args.split_values.split(",") if x.strip()]
    output_name = f"ood_{args.split_column}.json"
    run_evaluation(args.config, split_column=args.split_column, split_values=split_values, output_name=output_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
