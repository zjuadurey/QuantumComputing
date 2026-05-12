"""Generate a unified Schrödinger-flow dataset for baseline comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shiftflow.bench.config import load_config  # noqa: E402
from shiftflow.bench.data import save_generated_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate the unified Schrödinger-flow dataset")
    p.add_argument("--config", required=True)
    args = p.parse_args(argv)

    config = load_config(args.config)
    data_path, manifest_path = save_generated_dataset(config)
    print(f"Wrote dataset: {data_path}")
    print(f"Wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
