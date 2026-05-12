"""End-to-end smoke tests for the unified experimental pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shiftflow.bench.data import save_generated_dataset  # noqa: E402
from shiftflow.bench.config import load_config  # noqa: E402
from shiftflow.bench.runner import run_training  # noqa: E402


DEFAULT_CONFIGS = [
    "configs/fixed_shadowfluid.yaml",
    "configs/ae_gru.yaml",
    "configs/deep_koopman.yaml",
    "configs/fno.yaml",
    "configs/deeponet.yaml",
    "configs/nso.yaml",
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run a small unified smoke-test suite")
    p.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    args = p.parse_args(argv)

    configs = [str((ROOT / x).resolve()) if not Path(x).is_absolute() else x for x in args.configs.split(",") if x.strip()]
    base_cfg = load_config(configs[0])
    save_generated_dataset(base_cfg)
    for cfg in configs:
        print(f"[smoke] running {cfg}")
        run_training(cfg)
    print("[smoke] all requested configs completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
