"""Experiment manifest helpers for P4 reproducibility."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from failureops.intervention_registry import REGISTRY_VERSION
from failureops.io_utils import ensure_parent_dir
from failureops.pairing import stable_hash


def write_manifest(
    path: str | Path,
    *,
    config: dict[str, Any],
    command: list[str],
    outputs: dict[str, str],
    row_counts: dict[str, int],
) -> None:
    manifest = {
        "experiment_id": config["experiment_id"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(git_output(["git", "status", "--short"])),
        "command": command,
        "config_hash": stable_hash(config),
        "outputs": outputs,
        "row_counts": row_counts,
        "intervention_registry_version": REGISTRY_VERSION,
    }
    ensure_parent_dir(path)
    with Path(path).open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""

