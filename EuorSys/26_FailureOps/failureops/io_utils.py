"""Small CSV and type helpers for FailureOps P0."""

from __future__ import annotations

import csv
from pathlib import Path


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(
    path: str | Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> None:
    ensure_parent_dir(path)
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_int(value: object, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(float(str(value)))


def parse_float(value: object, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(str(value))


def fmt_float(value: float) -> str:
    return f"{value:.6f}"

