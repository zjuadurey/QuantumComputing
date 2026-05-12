"""Lightweight YAML config utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _resolve_path(value: str, root: Path) -> str:
    if not value:
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((root / path).resolve())


def _resolve_paths(obj: Any, root: Path) -> Any:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if key.endswith("_path") or key.endswith("_dir"):
                if isinstance(value, str):
                    out[key] = _resolve_path(value, root)
                else:
                    out[key] = _resolve_paths(value, root)
            else:
                out[key] = _resolve_paths(value, root)
        return out
    if isinstance(obj, list):
        return [_resolve_paths(x, root) for x in obj]
    return obj


def load_config(path: str) -> dict[str, Any]:
    """Load a YAML config, supporting optional `inherits`."""
    cfg_path = Path(path).resolve()
    raw = yaml.safe_load(cfg_path.read_text())
    if raw is None:
        raw = {}

    inherits = raw.pop("inherits", None)
    if inherits:
        parent = load_config(str((cfg_path.parent / inherits).resolve()))
        raw = _deep_update(parent, raw)

    raw = _resolve_paths(raw, cfg_path.parent)
    raw["_config_path"] = str(cfg_path)
    return raw


def merge_config_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Public deep-merge helper for programmatic experiment overrides."""
    return _deep_update(base, override)
