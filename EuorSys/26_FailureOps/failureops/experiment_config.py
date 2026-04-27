"""P4 YAML experiment configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal envs.
    yaml = None

from failureops.intervention_registry import get_intervention_spec
from failureops.runtime_service import P3_STRESS_CONFIGS
from failureops.workloads import get_workload


DEFAULT_OUTPUTS = {
    "baseline": "data/results/p4_baseline_runs.csv",
    "interventions": "data/results/p4_intervened_runs.csv",
    "validation": "data/results/p4_pairing_validation.csv",
    "effects": "data/results/p4_paired_effects.csv",
    "manifest": "data/results/p4_manifest.json",
}


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        if yaml is not None:
            config = yaml.safe_load(handle)
        else:
            config = parse_simple_yaml(handle.read())
    if not isinstance(config, dict):
        raise ValueError(f"config {path} must contain a YAML mapping")
    return normalize_experiment_config(config)


def normalize_experiment_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "experiment_id": config.get("experiment_id", "p4_default"),
        "workloads": list(config.get("workloads", ["memory_x", "idle_heavy_memory"])),
        "stress_levels": list(config.get("stress_levels", ["low", "high"])),
        "num_seeds": int(config.get("num_seeds", 2)),
        "num_shots_per_seed": int(config.get("num_shots_per_seed", 100)),
        "seed_start": int(config.get("seed_start", 42)),
        "seed_stride": int(config.get("seed_stride", 10000)),
        "interventions": list(config.get("interventions", [])),
        "bootstrap": dict(config.get("bootstrap", {})),
        "outputs": {**DEFAULT_OUTPUTS, **dict(config.get("outputs", {}))},
    }
    if not normalized["interventions"]:
        from failureops.runtime_service import P3_INTERVENTIONS

        normalized["interventions"] = list(P3_INTERVENTIONS)
    normalized["bootstrap"].setdefault("num_resamples", 1000)
    normalized["bootstrap"].setdefault("seed", 2026)
    validate_experiment_config(normalized)
    return normalized


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by configs/p4_default.yaml."""
    root: dict[str, Any] = {}
    current_key = ""
    current_child = ""
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if indent == 0 and line.endswith(":"):
            current_key = line[:-1]
            root[current_key] = {}
            current_child = ""
        elif indent == 0 and ":" in line:
            key, value = line.split(":", 1)
            current_key = key.strip()
            root[current_key] = parse_scalar(value.strip())
            current_child = ""
        elif indent == 2 and line.startswith("- "):
            if not isinstance(root.get(current_key), list):
                root[current_key] = []
            root[current_key].append(parse_scalar(line[2:].strip()))
        elif indent == 2 and ":" in line:
            key, value = line.split(":", 1)
            if not isinstance(root.get(current_key), dict):
                root[current_key] = {}
            current_child = key.strip()
            root[current_key][current_child] = parse_scalar(value.strip())
        elif indent == 4 and line.startswith("- "):
            if not isinstance(root[current_key].get(current_child), list):
                root[current_key][current_child] = []
            root[current_key][current_child].append(parse_scalar(line[2:].strip()))
        else:
            raise ValueError(f"unsupported YAML line: {raw_line}")
    return root


def parse_scalar(value: str) -> Any:
    if value == "":
        return {}
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")


def validate_experiment_config(config: dict[str, Any]) -> None:
    for workload_id in config["workloads"]:
        get_workload(workload_id)
    for stress_level in config["stress_levels"]:
        if stress_level not in P3_STRESS_CONFIGS:
            choices = ", ".join(sorted(P3_STRESS_CONFIGS))
            raise ValueError(f"unknown stress level {stress_level!r}; choose from: {choices}")
    for intervention in config["interventions"]:
        get_intervention_spec(intervention)
    if config["num_seeds"] <= 0:
        raise ValueError("num_seeds must be positive")
    if config["num_shots_per_seed"] <= 0:
        raise ValueError("num_shots_per_seed must be positive")
    if int(config["bootstrap"]["num_resamples"]) <= 0:
        raise ValueError("bootstrap.num_resamples must be positive")
