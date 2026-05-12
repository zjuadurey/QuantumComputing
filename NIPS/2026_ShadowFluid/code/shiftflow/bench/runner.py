"""Unified training and evaluation runner."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch import nn
import yaml

from shiftflow import core_v0
from shiftflow.bench.config import load_config, merge_config_dicts
from shiftflow.bench.data import SchrodingerFlowDataset, build_dataloaders, save_generated_dataset
from shiftflow.bench.metrics import build_metric_context, compute_metrics
from shiftflow.bench.models import build_model


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _density_lowfreq_from_rho_torch(rho: torch.Tensor, mask_flat: torch.Tensor) -> torch.Tensor:
    B, T, _C, N, _N = rho.shape
    coeffs = torch.fft.fft2(rho[:, :, 0], dim=(-2, -1)) / float(N)
    coeffs_flat = coeffs.reshape(B, T, -1)
    kept = coeffs_flat[:, :, mask_flat]
    return torch.cat([kept.real, kept.imag], dim=-1)


def compute_loss(
    pred: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
    metric_ctx: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Common loss decomposition used by all trainable baselines."""
    loss_cfg = config.get("loss", {})
    device = batch["rho_target"].device
    zero = torch.tensor(0.0, device=device)

    rho_loss = torch.nn.functional.mse_loss(pred["rho"], batch["rho_target"])
    if "lowfreq" in pred:
        lowfreq_pred = pred["lowfreq"]
    else:
        lowfreq_pred = _density_lowfreq_from_rho_torch(pred["rho"], metric_ctx["mask_flat"])
    spectral_loss = torch.nn.functional.mse_loss(lowfreq_pred, batch["lowfreq_target"])

    if "energy" in pred:
        energy_loss = torch.nn.functional.mse_loss(pred["energy"], batch["energy_target"])
    else:
        energy_loss = zero

    leakage_loss = pred["leakage"].mean() if "leakage" in pred else zero
    residual_loss = pred["residual_lowfreq_norm"].mean() if "residual_lowfreq_norm" in pred else zero
    selection_entropy_loss = pred["selection_entropy"].mean() if "selection_entropy" in pred else zero
    selection_coverage_loss = (
        pred["selection_coverage_deficit"].mean() if "selection_coverage_deficit" in pred else zero
    )
    total = (
        float(loss_cfg.get("w_rho", 1.0)) * rho_loss
        + float(loss_cfg.get("w_spectral", 0.5)) * spectral_loss
        + float(loss_cfg.get("w_energy", 0.1)) * energy_loss
        + float(loss_cfg.get("w_leakage", 0.0)) * leakage_loss
        + float(loss_cfg.get("w_residual", 0.0)) * residual_loss
        + float(loss_cfg.get("w_selection_entropy", 0.0)) * selection_entropy_loss
        + float(loss_cfg.get("w_selection_coverage", 0.0)) * selection_coverage_loss
    )
    parts = {
        "loss_total": float(total.detach().item()),
        "loss_rho": float(rho_loss.detach().item()),
        "loss_spectral": float(spectral_loss.detach().item()),
        "loss_energy": float(energy_loss.detach().item()),
        "loss_leakage": float(leakage_loss.detach().item()),
        "loss_residual": float(residual_loss.detach().item()),
        "loss_selection_entropy": float(selection_entropy_loss.detach().item()),
        "loss_selection_coverage": float(selection_coverage_loss.detach().item()),
    }
    return total, parts


def _count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    config: dict[str, Any],
    metric_ctx: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    """Train for one epoch, or no-op for analytical baselines."""
    if optimizer is None:
        return {
            "loss_total": 0.0,
            "loss_rho": 0.0,
            "loss_spectral": 0.0,
            "loss_energy": 0.0,
            "loss_leakage": 0.0,
            "loss_residual": 0.0,
            "loss_selection_entropy": 0.0,
            "loss_selection_coverage": 0.0,
        }

    model.train()
    sums: dict[str, float] = {}
    denom = 0
    for batch in dataloader:
        batch = _move_batch_to_device(batch, device)
        optimizer.zero_grad()
        pred = model(batch)
        total, parts = compute_loss(pred, batch, config, metric_ctx)
        total.backward()
        optimizer.step()
        batch_size = int(batch["rho0"].shape[0])
        denom += batch_size
        for key, value in parts.items():
            sums[key] = sums.get(key, 0.0) + float(value) * batch_size
    if denom == 0:
        return {
            key: 0.0
            for key in [
                "loss_total",
                "loss_rho",
                "loss_spectral",
                "loss_energy",
                "loss_leakage",
                "loss_residual",
                "loss_selection_entropy",
                "loss_selection_coverage",
            ]
        }
    return {key: value / denom for key, value in sums.items()}


def evaluate(
    model: nn.Module,
    dataloader,
    config: dict[str, Any],
    metric_ctx: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate one split and return a common metrics dictionary."""
    model.eval()
    totals: dict[str, float] = {}
    curves: dict[str, np.ndarray] = {}
    denom = 0
    runtime_total = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch_to_device(batch, device)
            t0 = time.perf_counter()
            pred = model(batch)
            runtime_total += time.perf_counter() - t0
            _, loss_parts = compute_loss(pred, batch, config, metric_ctx)
            metrics = compute_metrics(pred, batch, config, metric_ctx)
            batch_size = int(batch["rho0"].shape[0])
            denom += batch_size

            merged = dict(loss_parts)
            for key, value in metrics.items():
                if key.endswith("_curve"):
                    arr = np.asarray(value, dtype=np.float64)
                    curves[key] = curves.get(key, np.zeros_like(arr)) + arr * batch_size
                else:
                    merged[key] = float(value)
            for key, value in merged.items():
                totals[key] = totals.get(key, 0.0) + float(value) * batch_size

    out = {key: (value / max(denom, 1)) for key, value in totals.items()}
    for key, value in curves.items():
        out[key] = (value / max(denom, 1)).tolist()
    out["runtime_seconds"] = float(runtime_total)
    out["parameter_count"] = _count_parameters(model)
    return out


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer,
    epoch: int,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    """Persist one training checkpoint."""
    ckpt = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "config": config,
        "metrics": metrics,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def save_results_json_or_csv(path: str, rows: dict[str, Any] | list[dict[str, Any]]) -> None:
    """Save metrics/results in JSON or CSV based on file suffix."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".json":
        out_path.write_text(json.dumps(rows, indent=2))
        return
    if not isinstance(rows, list):
        rows = [rows]
    fieldnames = list(rows[0].keys()) if rows else []
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _selection_score(metrics: dict[str, Any], config: dict[str, Any]) -> float:
    selection = str(config.get("train", {}).get("selection_metric", "loss_total"))
    if selection == "nso_objective":
        leak_weight = float(config.get("loss", {}).get("selection_leakage_weight", 0.1))
        return float(metrics.get("density_rel_l2", metrics.get("loss_total", 0.0))) + leak_weight * float(metrics.get("commutator_leakage", 0.0))
    return float(metrics.get(selection, metrics.get("loss_total", 0.0)))


def _selection_score_for_metric(metrics: dict[str, Any], config: dict[str, Any], metric_name: str | None) -> float:
    if metric_name is None:
        return _selection_score(metrics, config)
    if metric_name == "nso_objective":
        leak_weight = float(config.get("loss", {}).get("selection_leakage_weight", 0.1))
        return float(metrics.get("density_rel_l2", metrics.get("loss_total", 0.0))) + leak_weight * float(metrics.get("commutator_leakage", 0.0))
    return float(metrics.get(metric_name, metrics.get("loss_total", 0.0)))


def _build_extra_selection_loaders(config: dict[str, Any]) -> list[tuple[dict[str, Any], Any]]:
    train_cfg = config.get("train", {})
    specs = train_cfg.get("extra_selection_splits", [])
    if not specs:
        return []
    data_cfg = config["dataset"]
    batch_size = int(config.get("evaluation", {}).get("batch_size", train_cfg.get("batch_size", 4)))
    loaders: list[tuple[dict[str, Any], Any]] = []
    for spec in specs:
        spec = dict(spec)
        dataset = SchrodingerFlowDataset(
            data_path=data_cfg["data_path"],
            manifest_path=data_cfg["manifest_path"],
            split_column=str(spec["split_column"]),
            split_values=[str(x) for x in spec.get("split_values", ["val"])],
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(data_cfg.get("num_workers", 0)),
        )
        loaders.append((spec, loader))
    return loaders


def _build_optimizer(model: nn.Module, config: dict[str, Any]):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return None
    train_cfg = config.get("train", {})
    return torch.optim.Adam(
        params,
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )


def run_training_from_config(config: dict[str, Any], *, config_path: str | None = None) -> dict[str, Any]:
    """Shared training entrypoint used by all baseline scripts."""
    config = dict(config)
    config_ref = str(config_path or config.get("_config_path", "<in-memory-config>"))
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(config.get("seed", 0)))

    if bool(config["dataset"].get("generate_if_missing", False)):
        data_path = Path(config["dataset"]["data_path"])
        manifest_path = Path(config["dataset"]["manifest_path"])
        if not data_path.exists() or not manifest_path.exists():
            save_generated_dataset(config)

    (out_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    device = torch.device(str(config.get("device", "cpu")))
    loaders = build_dataloaders(config)
    extra_selection_loaders = _build_extra_selection_loaders(config)
    model = build_model(config).to(device)
    optimizer = _build_optimizer(model, config)
    metric_ctx = build_metric_context(config, device=device)

    best_state = None
    best_score = float("inf")
    epochs = int(config.get("train", {}).get("epochs", 1))
    history: list[dict[str, Any]] = []

    for epoch in range(epochs):
        train_metrics = train_one_epoch(model, loaders["train"], optimizer, config, metric_ctx, device)
        val_metrics = evaluate(model, loaders["val"], config, metric_ctx, device)
        primary_weight = float(config.get("train", {}).get("selection_primary_weight", 1.0))
        val_score = primary_weight * _selection_score(val_metrics, config)
        row = {"epoch": int(epoch), **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items() if not k.endswith("_curve")}}
        for spec, extra_loader in extra_selection_loaders:
            extra_metrics = evaluate(model, extra_loader, config, metric_ctx, device)
            score = _selection_score_for_metric(extra_metrics, config, spec.get("metric"))
            weight = float(spec.get("weight", 1.0))
            val_score += weight * score
            label = str(spec.get("name", spec["split_column"]))
            for key, value in extra_metrics.items():
                if not key.endswith("_curve"):
                    row[f"sel_{label}_{key}"] = value
        history.append(row)
        if val_score < best_score:
            best_score = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_checkpoint(str(out_dir / "best_checkpoint.pt"), model, optimizer, epoch, config, val_metrics)

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, loaders["test"], config, metric_ctx, device)
    test_metrics["method"] = str(config["model"]["name"])
    test_metrics["config_path"] = config_ref
    test_metrics["output_dir"] = str(out_dir)

    save_results_json_or_csv(str(out_dir / "history.csv"), history if history else [{"epoch": 0}])
    save_results_json_or_csv(str(out_dir / "test_metrics.json"), test_metrics)
    save_results_json_or_csv(str(out_dir / "test_metrics.csv"), {k: v for k, v in test_metrics.items() if not isinstance(v, list)})
    return test_metrics


def run_training(config_path: str) -> dict[str, Any]:
    """Load a config from disk, then train."""
    config = load_config(config_path)
    return run_training_from_config(config, config_path=config_path)


def run_evaluation_from_config(
    config: dict[str, Any],
    *,
    config_path: str | None = None,
    split_column: str | None = None,
    split_values: list[str] | None = None,
    output_name: str = "evaluation.json",
) -> dict[str, Any]:
    """Evaluate an in-memory config on an arbitrary split."""
    config = dict(config)
    config_ref = str(config_path or config.get("_config_path", "<in-memory-config>"))
    if split_column is not None:
        config["dataset"]["split_column"] = split_column
    if split_values is not None:
        config["dataset"]["test_splits"] = split_values
    device = torch.device(str(config.get("device", "cpu")))
    loaders = build_dataloaders(config)
    model = build_model(config).to(device)
    out_dir = Path(config["output_dir"])
    ckpt_path = out_dir / "best_checkpoint.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
    metric_ctx = build_metric_context(config, device=device)
    metrics = evaluate(model, loaders["test"], config, metric_ctx, device)
    metrics["method"] = str(config["model"]["name"])
    metrics["split_column"] = str(config["dataset"].get("split_column", "split_id"))
    metrics["split_values"] = list(config["dataset"].get("test_splits", ["test"]))
    metrics["config_path"] = config_ref
    metrics["output_dir"] = str(out_dir)
    save_results_json_or_csv(str(out_dir / output_name), metrics)
    return metrics


def run_evaluation(config_path: str, *, split_column: str | None = None, split_values: list[str] | None = None, output_name: str = "evaluation.json") -> dict[str, Any]:
    """Load a config from disk, then evaluate."""
    config = load_config(config_path)
    return run_evaluation_from_config(
        config,
        config_path=config_path,
        split_column=split_column,
        split_values=split_values,
        output_name=output_name,
    )


def build_overridden_config(config_path: str, overrides: dict[str, Any]) -> dict[str, Any]:
    """Load a config and apply programmatic overrides."""
    base = load_config(config_path)
    merged = merge_config_dicts(base, overrides)
    merged["_config_path"] = str(config_path)
    return merged
