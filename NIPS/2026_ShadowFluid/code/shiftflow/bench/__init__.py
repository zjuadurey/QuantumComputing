"""Unified experiment scaffold for NeurIPS-style ShadowFluid baselines."""

from shiftflow.bench.config import load_config, merge_config_dicts
from shiftflow.bench.data import SchrodingerFlowDataset, build_dataloaders
from shiftflow.bench.metrics import compute_metrics
from shiftflow.bench.models import build_model
from shiftflow.bench.runner import (
    build_overridden_config,
    compute_loss,
    evaluate,
    run_evaluation,
    run_evaluation_from_config,
    run_training,
    run_training_from_config,
    save_checkpoint,
    save_results_json_or_csv,
    train_one_epoch,
)

__all__ = [
    "SchrodingerFlowDataset",
    "build_dataloaders",
    "build_model",
    "compute_loss",
    "compute_metrics",
    "evaluate",
    "build_overridden_config",
    "load_config",
    "merge_config_dicts",
    "run_evaluation",
    "run_evaluation_from_config",
    "run_training",
    "run_training_from_config",
    "save_checkpoint",
    "save_results_json_or_csv",
    "train_one_epoch",
]
