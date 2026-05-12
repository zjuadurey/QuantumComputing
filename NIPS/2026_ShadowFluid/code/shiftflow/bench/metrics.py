"""Standardized experiment metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from shiftflow import core_v0


def _rel_l2_torch(pred: torch.Tensor, target: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    num = torch.linalg.norm((pred - target).reshape(pred.shape[0], -1), dim=1)
    den = torch.linalg.norm(target.reshape(target.shape[0], -1), dim=1).clamp_min(1e-12)
    return num / den


def _density_lowfreq_from_rho_torch(rho: torch.Tensor, mask_flat: torch.Tensor) -> torch.Tensor:
    """Compute masked density FFT coefficients as real/imag features."""
    B, T, _C, N, _N = rho.shape
    rho2 = rho[:, :, 0]
    coeffs = torch.fft.fft2(rho2, dim=(-2, -1)) / float(N)
    coeffs_flat = coeffs.reshape(B, T, -1)
    kept = coeffs_flat[:, :, mask_flat]
    return torch.cat([kept.real, kept.imag], dim=-1)


def build_metric_context(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    N = 2 ** int(config["dataset"]["nx"])
    K0 = float(config["dataset"]["K0"])
    mask = core_v0.low_freq_mask(N, K0).reshape(-1)
    return {
        "N": N,
        "K0": K0,
        "mask_flat": torch.from_numpy(mask.astype(np.bool_)).to(device),
    }


def compute_metrics(
    pred: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
    metric_ctx: dict[str, Any],
) -> dict[str, Any]:
    """Compute scalar and timewise metrics with a common schema."""
    rho_pred = pred["rho"]
    rho_true = batch["rho_target"]
    B, T = int(rho_pred.shape[0]), int(rho_pred.shape[1])

    density_err_time = []
    for t_idx in range(T):
        density_err_time.append(
            _rel_l2_torch(rho_pred[:, t_idx], rho_true[:, t_idx], dims=(1, 2, 3)).mean().item()
        )
    density_err = float(np.mean(density_err_time))

    if "lowfreq" in pred:
        lowfreq_pred = pred["lowfreq"]
    else:
        lowfreq_pred = _density_lowfreq_from_rho_torch(rho_pred, metric_ctx["mask_flat"])
    lowfreq_true = batch["lowfreq_target"]
    spectral_err_time = []
    for t_idx in range(T):
        rel = _rel_l2_torch(
            lowfreq_pred[:, t_idx],
            lowfreq_true[:, t_idx],
            dims=(1,),
        )
        spectral_err_time.append(rel.mean().item())
    spectral_err = float(np.mean(spectral_err_time))

    if "energy" in pred:
        energy_pred = pred["energy"]
        energy_true = batch["energy_target"]
        energy_rel = _rel_l2_torch(energy_pred, energy_true, dims=(1,)).mean().item()
    else:
        energy_rel = float("nan")

    metrics: dict[str, Any] = {
        "density_rel_l2": density_err,
        "lowfreq_spectral_rel_l2": spectral_err,
        "lowpass_energy_rel_l2": float(energy_rel),
        "density_rel_l2_curve": [float(x) for x in density_err_time],
        "lowfreq_spectral_rel_l2_curve": [float(x) for x in spectral_err_time],
    }

    if "leakage" in pred:
        leak = pred["leakage"]
        metrics["commutator_leakage"] = float(torch.mean(leak).item())
        density_per_sample = _rel_l2_torch(
            rho_pred.reshape(B, -1),
            rho_true.reshape(B, -1),
            dims=(1,),
        ).detach().cpu().numpy()
        leak_np = leak.detach().reshape(B, -1).mean(dim=1).cpu().numpy()
        if len(leak_np) >= 2 and np.std(leak_np) > 1e-12 and np.std(density_per_sample) > 1e-12:
            corr = np.corrcoef(leak_np, density_per_sample)[0, 1]
        else:
            corr = float("nan")
        metrics["leakage_error_correlation"] = float(corr)
    else:
        metrics["commutator_leakage"] = float("nan")
        metrics["leakage_error_correlation"] = float("nan")

    if "latent" in pred:
        latent = pred["latent"]
        if latent.ndim >= 3 and latent.shape[1] >= 2:
            diffs = torch.linalg.norm((latent[:, 1:] - latent[:, :-1]).reshape(latent.shape[0], latent.shape[1] - 1, -1), dim=-1)
            metrics["latent_step_norm"] = float(diffs.mean().item())
        else:
            metrics["latent_step_norm"] = float("nan")
    else:
        metrics["latent_step_norm"] = float("nan")

    if "selection_entropy" in pred:
        metrics["selection_entropy"] = float(pred["selection_entropy"].mean().item())
    else:
        metrics["selection_entropy"] = float("nan")

    if "selection_coverage_deficit" in pred:
        metrics["selection_coverage_deficit"] = float(pred["selection_coverage_deficit"].mean().item())
    else:
        metrics["selection_coverage_deficit"] = float("nan")

    if "residual_lowfreq_norm" in pred:
        metrics["residual_lowfreq_norm"] = float(pred["residual_lowfreq_norm"].mean().item())
    else:
        metrics["residual_lowfreq_norm"] = float("nan")

    if "residual_gate_mean" in pred:
        metrics["residual_gate_mean"] = float(pred["residual_gate_mean"].mean().item())
    else:
        metrics["residual_gate_mean"] = float("nan")

    return metrics
