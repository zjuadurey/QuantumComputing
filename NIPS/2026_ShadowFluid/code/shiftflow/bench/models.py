"""Unified baseline and NSO model implementations."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import torch
from torch import nn

from shiftflow import core_v0
from shiftflow.bench.math_utils import (
    channels_to_psi_components,
    fixed_shadow_rollout,
    param_array_to_components,
)


def _time_features(time_grid: torch.Tensor, batch_size: int) -> torch.Tensor:
    return time_grid.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)


def _broadcast_features(hfeat: torch.Tensor, T: int) -> torch.Tensor:
    return hfeat.unsqueeze(1).expand(-1, T, -1)


def _centered_mode_coords(candidate_flat: torch.Tensor, N: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = candidate_flat.clamp_min(0)
    qx_idx = torch.div(flat, int(N), rounding_mode="floor").float()
    qy_idx = (flat % int(N)).float()
    half = float(int(N) // 2)
    qx = torch.where(qx_idx <= half, qx_idx, qx_idx - float(N))
    qy = torch.where(qy_idx <= half, qy_idx, qy_idx - float(N))
    scale = max(1.0, half)
    qx = qx / scale
    qy = qy / scale
    radius = torch.sqrt(qx.square() + qy.square())
    valid = (candidate_flat >= 0).float()
    return qx * valid, qy * valid, radius * valid


def _normalize_density_rollout(rho: torch.Tensor, rho0: torch.Tensor) -> torch.Tensor:
    """Rescale each predicted density frame to match the initial total mass."""
    target_mass = rho0.sum(dim=(-1, -2, -3), keepdim=True).unsqueeze(1)
    pred_mass = rho.sum(dim=(-1, -2, -3), keepdim=True).clamp_min(1e-12)
    return rho * (target_mass / pred_mass)


def _lowfreq_to_density_torch(
    lowfreq: torch.Tensor,
    mask_indices: torch.Tensor,
    N: int,
) -> torch.Tensor:
    """Reconstruct a real density field from masked FFT coefficients."""
    B, T, F = lowfreq.shape
    M = int(mask_indices.numel())
    real = lowfreq[..., :M]
    imag = lowfreq[..., M:]
    coeffs = torch.complex(real, imag)
    coeffs_flat = torch.zeros(B, T, int(N) * int(N), dtype=coeffs.dtype, device=lowfreq.device)
    coeffs_flat[..., mask_indices] = coeffs
    rho = torch.fft.ifft2(coeffs_flat.view(B, T, int(N), int(N)), dim=(-2, -1)).real * float(N)
    return rho.unsqueeze(2)


def _soft_dictionary_leakage(
    weights: torch.Tensor,
    mask: torch.Tensor,
    outside_pool_leakage: torch.Tensor,
    pair_coupling_sq: torch.Tensor,
) -> torch.Tensor:
    """Set-dependent differentiable leakage surrogate over a candidate pool.

    `weights` has shape (B, L, P). We turn it into a soft coverage over pool
    entries, then penalize selected references whose bra-side Hamiltonian
    coupling points to uncovered pool modes or outside the pool entirely.
    """
    valid = mask.clamp(0.0, 1.0)
    coverage = 1.0 - torch.prod(1.0 - weights * valid[:, None, :], dim=1)
    coverage = coverage * valid
    uncovered = (1.0 - coverage) * valid
    missed_inside = torch.einsum("bq,bqr->br", uncovered, pair_coupling_sq)
    leak_sq = outside_pool_leakage * coverage + missed_inside * coverage
    denom = coverage.sum(dim=-1).clamp_min(1e-6)
    return torch.sqrt(leak_sq.sum(dim=-1) / denom + 1e-12)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int = 2) -> None:
        super().__init__()
        dims = [in_dim] + [hidden_dim] * max(depth - 1, 0) + [out_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallConvEncoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, N: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * N * N, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FieldDecoder(nn.Module):
    def __init__(self, in_dim: int, N: int, hidden_dim: int) -> None:
        super().__init__()
        self.N = int(N)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.N * self.N),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.net(z)
        out = torch.nn.functional.softplus(out)
        return out.view(z.shape[0], 1, self.N, self.N)


class EnergyHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class FixedShadowFluidModel(nn.Module):
    """Analytical fixed-dictionary baseline wrapper."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = config.get("model", {})
        self.K0 = float(config["dataset"]["K0"])
        self.reference_mode = str(model_cfg.get("reference_mode", "bfs"))
        self.reference_budget = model_cfg.get("reference_budget")
        self.R_hops = int(model_cfg.get("R_hops", 1))
        self.pool_hops = int(model_cfg.get("pool_hops", 2))
        self.pool_max_candidates = int(model_cfg.get("pool_max_candidates", 8))
        self._rollout_cache: dict[int, dict[str, Any]] = {}

    def _compute_rollout(
        self,
        psi0_channels: np.ndarray,
        potential_params: np.ndarray,
        time_grid: list[float],
    ) -> dict[str, Any]:
        psi1_0, psi2_0 = channels_to_psi_components(psi0_channels)
        components = param_array_to_components(potential_params)
        rollout = fixed_shadow_rollout(
            psi1_0=psi1_0,
            psi2_0=psi2_0,
            components=components,
            time_grid=time_grid,
            K0=self.K0,
            reference_mode=self.reference_mode,
            budget=None if self.reference_budget is None else int(self.reference_budget),
            R_hops=self.R_hops,
            pool_hops=self.pool_hops,
            pool_max_candidates=self.pool_max_candidates,
        )
        return {
            "rho": rollout.rho,
            "lowfreq": rollout.lowfreq,
            "energy": rollout.energy,
            "leakage": float(rollout.leakage),
        }

    def _get_rollout(
        self,
        sample_index: int,
        psi0_channels: np.ndarray,
        potential_params: np.ndarray,
        time_grid: list[float],
    ) -> dict[str, Any]:
        key = int(sample_index)
        cached = self._rollout_cache.get(key)
        if cached is None:
            cached = self._compute_rollout(psi0_channels, potential_params, time_grid)
            self._rollout_cache[key] = cached
        return cached

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        device = batch["rho0"].device
        rho_preds: list[torch.Tensor] = []
        lowfreq_preds: list[torch.Tensor] = []
        energy_preds: list[torch.Tensor] = []
        leakages: list[float] = []
        time_grid = batch["time_grid"][0].detach().cpu().numpy().tolist()

        for idx in range(batch["psi0"].shape[0]):
            sample_index = int(batch.get("sample_index", torch.arange(batch["psi0"].shape[0]))[idx].item())
            rollout = self._get_rollout(
                sample_index=sample_index,
                psi0_channels=batch["psi0"][idx].detach().cpu().numpy(),
                potential_params=batch["potential_params"][idx].detach().cpu().numpy(),
                time_grid=time_grid,
            )
            rho_preds.append(torch.from_numpy(np.asarray(rollout["rho"], dtype=np.float32)))
            lowfreq_preds.append(torch.from_numpy(np.asarray(rollout["lowfreq"], dtype=np.float32)))
            energy_preds.append(torch.from_numpy(np.asarray(rollout["energy"], dtype=np.float32)))
            leakages.append(float(rollout["leakage"]))

        rho = torch.stack(rho_preds, dim=0).to(device=device, dtype=torch.float32)
        lowfreq = torch.stack(lowfreq_preds, dim=0).to(device=device, dtype=torch.float32)
        energy = torch.stack(energy_preds, dim=0).to(device=device, dtype=torch.float32)
        leakage = torch.tensor(leakages, dtype=torch.float32, device=device)
        return {"rho": rho, "lowfreq": lowfreq, "energy": energy, "leakage": leakage}


class ShadowFluidResidualModel(nn.Module):
    """ShadowFluid plus a learned low-frequency residual correction head."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        data_cfg = config["dataset"]
        model_cfg = config["model"]
        self.N = 2 ** int(data_cfg["nx"])
        self.target_K0 = float(data_cfg["K0"])
        self.base_K0 = float(model_cfg.get("base_K0", self.target_K0))
        self.hfeat_dim = int(model_cfg.get("hamiltonian_feature_dim", model_cfg.get("hfeat_dim", 15)))
        self.hidden_dim = int(model_cfg.get("hidden_dim", 192))
        self.context_dim = int(model_cfg.get("context_dim", 16))
        self.residual_scale = float(model_cfg.get("residual_scale", 1.0))
        self.zero_init_residual_heads = bool(model_cfg.get("zero_init_residual_heads", True))
        self.use_leakage_gate = bool(model_cfg.get("use_leakage_gate", False))
        self.leakage_gate_floor = float(model_cfg.get("leakage_gate_floor", 0.0))
        self.leakage_gate_threshold = float(model_cfg.get("leakage_gate_threshold", 0.0))
        self.leakage_gate_sharpness = float(model_cfg.get("leakage_gate_sharpness", 10.0))

        base_cfg = copy.deepcopy(config)
        base_cfg["dataset"]["K0"] = self.base_K0
        base_cfg["model"]["reference_mode"] = str(model_cfg.get("base_reference_mode", "coupling_greedy"))
        base_cfg["model"]["reference_budget"] = model_cfg.get("base_reference_budget", 6)
        base_cfg["model"]["R_hops"] = int(model_cfg.get("base_R_hops", 1))
        base_cfg["model"]["pool_hops"] = int(model_cfg.get("base_pool_hops", data_cfg.get("candidate_pool_hops", 2)))
        base_cfg["model"]["pool_max_candidates"] = int(
            model_cfg.get("base_pool_max_candidates", data_cfg.get("candidate_pool_max", 8))
        )
        self.base_model = FixedShadowFluidModel(base_cfg)

        target_mask = core_v0.low_freq_mask(self.N, self.target_K0).reshape(-1)
        base_mask = core_v0.low_freq_mask(self.N, self.base_K0).reshape(-1)
        target_idx = np.flatnonzero(target_mask).astype(np.int64)
        base_idx = np.flatnonzero(base_mask).astype(np.int64)
        target_pos = {int(flat): pos for pos, flat in enumerate(target_idx.tolist())}
        if not all(int(flat) in target_pos for flat in base_idx.tolist()):
            raise ValueError("base_K0 mask must be a subset of target K0 mask")
        base_to_target = np.asarray([target_pos[int(flat)] for flat in base_idx.tolist()], dtype=np.int64)
        self.target_lowfreq_dim = 2 * int(len(target_idx))
        self.base_lowfreq_dim = 2 * int(len(base_idx))
        self.register_buffer("target_mask_idx", torch.from_numpy(target_idx), persistent=False)
        self.register_buffer("base_to_target_idx", torch.from_numpy(base_to_target), persistent=False)

        if self.context_dim > 0:
            self.context_encoder = SmallConvEncoder(in_channels=2, latent_dim=self.context_dim, N=self.N)
        else:
            self.context_encoder = None

        trunk_in = self.target_lowfreq_dim + self.hfeat_dim + self.context_dim + 3
        self.trunk = MLP(trunk_in, self.hidden_dim, self.hidden_dim, depth=3)
        self.delta_lowfreq_head = nn.Linear(self.hidden_dim, self.target_lowfreq_dim)
        self.delta_energy_head = nn.Linear(self.hidden_dim, 1)
        if self.zero_init_residual_heads:
            nn.init.zeros_(self.delta_lowfreq_head.weight)
            nn.init.zeros_(self.delta_lowfreq_head.bias)
            nn.init.zeros_(self.delta_energy_head.weight)
            nn.init.zeros_(self.delta_energy_head.bias)

    def _expand_base_lowfreq(self, base_lowfreq: torch.Tensor) -> torch.Tensor:
        if self.base_lowfreq_dim == self.target_lowfreq_dim:
            return base_lowfreq
        B, T, _F = base_lowfreq.shape
        out = torch.zeros(B, T, self.target_lowfreq_dim, dtype=base_lowfreq.dtype, device=base_lowfreq.device)
        M_base = self.base_lowfreq_dim // 2
        out[..., self.base_to_target_idx] = base_lowfreq[..., :M_base]
        out[..., self.base_to_target_idx + (self.target_lowfreq_dim // 2)] = base_lowfreq[..., M_base:]
        return out

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        base = self.base_model(batch)
        base_lowfreq = self._expand_base_lowfreq(base["lowfreq"])
        base_energy = base["energy"]
        leakage = base["leakage"]
        hfeat = batch["hamiltonian_features"]
        B, T, _F = base_lowfreq.shape
        times = batch["time_grid"]
        if times.ndim == 1:
            times = times.unsqueeze(0).expand(B, -1)

        if self.context_encoder is not None:
            context_feat = self.context_encoder(torch.cat([batch["rho0"], batch["potential_field"]], dim=1))
        else:
            context_feat = None

        trunk_parts = [
            base_lowfreq,
            _broadcast_features(hfeat, T),
            base_energy.unsqueeze(-1),
            times.unsqueeze(-1),
            leakage.view(B, 1, 1).expand(-1, T, 1),
        ]
        if context_feat is not None:
            trunk_parts.append(_broadcast_features(context_feat, T))
        trunk_in = torch.cat(trunk_parts, dim=-1)
        hidden = self.trunk(trunk_in.reshape(B * T, -1)).view(B, T, -1)
        delta_lowfreq = self.delta_lowfreq_head(hidden) * self.residual_scale
        delta_energy = self.delta_energy_head(hidden).squeeze(-1) * self.residual_scale
        if self.use_leakage_gate:
            gate = torch.sigmoid(
                (leakage.view(B, 1, 1) - self.leakage_gate_threshold) * self.leakage_gate_sharpness
            )
            gate = self.leakage_gate_floor + (1.0 - self.leakage_gate_floor) * gate
            delta_lowfreq = delta_lowfreq * gate
            delta_energy = delta_energy * gate.squeeze(-1)
        else:
            gate = torch.ones(B, 1, 1, dtype=base_lowfreq.dtype, device=base_lowfreq.device)
        lowfreq = base_lowfreq + delta_lowfreq
        rho = _lowfreq_to_density_torch(lowfreq, self.target_mask_idx, self.N)
        rho = torch.clamp(rho, min=0.0)
        rho = _normalize_density_rollout(rho, batch["rho0"])
        energy = torch.clamp(base_energy + delta_energy, min=0.0)
        residual_norm = delta_lowfreq.square().mean(dim=(-1, -2))
        return {
            "rho": rho,
            "lowfreq": lowfreq,
            "energy": energy,
            "leakage": leakage,
            "base_lowfreq": base_lowfreq,
            "base_energy": base_energy,
            "residual_lowfreq_norm": residual_norm,
            "residual_gate_mean": gate.mean(dim=(1, 2)),
        }


class AELatentModel(nn.Module):
    def __init__(self, config: dict[str, Any], mode: str) -> None:
        super().__init__()
        data_cfg = config["dataset"]
        model_cfg = config["model"]
        self.mode = mode
        self.N = 2 ** int(data_cfg["nx"])
        self.latent_dim = int(model_cfg.get("latent_dim", 8))
        self.h_dim = int(model_cfg.get("hidden_dim", 64))
        self.hfeat_dim = int(model_cfg.get("hamiltonian_feature_dim", model_cfg.get("hfeat_dim", 15)))

        self.encoder = SmallConvEncoder(in_channels=2, latent_dim=self.latent_dim, N=self.N)
        if self.mode == "gru":
            self.transition = nn.GRUCell(self.hfeat_dim + 1, self.latent_dim)
        else:
            self.transition = MLP(self.latent_dim + self.hfeat_dim + 1, self.h_dim, self.latent_dim, depth=3)
        self.decoder = FieldDecoder(self.latent_dim + self.hfeat_dim, N=self.N, hidden_dim=self.h_dim)
        self.energy_head = EnergyHead(self.latent_dim + self.hfeat_dim, hidden_dim=self.h_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        rho0 = batch["rho0"]
        pot = batch["potential_field"]
        x = torch.cat([rho0, pot], dim=1)
        hfeat = batch["hamiltonian_features"]
        times = batch["time_grid"]
        B = int(x.shape[0])
        T = int(times.shape[1] if times.ndim == 2 else times.shape[0])
        if times.ndim == 1:
            times = times.unsqueeze(0).expand(B, -1)

        z = self.encoder(x)
        rho_out = []
        energy_out = []
        latents = []
        for t_idx in range(T):
            t_feat = times[:, t_idx : t_idx + 1]
            if self.mode == "gru":
                z = self.transition(torch.cat([hfeat, t_feat], dim=-1), z)
            else:
                z = z + self.transition(torch.cat([z, hfeat, t_feat], dim=-1))
            zdec = torch.cat([z, hfeat], dim=-1)
            rho_out.append(self.decoder(zdec))
            energy_out.append(self.energy_head(zdec))
            latents.append(z)
        return {
            "rho": _normalize_density_rollout(torch.stack(rho_out, dim=1), batch["rho0"]),
            "energy": torch.stack(energy_out, dim=1),
            "latent": torch.stack(latents, dim=1),
        }


class DeepKoopmanModel(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        data_cfg = config["dataset"]
        model_cfg = config["model"]
        self.N = 2 ** int(data_cfg["nx"])
        self.latent_dim = int(model_cfg.get("latent_dim", 8))
        self.h_dim = int(model_cfg.get("hidden_dim", 64))
        self.hfeat_dim = int(model_cfg.get("hamiltonian_feature_dim", model_cfg.get("hfeat_dim", 15)))
        self.encoder = SmallConvEncoder(in_channels=2, latent_dim=self.latent_dim, N=self.N)
        self.hyper = MLP(self.hfeat_dim, self.h_dim, self.latent_dim * self.latent_dim, depth=3)
        self.decoder = FieldDecoder(self.latent_dim + self.hfeat_dim, N=self.N, hidden_dim=self.h_dim)
        self.energy_head = EnergyHead(self.latent_dim + self.hfeat_dim, hidden_dim=self.h_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        rho0 = batch["rho0"]
        pot = batch["potential_field"]
        x = torch.cat([rho0, pot], dim=1)
        hfeat = batch["hamiltonian_features"]
        B = int(x.shape[0])
        times = batch["time_grid"]
        if times.ndim == 1:
            T = int(times.shape[0])
        else:
            T = int(times.shape[1])

        z = self.encoder(x)
        A = self.hyper(hfeat).view(B, self.latent_dim, self.latent_dim) / float(self.latent_dim)
        eye = torch.eye(self.latent_dim, device=z.device).unsqueeze(0).expand(B, -1, -1)
        A = eye + 0.1 * A

        rho_out = []
        energy_out = []
        latents = []
        zt = z
        for _t in range(T):
            zt = torch.einsum("bij,bj->bi", A, zt)
            zdec = torch.cat([zt, hfeat], dim=-1)
            rho_out.append(self.decoder(zdec))
            energy_out.append(self.energy_head(zdec))
            latents.append(zt)
        return {
            "rho": _normalize_density_rollout(torch.stack(rho_out, dim=1), batch["rho0"]),
            "energy": torch.stack(energy_out, dim=1),
            "latent": torch.stack(latents, dim=1),
        }


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / max(1, in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            B,
            self.out_channels,
            H,
            W // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2],
            self.weights,
        )
        return torch.fft.irfft2(out_ft, s=(H, W))


class MinimalFNOModel(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        data_cfg = config["dataset"]
        model_cfg = config["model"]
        self.N = 2 ** int(data_cfg["nx"])
        self.T = len(data_cfg.get("time_grid", [0.2, 0.4, 0.6, 0.8]))
        self.width = int(model_cfg.get("width", 24))
        modes = int(model_cfg.get("modes", min(8, self.N // 2)))
        in_channels = 6
        self.input_proj = nn.Conv2d(in_channels, self.width, kernel_size=1)
        self.spec_layers = nn.ModuleList([SpectralConv2d(self.width, self.width, modes, modes) for _ in range(3)])
        self.ws = nn.ModuleList([nn.Conv2d(self.width, self.width, kernel_size=1) for _ in range(3)])
        self.output_proj = nn.Conv2d(self.width, self.T, kernel_size=1)
        self.energy_head = EnergyHead(self.width, hidden_dim=max(32, self.width))

        x = torch.linspace(0.0, 1.0, self.N)
        X, Y = torch.meshgrid(x, x, indexing="ij")
        self.register_buffer("grid_x", X[None, None, :, :], persistent=False)
        self.register_buffer("grid_y", Y[None, None, :, :], persistent=False)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        B = int(batch["rho0"].shape[0])
        rho0 = batch["rho0"]
        pot = batch["potential_field"]
        alpha = batch["alpha"].view(B, 1, 1, 1).expand(-1, 1, self.N, self.N)
        qx = batch["q"][:, 0].float().view(B, 1, 1, 1).expand(-1, 1, self.N, self.N)
        inp = torch.cat(
            [
                rho0,
                pot,
                self.grid_x.expand(B, -1, -1, -1),
                self.grid_y.expand(B, -1, -1, -1),
                alpha,
                qx,
            ],
            dim=1,
        )
        x = self.input_proj(inp)
        for spec, w in zip(self.spec_layers, self.ws):
            x = torch.relu(spec(x) + w(x))
        rho = torch.nn.functional.softplus(self.output_proj(x)).unsqueeze(2)
        rho = _normalize_density_rollout(rho, batch["rho0"])
        pooled = x.mean(dim=(-1, -2))
        energy = self.energy_head(pooled).unsqueeze(1).expand(-1, self.T)
        return {"rho": rho, "energy": energy}


class DeepONetModel(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        data_cfg = config["dataset"]
        model_cfg = config["model"]
        self.N = 2 ** int(data_cfg["nx"])
        self.time_grid = [float(x) for x in data_cfg.get("time_grid", [0.2, 0.4, 0.6, 0.8])]
        self.branch_width = int(model_cfg.get("branch_width", 128))
        self.trunk_width = int(model_cfg.get("trunk_width", 64))
        self.rank = int(model_cfg.get("rank", 64))
        self.hfeat_dim = int(model_cfg.get("hamiltonian_feature_dim", model_cfg.get("hfeat_dim", 15)))

        branch_in = 2 * self.N * self.N + self.hfeat_dim
        self.branch = MLP(branch_in, self.branch_width, self.rank, depth=3)
        self.trunk = MLP(3, self.trunk_width, self.rank, depth=3)
        self.energy_head = EnergyHead(self.rank + self.hfeat_dim, hidden_dim=max(64, self.rank))

        x = torch.linspace(0.0, 1.0, self.N)
        X, Y = torch.meshgrid(x, x, indexing="ij")
        query = []
        for t in self.time_grid:
            t_plane = torch.full_like(X, float(t))
            query.append(torch.stack([X, Y, t_plane], dim=-1).reshape(-1, 3))
        self.register_buffer("query_points", torch.cat(query, dim=0), persistent=False)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        B = int(batch["rho0"].shape[0])
        rho0 = batch["rho0"].reshape(B, -1)
        pot = batch["potential_field"].reshape(B, -1)
        hfeat = batch["hamiltonian_features"]
        branch = self.branch(torch.cat([rho0, pot, hfeat], dim=-1))
        trunk = self.trunk(self.query_points.to(branch.device))
        out = torch.matmul(branch, trunk.T)
        out = torch.nn.functional.softplus(out).view(B, len(self.time_grid), 1, self.N, self.N)
        out = _normalize_density_rollout(out, batch["rho0"])
        energy = self.energy_head(torch.cat([branch, hfeat], dim=-1)).unsqueeze(1).expand(-1, len(self.time_grid))
        return {"rho": out, "energy": energy}


class NSOModel(nn.Module):
    def __init__(self, config: dict[str, Any], variant: str) -> None:
        super().__init__()
        data_cfg = config["dataset"]
        model_cfg = config["model"]
        self.variant = variant
        self.N = 2 ** int(data_cfg["nx"])
        self.time_grid = [float(x) for x in data_cfg.get("time_grid", [0.2, 0.4, 0.6, 0.8])]
        self.P = int(data_cfg.get("candidate_pool_max", 8))
        self.num_slots = int(model_cfg.get("latent_dim", model_cfg.get("dictionary_size", 4)))
        self.slot_embed_dim = int(model_cfg.get("slot_embed_dim", 16))
        self.hidden_dim = int(model_cfg.get("hidden_dim", 64))
        self.hfeat_dim = int(model_cfg.get("hamiltonian_feature_dim", model_cfg.get("hfeat_dim", 15)))
        self.selection_temperature = float(model_cfg.get("selection_temperature", 1.0))
        self.selection_mode = str(model_cfg.get("selection_mode", "generator"))
        self.context_dim = int(model_cfg.get("context_dim", 0))
        self.decoder_raw_features = bool(model_cfg.get("decoder_raw_features", False))
        K0 = float(data_cfg.get("K0", 4.0))
        self.shadow_feat_dim = 4 * int(np.count_nonzero(core_v0.low_freq_mask(self.N, K0)))
        self.selection_input_dim = self.hfeat_dim + self.context_dim
        self.latent_feature_dim = self.shadow_feat_dim if self.decoder_raw_features else self.slot_embed_dim

        if self.context_dim > 0:
            self.context_encoder = SmallConvEncoder(in_channels=2, latent_dim=self.context_dim, N=self.N)
        else:
            self.context_encoder = None

        if variant == "nso_no_hcond":
            if self.selection_mode == "candidate_attention":
                self.global_slot_queries = nn.Parameter(torch.randn(self.num_slots, self.slot_embed_dim) * 0.02)
                self.global_logits = None
            else:
                self.global_logits = nn.Parameter(torch.zeros(self.num_slots, self.P))
                self.global_slot_queries = None
            self.generator = None
        elif variant == "nso_random_dictionary":
            rand_logits = torch.randn(self.num_slots, self.P)
            self.register_buffer("random_logits", rand_logits, persistent=False)
            self.generator = None
            self.global_logits = None
            self.global_slot_queries = None
        elif variant == "nso_fixed_dictionary_learned_readout":
            self.generator = None
            self.global_logits = None
            self.global_slot_queries = None
        else:
            self.global_logits = None
            self.global_slot_queries = None
            if self.selection_mode == "candidate_attention":
                self.generator = None
            else:
                self.generator = MLP(self.selection_input_dim, self.hidden_dim, self.num_slots * self.P, depth=3)

        candidate_hidden = int(model_cfg.get("candidate_hidden_dim", max(32, self.hidden_dim // 2)))
        if self.selection_mode == "candidate_attention":
            self.candidate_time_encoder = nn.Sequential(
                nn.Linear(self.shadow_feat_dim, candidate_hidden),
                nn.ReLU(),
                nn.Linear(candidate_hidden, candidate_hidden),
                nn.ReLU(),
            )
            meta_dim = 2 * candidate_hidden + 6
            self.candidate_key = MLP(meta_dim, candidate_hidden, self.slot_embed_dim, depth=2)
            if variant != "nso_no_hcond":
                self.slot_query_net = MLP(
                    self.selection_input_dim,
                    self.hidden_dim,
                    self.num_slots * self.slot_embed_dim,
                    depth=2,
                )
            else:
                self.slot_query_net = None
        else:
            self.candidate_time_encoder = None
            self.candidate_key = None
            self.slot_query_net = None

        self.slot_projector = nn.Sequential(
            nn.Linear(self.shadow_feat_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.slot_embed_dim),
        )
        decoder_in_dim = self.num_slots * self.latent_feature_dim + self.hfeat_dim + self.context_dim
        self.decoder = FieldDecoder(decoder_in_dim, self.N, self.hidden_dim)
        self.energy_head = EnergyHead(decoder_in_dim, self.hidden_dim)
        if variant == "nso_unconstrained_latent":
            self.roll_gru = nn.GRUCell(self.hfeat_dim + self.context_dim + 1, self.num_slots * self.latent_feature_dim)
        self.cached_uniform = nn.Parameter(torch.zeros(1), requires_grad=False)

    def _selection_context(
        self,
        batch: dict[str, torch.Tensor],
        context_feat: torch.Tensor | None,
    ) -> torch.Tensor:
        if context_feat is None:
            return batch["hamiltonian_features"]
        return torch.cat([batch["hamiltonian_features"], context_feat], dim=-1)

    def _candidate_keys(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        feats = batch["shadow_pool_features"]
        B, T, P, F = feats.shape
        encoded = self.candidate_time_encoder(feats.reshape(B * T * P, F)).view(B, T, P, -1)
        mean_feat = encoded.mean(dim=1)
        last_feat = encoded[:, -1]
        qx, qy, radius = _centered_mode_coords(batch["shadow_pool_candidate_flat"], self.N)
        meta = torch.stack(
            [
                batch["shadow_pool_leakage"],
                batch["shadow_pool_outside_leakage"],
                qx,
                qy,
                radius,
                batch["shadow_pool_mask"],
            ],
            dim=-1,
        )
        summary = torch.cat([mean_feat, last_feat, meta], dim=-1)
        return self.candidate_key(summary.reshape(B * P, -1)).view(B, P, self.slot_embed_dim)

    def _weights(
        self,
        batch: dict[str, torch.Tensor],
        context_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = int(batch["shadow_pool_features"].shape[0])
        mask = batch["shadow_pool_mask"]
        neg = -1e9 * (1.0 - mask[:, None, :])

        if self.variant == "nso_no_hcond":
            if self.selection_mode == "candidate_attention":
                cand_keys = self._candidate_keys(batch)
                queries = self.global_slot_queries.unsqueeze(0).expand(B, -1, -1)
                logits = torch.einsum("bld,bpd->blp", queries, cand_keys) / math.sqrt(self.slot_embed_dim)
            else:
                logits = self.global_logits.unsqueeze(0).expand(B, -1, -1)
        elif self.variant == "nso_random_dictionary":
            logits = self.random_logits.unsqueeze(0).expand(B, -1, -1)
        elif self.variant == "nso_fixed_dictionary_learned_readout":
            logits = torch.full((B, self.num_slots, self.P), -1e9, device=mask.device, dtype=mask.dtype)
            for slot in range(self.num_slots):
                logits[:, slot, min(slot, self.P - 1)] = 1e9
        else:
            if self.selection_mode == "candidate_attention":
                cand_keys = self._candidate_keys(batch)
                query_inp = self._selection_context(batch, context_feat)
                queries = self.slot_query_net(query_inp).view(B, self.num_slots, self.slot_embed_dim)
                logits = torch.einsum("bld,bpd->blp", queries, cand_keys) / math.sqrt(self.slot_embed_dim)
            else:
                query_inp = self._selection_context(batch, context_feat)
                logits = self.generator(query_inp).view(B, self.num_slots, self.P)
        logits = logits + neg
        temp = max(self.selection_temperature, 1e-3)
        return torch.softmax(logits / temp, dim=-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        feats = batch["shadow_pool_features"]
        hfeat = batch["hamiltonian_features"]
        B, T, _P, _F = feats.shape
        if self.context_encoder is not None:
            context_feat = self.context_encoder(torch.cat([batch["rho0"], batch["potential_field"]], dim=1))
        else:
            context_feat = None
        weights = self._weights(batch, context_feat=context_feat)
        pooled = torch.einsum("blp,btpf->btlf", weights, feats)
        if self.decoder_raw_features:
            flat = pooled.reshape(B, T, -1)
        else:
            slot_emb = self.slot_projector(pooled)
            flat = slot_emb.reshape(B, T, -1)
        leakage = _soft_dictionary_leakage(
            weights=weights,
            mask=batch["shadow_pool_mask"],
            outside_pool_leakage=batch["shadow_pool_outside_leakage"],
            pair_coupling_sq=batch["shadow_pool_pair_coupling_sq"],
        )
        entropy = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(dim=-1).mean(dim=-1)
        valid = batch["shadow_pool_mask"].clamp(0.0, 1.0)
        coverage = 1.0 - torch.prod(1.0 - weights * valid[:, None, :], dim=1)
        coverage = coverage * valid
        desired_coverage = torch.minimum(
            valid.sum(dim=-1),
            torch.full((B,), float(self.num_slots), device=valid.device),
        )
        coverage_deficit = torch.relu(desired_coverage - coverage.sum(dim=-1))

        if self.variant == "nso_unconstrained_latent":
            times = batch["time_grid"]
            if times.ndim == 1:
                times = times.unsqueeze(0).expand(B, -1)
            z = flat[:, 0]
            rho_list = []
            energy_list = []
            latents = []
            for t_idx in range(T):
                t_feat = times[:, t_idx : t_idx + 1]
                roll_inp = [hfeat]
                if context_feat is not None:
                    roll_inp.append(context_feat)
                roll_inp.append(t_feat)
                z = self.roll_gru(torch.cat(roll_inp, dim=-1), z)
                zdec_parts = [z, hfeat]
                if context_feat is not None:
                    zdec_parts.append(context_feat)
                zdec = torch.cat(zdec_parts, dim=-1)
                rho_list.append(self.decoder(zdec))
                energy_list.append(self.energy_head(zdec))
                latents.append(z)
            rho = _normalize_density_rollout(torch.stack(rho_list, dim=1), batch["rho0"])
            energy = torch.stack(energy_list, dim=1)
            latent = torch.stack(latents, dim=1)
        else:
            zdec_parts = [flat, _broadcast_features(hfeat, T)]
            if context_feat is not None:
                zdec_parts.append(_broadcast_features(context_feat, T))
            zdec = torch.cat(zdec_parts, dim=-1)
            rho = _normalize_density_rollout(
                torch.stack([self.decoder(zdec[:, t]) for t in range(T)], dim=1),
                batch["rho0"],
            )
            energy = torch.stack([self.energy_head(zdec[:, t]) for t in range(T)], dim=1)
            latent = flat

        return {
            "rho": rho,
            "energy": energy,
            "latent": latent,
            "leakage": leakage,
            "dictionary_weights": weights,
            "selection_entropy": entropy,
            "selection_coverage_deficit": coverage_deficit,
        }


def build_model(config: dict[str, Any]) -> nn.Module:
    """Model factory used by all train/eval scripts."""
    name = str(config["model"]["name"])
    if name == "fixed_shadowfluid":
        return FixedShadowFluidModel(config)
    if name == "ae_mlp":
        return AELatentModel(config, mode="mlp")
    if name == "ae_gru":
        return AELatentModel(config, mode="gru")
    if name == "deep_koopman":
        return DeepKoopmanModel(config)
    if name == "fno":
        return MinimalFNOModel(config)
    if name == "deeponet":
        return DeepONetModel(config)
    if name == "shadowfluid_residual":
        return ShadowFluidResidualModel(config)
    if name in {
        "nso",
        "nso_no_leakage",
        "nso_no_hcond",
        "nso_unconstrained_latent",
        "nso_random_dictionary",
        "nso_fixed_dictionary_learned_readout",
        "nso_no_leakage_model_selection",
    }:
        return NSOModel(config, variant=name)
    raise ValueError(f"Unknown model name: {name}")
