"""Dataset generation and loading for unified Schrödinger-flow experiments."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import expm
import torch
from torch.utils.data import DataLoader, Dataset

from shiftflow import cases, core_v0, core_v1, selector_learning
from shiftflow.bench.math_utils import (
    build_shadow_pool_trajectory,
    components_cache_key,
    components_to_param_array,
    density_to_channels,
    exact_rollout,
    hamiltonian_feature_vector,
    potential_field_from_components,
    psi_components_to_channels,
)


def _parse_float_list(value: Any) -> list[float]:
    if isinstance(value, str):
        return [float(x.strip()) for x in value.split(",") if x.strip()]
    return [float(x) for x in value]


def _parse_int_list(value: Any) -> list[int]:
    if isinstance(value, str):
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    return [int(x) for x in value]


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _random_split_labels(num_items: int, seed: int) -> list[str]:
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(int(num_items))
    n_train = max(1, int(round(0.7 * num_items)))
    n_val = max(1, int(round(0.15 * num_items)))
    n_test = max(1, num_items - n_train - n_val)
    if n_train + n_val + n_test > num_items:
        n_train = max(1, num_items - n_val - n_test)
    labels = ["train"] * num_items
    for idx in perm[n_train : n_train + n_val]:
        labels[int(idx)] = "val"
    for idx in perm[n_train + n_val :]:
        labels[int(idx)] = "test"
    return labels


@dataclass(frozen=True)
class TrajectorySpec:
    trajectory_id: int
    seed: int
    potential_type: str
    components: list[core_v1.FourierPotential]


def build_trajectory_specs(config: dict[str, Any]) -> list[TrajectorySpec]:
    """Create a deterministic list of trajectory specifications."""
    data_cfg = config["dataset"]
    seeds = _parse_int_list(data_cfg.get("seeds", [0, 1, 2]))
    single_alphas = _parse_float_list(data_cfg.get("single_alphas", [0.1, 0.4, 0.7, 1.0]))
    single_q = tuple(_parse_int_list(data_cfg.get("single_q", [1, 0])))
    multi_count = int(data_cfg.get("num_multi_potentials", 4))
    multi_J_choices = _parse_int_list(data_cfg.get("multi_J_choices", [2, 3]))
    alpha_scale = float(data_cfg.get("multi_alpha_scale", 0.5))
    q_max = int(data_cfg.get("multi_q_max", 3))
    base_seed = int(data_cfg.get("multi_seed", 20260505))

    specs: list[TrajectorySpec] = []
    traj_id = 0
    for alpha in single_alphas:
        comps = core_v1.potential_single(alpha=float(alpha), qx=int(single_q[0]), qy=int(single_q[1]))
        for seed in seeds:
            specs.append(
                TrajectorySpec(
                    trajectory_id=traj_id,
                    seed=int(seed),
                    potential_type="single",
                    components=comps,
                )
            )
            traj_id += 1

    for multi_idx in range(multi_count):
        J = int(multi_J_choices[multi_idx % len(multi_J_choices)])
        comps = core_v1.potential_multi_random(
            J=J,
            alpha_scale=alpha_scale,
            q_max=q_max,
            seed=base_seed + multi_idx,
        )
        for seed in seeds:
            specs.append(
                TrajectorySpec(
                    trajectory_id=traj_id,
                    seed=int(seed),
                    potential_type="multi",
                    components=comps,
                )
            )
            traj_id += 1
    return specs


def generate_dataset_arrays(config: dict[str, Any]) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    """Generate dataset arrays and a sample-level manifest."""
    data_cfg = config["dataset"]
    nx = int(data_cfg["nx"])
    N = 2 ** nx
    K0 = float(data_cfg["K0"])
    time_grid = _parse_float_list(data_cfg.get("time_grid", [0.2, 0.4, 0.6, 0.8]))
    max_modes = int(data_cfg.get("max_modes", 3))
    pool_hops = int(data_cfg.get("candidate_pool_hops", 2))
    pool_max = int(data_cfg.get("candidate_pool_max", 8))
    split_seed = int(data_cfg.get("split_seed", 123))
    exact_solver = str(data_cfg.get("exact_solver", "eigh"))
    verbose = bool(data_cfg.get("verbose", False))

    specs = build_trajectory_specs(config)
    split_labels = _random_split_labels(len(specs), seed=split_seed)

    psi0_list: list[np.ndarray] = []
    rho0_list: list[np.ndarray] = []
    psi_target_list: list[np.ndarray] = []
    rho_target_list: list[np.ndarray] = []
    lowfreq_target_list: list[np.ndarray] = []
    energy_target_list: list[np.ndarray] = []
    potential_field_list: list[np.ndarray] = []
    hfeat_list: list[np.ndarray] = []
    alpha_list: list[float] = []
    q_list: list[np.ndarray] = []
    pparam_list: list[np.ndarray] = []
    pool_feat_list: list[np.ndarray] = []
    pool_mask_list: list[np.ndarray] = []
    pool_leak_list: list[np.ndarray] = []
    pool_flat_list: list[np.ndarray] = []
    pool_outside_leak_list: list[np.ndarray] = []
    pool_pair_sq_list: list[np.ndarray] = []
    manifest: list[dict[str, object]] = []
    potential_cache: dict[tuple[tuple[float, int, int], ...], dict[str, Any]] = {}

    for idx, spec in enumerate(specs):
        cache_key = components_cache_key(spec.components)
        cached = potential_cache.get(cache_key)
        if cached is None:
            if verbose:
                print(
                    f"[data] build cache for potential {cache_key} with exact_solver={exact_solver}",
                    flush=True,
                )
            mask = core_v0.low_freq_mask(N, K0)
            K_flat = core_v1.mask_to_flat(mask, N)
            H_dense = core_v1.build_H_dense(N, spec.components)
            eig_vals = None
            eig_vecs = None
            H_sparse = None
            if exact_solver == "eigh":
                eig_vals, eig_vecs = core_v1.eigendecompose(H_dense)
            else:
                from scipy.sparse import csr_matrix

                H_sparse = csr_matrix(H_dense)
            pool_flat = selector_learning.build_candidate_reference_pool(
                N=N,
                components=spec.components,
                r0=(0, 0),
                max_hops=pool_hops,
                max_candidates=pool_max,
            )
            pool_len = min(len(pool_flat), int(pool_max))
            H_K = core_v1.extract_submatrix(H_dense, K_flat)
            H_R = core_v1.extract_submatrix(H_dense, pool_flat[:pool_len])
            candidate_leakage = np.asarray(
                [
                    core_v1.leakage_apriori(
                        H_dense=H_dense,
                        K_flat=K_flat,
                        R_flat=np.asarray([int(flat)], dtype=int),
                    )
                    for flat in pool_flat[:pool_len]
                ],
                dtype=np.float32,
            )
            pool_set = set(int(x) for x in pool_flat[:pool_len].tolist())
            not_pool = np.asarray(sorted(set(range(N * N)) - pool_set), dtype=int)
            outside_pool_bra_leakage = np.zeros(int(pool_max), dtype=np.float32)
            for j, flat in enumerate(pool_flat[:pool_len]):
                if len(not_pool) > 0:
                    outside_pool_bra_leakage[j] = float(
                        np.sum(np.abs(H_dense[np.ix_(not_pool, np.asarray([int(flat)], dtype=int))]) ** 2)
                    )
            pair_bra_coupling_sq = np.zeros((int(pool_max), int(pool_max)), dtype=np.float32)
            pool_pair_sq = np.abs(H_R) ** 2
            np.fill_diagonal(pool_pair_sq, 0.0)
            pair_bra_coupling_sq[:pool_len, :pool_len] = pool_pair_sq.astype(np.float32, copy=False)
            cached = {
                "mask": mask,
                "K_flat": K_flat,
                "H_dense": H_dense,
                "H_sparse": H_sparse,
                "eig_vals": eig_vals,
                "eig_vecs": eig_vecs,
                "exact_backend": exact_solver,
                "pool_flat": pool_flat,
                "H_K": H_K,
                "H_R": H_R,
                "candidate_leakage": candidate_leakage,
                "outside_pool_bra_leakage": outside_pool_bra_leakage,
                "pair_bra_coupling_sq": pair_bra_coupling_sq,
                "U_K_cache": {float(t): expm(-1j * H_K * float(t)) for t in time_grid},
                "U_R_cache": {float(t): expm(-1j * H_R * float(t)) for t in time_grid},
            }
            potential_cache[cache_key] = cached

        if verbose:
            print(
                f"[data] rollout sample {idx + 1}/{len(specs)} "
                f"(trajectory_id={spec.trajectory_id}, seed={spec.seed}, type={spec.potential_type})",
                flush=True,
            )
        psi1_0, psi2_0, _stacked, meta = cases.vortex_case(nx=nx, seed=spec.seed)
        exact = exact_rollout(
            psi1_0=psi1_0,
            psi2_0=psi2_0,
            components=spec.components,
            time_grid=time_grid,
            K0=K0,
            precomputed=cached,
        )
        shadow_pool = build_shadow_pool_trajectory(
            psi1_0=psi1_0,
            psi2_0=psi2_0,
            components=spec.components,
            time_grid=time_grid,
            K0=K0,
            max_hops=pool_hops,
            max_candidates=pool_max,
            precomputed=cached,
        )

        potential_field = potential_field_from_components(N, spec.components)
        params = components_to_param_array(spec.components, max_modes=max_modes)
        hfeat = hamiltonian_feature_vector(spec.components, max_modes=max_modes)
        primary = spec.components[0] if spec.components else core_v1.FourierPotential(alpha=0.0, qx=0, qy=0)
        alpha_total = float(sum(abs(float(c.alpha)) for c in spec.components))
        q_primary = np.asarray([int(primary.qx), int(primary.qy)], dtype=np.int64)

        psi0_list.append(psi_components_to_channels(psi1_0, psi2_0))
        rho0_list.append(density_to_channels(np.abs(psi1_0) ** 2 + np.abs(psi2_0) ** 2))
        psi_target_list.append(exact.psi_channels)
        rho_target_list.append(exact.rho)
        lowfreq_target_list.append(exact.lowfreq)
        energy_target_list.append(exact.energy)
        potential_field_list.append(potential_field[None, :, :].astype(np.float32))
        hfeat_list.append(hfeat.astype(np.float32))
        alpha_list.append(alpha_total)
        q_list.append(q_primary)
        pparam_list.append(params)
        pool_feat_list.append(shadow_pool.features.astype(np.float32))
        pool_mask_list.append(shadow_pool.candidate_mask.astype(np.float32))
        pool_leak_list.append(shadow_pool.candidate_leakage.astype(np.float32))
        pool_flat_list.append(shadow_pool.candidate_flat.astype(np.int64))
        pool_outside_leak_list.append(shadow_pool.outside_pool_bra_leakage.astype(np.float32))
        pool_pair_sq_list.append(shadow_pool.pair_bra_coupling_sq.astype(np.float32))

        split_alpha = "test" if alpha_total >= float(data_cfg.get("ood_alpha_threshold", 0.8)) else "train"
        split_structure = "test" if spec.potential_type != "single" else "train"
        if split_alpha == "train" and split_labels[idx] == "val":
            split_alpha = "val"
        if split_structure == "train" and split_labels[idx] == "val":
            split_structure = "val"

        manifest.append(
            {
                "sample_index": int(idx),
                "trajectory_id": int(spec.trajectory_id),
                "seed": int(spec.seed),
                "potential_type": spec.potential_type,
                "num_components": int(len(spec.components)),
                "alpha_total": alpha_total,
                "alpha_primary": float(primary.alpha),
                "qx_primary": int(primary.qx),
                "qy_primary": int(primary.qy),
                "split_id": split_labels[idx],
                "split_ood_alpha": split_alpha,
                "split_ood_structure": split_structure,
                "N": int(N),
                "nx": int(nx),
                "K0": float(K0),
                "candidate_pool_hops": int(pool_hops),
                "candidate_pool_max": int(pool_max),
                "shadow_pool_size": int(np.count_nonzero(shadow_pool.candidate_mask)),
                "sigma_eff": float(meta.sigma_eff),
                "shift_x": int(meta.shift_x),
                "shift_y": int(meta.shift_y),
                "noise_eps": float(meta.noise_eps),
            }
        )

    arrays = {
        "psi0": np.stack(psi0_list, axis=0).astype(np.float32),
        "rho0": np.stack(rho0_list, axis=0).astype(np.float32),
        "psi_target": np.stack(psi_target_list, axis=0).astype(np.float32),
        "rho_target": np.stack(rho_target_list, axis=0).astype(np.float32),
        "lowfreq_target": np.stack(lowfreq_target_list, axis=0).astype(np.float32),
        "energy_target": np.stack(energy_target_list, axis=0).astype(np.float32),
        "potential_field": np.stack(potential_field_list, axis=0).astype(np.float32),
        "hamiltonian_features": np.stack(hfeat_list, axis=0).astype(np.float32),
        "alpha": np.asarray(alpha_list, dtype=np.float32),
        "q": np.stack(q_list, axis=0).astype(np.int64),
        "potential_params": np.stack(pparam_list, axis=0).astype(np.float32),
        "time_grid": np.asarray(time_grid, dtype=np.float32),
        "trajectory_id": np.asarray([spec.trajectory_id for spec in specs], dtype=np.int64),
        "shadow_pool_features": np.stack(pool_feat_list, axis=0).astype(np.float32),
        "shadow_pool_mask": np.stack(pool_mask_list, axis=0).astype(np.float32),
        "shadow_pool_leakage": np.stack(pool_leak_list, axis=0).astype(np.float32),
        "shadow_pool_candidate_flat": np.stack(pool_flat_list, axis=0).astype(np.int64),
        "shadow_pool_outside_leakage": np.stack(pool_outside_leak_list, axis=0).astype(np.float32),
        "shadow_pool_pair_coupling_sq": np.stack(pool_pair_sq_list, axis=0).astype(np.float32),
    }
    return arrays, manifest


def save_generated_dataset(config: dict[str, Any]) -> tuple[str, str]:
    """Generate and save a dataset to NPZ + CSV manifest."""
    data_cfg = config["dataset"]
    arrays, manifest = generate_dataset_arrays(config)

    data_path = Path(data_cfg["data_path"])
    manifest_path = Path(data_cfg["manifest_path"])
    data_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data_path, **arrays)
    _write_csv(manifest_path, manifest, fieldnames=list(manifest[0].keys()))
    return str(data_path), str(manifest_path)


class SchrodingerFlowDataset(Dataset):
    """Unified batch dataset used by all baselines."""

    def __init__(
        self,
        data_path: str,
        manifest_path: str,
        *,
        split_column: str,
        split_values: list[str],
    ) -> None:
        self.data_path = str(data_path)
        self.manifest_path = str(manifest_path)
        self.data = np.load(self.data_path)

        with Path(self.manifest_path).open() as f:
            rows = list(csv.DictReader(f))
        keep = [row for row in rows if str(row[split_column]) in {str(x) for x in split_values}]
        self.rows = keep
        self.indices = [int(row["sample_index"]) for row in keep]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        idx = int(self.indices[item])
        row = self.rows[item]
        sample = {
            "psi0": torch.from_numpy(np.asarray(self.data["psi0"][idx], dtype=np.float32)),
            "rho0": torch.from_numpy(np.asarray(self.data["rho0"][idx], dtype=np.float32)),
            "psi_target": torch.from_numpy(np.asarray(self.data["psi_target"][idx], dtype=np.float32)),
            "rho_target": torch.from_numpy(np.asarray(self.data["rho_target"][idx], dtype=np.float32)),
            "lowfreq_target": torch.from_numpy(np.asarray(self.data["lowfreq_target"][idx], dtype=np.float32)),
            "energy_target": torch.from_numpy(np.asarray(self.data["energy_target"][idx], dtype=np.float32)),
            "potential_field": torch.from_numpy(np.asarray(self.data["potential_field"][idx], dtype=np.float32)),
            "hamiltonian_features": torch.from_numpy(np.asarray(self.data["hamiltonian_features"][idx], dtype=np.float32)),
            "alpha": torch.tensor(float(self.data["alpha"][idx]), dtype=torch.float32),
            "q": torch.from_numpy(np.asarray(self.data["q"][idx], dtype=np.int64)),
            "potential_params": torch.from_numpy(np.asarray(self.data["potential_params"][idx], dtype=np.float32)),
            "time_grid": torch.from_numpy(np.asarray(self.data["time_grid"], dtype=np.float32)),
            "trajectory_id": torch.tensor(int(self.data["trajectory_id"][idx]), dtype=torch.int64),
            "shadow_pool_features": torch.from_numpy(np.asarray(self.data["shadow_pool_features"][idx], dtype=np.float32)),
            "shadow_pool_mask": torch.from_numpy(np.asarray(self.data["shadow_pool_mask"][idx], dtype=np.float32)),
            "shadow_pool_leakage": torch.from_numpy(np.asarray(self.data["shadow_pool_leakage"][idx], dtype=np.float32)),
            "shadow_pool_candidate_flat": torch.from_numpy(np.asarray(self.data["shadow_pool_candidate_flat"][idx], dtype=np.int64)),
            "sample_index": torch.tensor(idx, dtype=torch.int64),
        }
        if "shadow_pool_outside_leakage" in self.data:
            sample["shadow_pool_outside_leakage"] = torch.from_numpy(
                np.asarray(self.data["shadow_pool_outside_leakage"][idx], dtype=np.float32)
            )
        else:
            sample["shadow_pool_outside_leakage"] = torch.zeros_like(sample["shadow_pool_leakage"])
        if "shadow_pool_pair_coupling_sq" in self.data:
            sample["shadow_pool_pair_coupling_sq"] = torch.from_numpy(
                np.asarray(self.data["shadow_pool_pair_coupling_sq"][idx], dtype=np.float32)
            )
        else:
            P = int(sample["shadow_pool_leakage"].shape[0])
            sample["shadow_pool_pair_coupling_sq"] = torch.zeros((P, P), dtype=torch.float32)
        sample["split_label"] = row["split_id"]
        return sample


def build_dataloaders(config: dict[str, Any]) -> dict[str, DataLoader]:
    """Construct standardized train/val/test dataloaders."""
    data_cfg = config["dataset"]
    batch_size = int(config.get("train", {}).get("batch_size", 4))
    eval_batch_size = int(config.get("evaluation", {}).get("batch_size", batch_size))
    split_column = str(data_cfg.get("split_column", "split_id"))

    loaders: dict[str, DataLoader] = {}
    for split_name, values, bs, shuffle in [
        ("train", data_cfg.get("train_splits", ["train"]), batch_size, True),
        ("val", data_cfg.get("val_splits", ["val"]), eval_batch_size, False),
        ("test", data_cfg.get("test_splits", ["test"]), eval_batch_size, False),
    ]:
        dataset = SchrodingerFlowDataset(
            data_path=data_cfg["data_path"],
            manifest_path=data_cfg["manifest_path"],
            split_column=split_column,
            split_values=[str(x) for x in values],
        )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=shuffle and len(dataset) > 0,
            num_workers=int(data_cfg.get("num_workers", 0)),
        )
    return loaders
