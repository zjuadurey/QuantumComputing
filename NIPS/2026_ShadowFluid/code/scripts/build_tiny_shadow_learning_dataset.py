"""Build a tiny downstream-learning dataset for the ShadowFluid rebuttal probe.

This script is prepared for the response to Reviewer oB6r. It reuses the
existing full-state and reduced ShadowFluid evolution paths to build a compact
supervised dataset:

- input family A: coherence-aware Shadow features Z(t)
- input family B: task-level low-pass density inputs at the same cutoff K0
- target: next-step unresolved high-frequency energy E_HF(t + dt)

The goal is a very small downstream-learning probe rather than a full ML
benchmark.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shiftflow import cases, core_v0, core_v1  # noqa: E402


DEFAULT_REVIEWER = "oB6r"
DEFAULT_ALPHAS = [0.2, 0.6, 1.0]
DEFAULT_TIMES = [0.0, 0.2, 0.4, 0.6]
DEFAULT_SEEDS = list(range(8))
STATE_DTYPE = "complex128"
OBS_DTYPE = "float64"


@dataclass(frozen=True)
class BuildConfig:
    reviewer: str
    out_dir: Path
    nx: int
    K0: float
    qx: int
    qy: int
    dt: float
    R_hops: int
    alphas: list[float]
    times: list[float]
    seeds: list[int]


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def float_str(x: float, digits: int = 6) -> str:
    xf = float(x)
    if abs(xf) < 1e-12:
        return "0"
    if abs(xf) >= 1e-2 and abs(xf) < 1e3:
        return f"{xf:.{digits}f}".rstrip("0").rstrip(".")
    return f"{xf:.{digits}e}"


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_shadow_features(
    b1_k: np.ndarray,
    b2_k: np.ndarray,
    b1_r: np.ndarray,
    b2_r: np.ndarray,
) -> np.ndarray:
    z1 = np.outer(b1_k, b1_r.conj())
    z2 = np.outer(b2_k, b2_r.conj())
    return np.concatenate(
        [
            z1.real.reshape(-1),
            z1.imag.reshape(-1),
            z2.real.reshape(-1),
            z2.imag.reshape(-1),
        ]
    ).astype(np.float64, copy=False)


def next_step_high_frequency_energy(
    b1_next: np.ndarray,
    b2_next: np.ndarray,
    mask: np.ndarray,
    N: int,
) -> float:
    b1_2d = b1_next.reshape(N, N)
    b2_2d = b2_next.reshape(N, N)
    hf_energy = np.sum(np.abs(b1_2d[~mask]) ** 2) + np.sum(np.abs(b2_2d[~mask]) ** 2)
    return float(np.real(hf_energy))


def build_alpha_cache(
    *,
    alpha: float,
    N: int,
    K_flat: np.ndarray,
    qx: int,
    qy: int,
    R_hops: int,
    times_needed: list[float],
) -> dict[str, object]:
    components = core_v1.potential_single(alpha, qx=qx, qy=qy)
    H_dense = core_v1.build_H_dense(N, components)
    eig_vals, eig_vecs = core_v1.eigendecompose(H_dense)
    R_flat = core_v1.build_R_closure((0, 0), components, N, max_hops=R_hops)
    H_K = core_v1.extract_submatrix(H_dense, K_flat)
    H_R = core_v1.extract_submatrix(H_dense, R_flat)

    phase_map = {
        float(t): np.exp(-1j * eig_vals * float(t)).astype(np.complex128, copy=False)
        for t in times_needed
    }
    U_K = {float(t): expm(-1j * H_K * float(t)) for t in times_needed}
    U_R = {float(t): expm(-1j * H_R * float(t)) for t in times_needed}

    return {
        "components": components,
        "H_dense": H_dense,
        "eig_vals": eig_vals,
        "eig_vecs": eig_vecs,
        "R_flat": R_flat,
        "phase_map": phase_map,
        "U_K": U_K,
        "U_R": U_R,
    }


def evolve_full_from_eig_coeffs(
    eig_vecs: np.ndarray,
    eig_coeffs_0: np.ndarray,
    phase_t: np.ndarray,
) -> np.ndarray:
    return eig_vecs @ (eig_coeffs_0 * phase_t)


def generate_dataset(config: BuildConfig) -> tuple[dict[str, np.ndarray], list[dict[str, object]], str]:
    N = 2 ** int(config.nx)
    mask = core_v0.low_freq_mask(N, config.K0)
    K_flat = core_v1.mask_to_flat(mask, N)
    times_needed = sorted({float(t) for t in config.times} | {float(t + config.dt) for t in config.times})

    shadow_features: list[np.ndarray] = []
    lowpass_density_features: list[np.ndarray] = []
    targets: list[float] = []
    manifest_rows: list[dict[str, object]] = []

    for alpha in config.alphas:
        cache = build_alpha_cache(
            alpha=alpha,
            N=N,
            K_flat=K_flat,
            qx=config.qx,
            qy=config.qy,
            R_hops=config.R_hops,
            times_needed=times_needed,
        )
        eig_vecs = cache["eig_vecs"]
        R_flat = cache["R_flat"]
        U_K = cache["U_K"]
        U_R = cache["U_R"]
        phase_map = cache["phase_map"]

        for seed in config.seeds:
            psi1_0, psi2_0, _, _meta = cases.vortex_case(nx=config.nx, seed=seed)
            b1_0 = core_v0.unitary_fft2(psi1_0).reshape(-1)
            b2_0 = core_v0.unitary_fft2(psi2_0).reshape(-1)
            eig_coeffs_1 = eig_vecs.conj().T @ b1_0
            eig_coeffs_2 = eig_vecs.conj().T @ b2_0

            b1_k_0 = b1_0[K_flat]
            b2_k_0 = b2_0[K_flat]
            b1_r_0 = b1_0[R_flat]
            b2_r_0 = b2_0[R_flat]

            for t in config.times:
                phase_t = phase_map[float(t)]
                phase_next = phase_map[float(t + config.dt)]

                b1_full_t = evolve_full_from_eig_coeffs(eig_vecs, eig_coeffs_1, phase_t)
                b2_full_t = evolve_full_from_eig_coeffs(eig_vecs, eig_coeffs_2, phase_t)
                b1_next = evolve_full_from_eig_coeffs(eig_vecs, eig_coeffs_1, phase_next)
                b2_next = evolve_full_from_eig_coeffs(eig_vecs, eig_coeffs_2, phase_next)

                b1_k_t = U_K[float(t)] @ b1_k_0
                b2_k_t = U_K[float(t)] @ b2_k_0
                b1_r_t = U_R[float(t)] @ b1_r_0
                b2_r_t = U_R[float(t)] @ b2_r_0

                shadow_features.append(
                    flatten_shadow_features(b1_k_t, b2_k_t, b1_r_t, b2_r_t)
                )

                psi1_lp, psi2_lp = core_v1.reconstruct_lowpass(
                    b1_full_t[K_flat],
                    b2_full_t[K_flat],
                    K_flat,
                    N,
                )
                rho_lp = core_v0.density_from_components(psi1_lp, psi2_lp)
                lowpass_density_features.append(
                    np.asarray(rho_lp.reshape(-1).real, dtype=np.float64)
                )

                target = next_step_high_frequency_energy(b1_next, b2_next, mask, N)
                targets.append(target)

                manifest_rows.append(
                    {
                        "reviewer": config.reviewer,
                        "initial_condition_family": "Gaussian vortex family via cases.vortex_case(nx, seed)",
                        "seed": int(seed),
                        "alpha": float(alpha),
                        "t": float(t),
                        "dt": float(config.dt),
                        "t_next": float(t + config.dt),
                        "nx": int(config.nx),
                        "N": int(N),
                        "K0": float(config.K0),
                        "qx": int(config.qx),
                        "qy": int(config.qy),
                        "R_hops": int(config.R_hops),
                        "M_K": int(len(K_flat)),
                        "R_size": int(len(R_flat)),
                        "shadow_raw_dim": int(2 * 2 * len(K_flat) * len(R_flat)),
                        "lowpass_density_raw_dim": int(N * N),
                        "target_E_HF_next": float(target),
                    }
                )

    dataset = {
        "shadow_features": np.asarray(shadow_features, dtype=np.float64),
        "lowpass_density_features": np.asarray(lowpass_density_features, dtype=np.float64),
        "targets": np.asarray(targets, dtype=np.float64),
        "alphas": np.asarray([row["alpha"] for row in manifest_rows], dtype=np.float64),
        "times": np.asarray([row["t"] for row in manifest_rows], dtype=np.float64),
        "seeds": np.asarray([row["seed"] for row in manifest_rows], dtype=np.int64),
        "K0": np.asarray([config.K0], dtype=np.float64),
        "dt": np.asarray([config.dt], dtype=np.float64),
        "N": np.asarray([N], dtype=np.int64),
        "qx": np.asarray([config.qx], dtype=np.int64),
        "qy": np.asarray([config.qy], dtype=np.int64),
        "R_hops": np.asarray([config.R_hops], dtype=np.int64),
    }

    feasibility_note = "\n".join(
        [
            "# Feasibility Note",
            "",
            f"Prepared for Reviewer {config.reviewer}.",
            "",
            "## Reused Repository Modules",
            "",
            f"- Full-state evolution: `shiftflow/core_v1.py::evolve_full`, with Hamiltonians from `build_H_dense` and `eigendecompose`.",
            f"- Shadow/reduced representation: `shiftflow/core_v1.py::evolve_galerkin`, `build_R_closure`, and the same rank-1 coherence construction `Z = b_K b_R^*` already used in `actual_Z_error`.",
            f"- Task-level low-pass inputs: `shiftflow/core_v0.py::low_freq_mask` and `shiftflow/core_v1.py::reconstruct_lowpass`, then density via `density_from_components`.",
            f"- Initial conditions: `shiftflow/cases.py::vortex_case`.",
            "",
            "## Actual Feature / Target Definitions",
            "",
            f"- Shadow features: flattened real/imaginary parts of the two-component reduced coherence matrices `Z_1(t)` and `Z_2(t)`, where `Z_c(t) = b_{{K,c}}(t) b_{{R,c}}(t)^*` from the existing reduced ShadowFluid evolution.",
            f"- Low-pass baseline features: the task-level low-pass density field at time `t`, reconstructed from the exact Fourier coefficients restricted to the same cutoff `K0 = {float_str(config.K0)}`. This is used only as an input representation, not as a separate forecasting method.",
            f"- Downstream target: next-step unresolved high-frequency energy `E_HF(t+dt) = sum_{{||k|| > K0}} (|b_1(k, t+dt)|^2 + |b_2(k, t+dt)|^2)`.",
            "",
            "## Setup Actually Used",
            "",
            f"- Benchmark: single-cosine potential with `q = ({config.qx}, {config.qy})`.",
            f"- Grid: `nx = {config.nx}`, `N = {N}`.",
            f"- Cutoff: `K0 = {float_str(config.K0)}`.",
            f"- Time settings: `t in {{{', '.join(float_str(t) for t in config.times)}}}`, `dt = {float_str(config.dt)}`.",
            f"- Coupling strengths: `alpha in {{{', '.join(float_str(a) for a in config.alphas)}}}`.",
            f"- Initial-condition family: Gaussian vortex only, but with deterministic `seed in {{{', '.join(str(s) for s in config.seeds)}}}` to reach enough samples for train sizes 16/32/64 while staying within one family.",
            f"- Numerical precision: states/coherences in `{STATE_DTYPE}`, observables/ML features in `{OBS_DTYPE}`.",
            "",
            "## Expected Runtime / Cost",
            "",
            "- Very small CPU job. The full benchmark uses `N=16`, three alphas, four time points, and eight deterministic seeds for a total of 96 supervised samples.",
            "- The only moderately expensive step is building reduced propagators via `scipy.linalg.expm`, but they are cached per `(alpha, t)`.",
        ]
    )

    return dataset, manifest_rows, feasibility_note


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build a tiny ShadowFluid downstream-learning dataset")
    p.add_argument("--reviewer", default=DEFAULT_REVIEWER)
    p.add_argument("--out-dir", default=str(ROOT / "results" / "tiny_shadow_learning"))
    p.add_argument("--nx", type=int, default=4)
    p.add_argument("--K0", type=float, default=4.0)
    p.add_argument("--qx", type=int, default=1)
    p.add_argument("--qy", type=int, default=0)
    p.add_argument("--dt", type=float, default=0.2)
    p.add_argument("--R-hops", type=int, default=1)
    p.add_argument("--alphas", default=",".join(map(str, DEFAULT_ALPHAS)))
    p.add_argument("--times", default=",".join(map(str, DEFAULT_TIMES)))
    p.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    args = p.parse_args(argv)

    config = BuildConfig(
        reviewer=str(args.reviewer),
        out_dir=Path(args.out_dir),
        nx=int(args.nx),
        K0=float(args.K0),
        qx=int(args.qx),
        qy=int(args.qy),
        dt=float(args.dt),
        R_hops=int(args.R_hops),
        alphas=parse_float_list(args.alphas),
        times=parse_float_list(args.times),
        seeds=parse_int_list(args.seeds),
    )

    dataset, manifest_rows, feasibility_note = generate_dataset(config)

    config.out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = config.out_dir / "dataset.npz"
    manifest_path = config.out_dir / "dataset_manifest.csv"
    note_path = config.out_dir / "feasibility_note.md"

    np.savez_compressed(dataset_path, **dataset)
    write_csv(
        manifest_path,
        manifest_rows,
        [
            "reviewer",
            "initial_condition_family",
            "seed",
            "alpha",
            "t",
            "dt",
            "t_next",
            "nx",
            "N",
            "K0",
            "qx",
            "qy",
            "R_hops",
            "M_K",
            "R_size",
            "shadow_raw_dim",
            "lowpass_density_raw_dim",
            "target_E_HF_next",
        ],
    )
    note_path.write_text(feasibility_note)

    n_samples = int(dataset["targets"].shape[0])
    shadow_dim = int(dataset["shadow_features"].shape[1])
    lowpass_dim = int(dataset["lowpass_density_features"].shape[1])

    print(f"Wrote dataset: {dataset_path}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote note: {note_path}")
    print("Setup:")
    print(
        f"  reviewer={config.reviewer} nx={config.nx} N={2 ** config.nx} "
        f"K0={float_str(config.K0)} q=({config.qx},{config.qy}) dt={float_str(config.dt)} "
        f"alphas={config.alphas} times={config.times} seeds={config.seeds}"
    )
    print(
        f"  n_samples={n_samples} shadow_raw_dim={shadow_dim} "
        f"lowpass_density_raw_dim={lowpass_dim}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
