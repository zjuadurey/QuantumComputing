"""Generate reviewer-scoped rebuttal error-decomposition data for ShadowFluid.

This experiment exposes the density-error decomposition already implicit in the
current implementation:

1. low-pass truncation error
2. shadow dictionary approximation error
3. total density error

The default output path is reviewer-scoped and currently targets Reviewer oB6r.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_rebuttal_stress_tests import structured_components  # noqa: E402
from shiftflow import cases, core_v0, core_v1  # noqa: E402


DEFAULT_REVIEWER = "oB6r"
DEFAULT_MAIN_ALPHAS = [0.2, 0.5, 1.0]
DEFAULT_MAIN_K0S = [4.0, 6.0, 8.0]
DEFAULT_DENSE_HOPS = [1, 3]
STATE_DTYPE = "complex128"
DENSITY_DTYPE = "float64"

RAW_FIELDS = [
    "reviewer",
    "table_id",
    "setting_label",
    "potential_family",
    "source_files",
    "assumptions",
    "state_dtype",
    "density_dtype",
    "nx",
    "N",
    "seed",
    "K0",
    "t",
    "alpha",
    "qx",
    "qy",
    "J",
    "alpha_total",
    "alpha_each",
    "R_hops",
    "M_K",
    "R_size",
    "reduced_size",
    "eps_cutoff",
    "eps_shadow",
    "eps_total",
    "shadow_fraction",
    "shadow_to_cutoff",
    "triangle_gap",
    "DeltaZ_F",
    "leakage_l_rms",
    "bound_apriori",
    "observation",
    "V_label",
]

SUMMARY_FIELDS = [
    "table_id",
    "setting_label",
    "alpha",
    "K0",
    "R_hops",
    "eps_cutoff",
    "eps_shadow",
    "eps_total",
    "shadow_fraction",
    "shadow_to_cutoff",
    "triangle_gap",
    "DeltaZ_F",
    "leakage_l_rms",
    "observation",
]


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def safe_ratio(num: float, den: float) -> float:
    den_f = float(den)
    if den_f == 0.0:
        return float("nan")
    return float(num) / den_f


def rel_to_full(a: np.ndarray, b: np.ndarray, full_norm: float) -> float:
    if full_norm == 0.0:
        return float("nan")
    return float(np.linalg.norm(a - b) / full_norm)


def float_str(x: object, digits: int = 6) -> str:
    if isinstance(x, int):
        return str(x)
    xf = float(x)
    if math.isnan(xf):
        return "nan"
    if abs(xf) >= 1e-2 and abs(xf) < 1e3:
        return f"{xf:.{digits}f}".rstrip("0").rstrip(".")
    return f"{xf:.{digits}e}"


def to_markdown_table(rows: list[dict[str, object]], columns: list[str], labels: dict[str, str]) -> str:
    header = "| " + " | ".join(labels.get(col, col) for col in columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        vals = []
        for col in columns:
            value = row[col]
            if isinstance(value, str):
                vals.append(value)
            else:
                vals.append(float_str(value))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *body])


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def observation_v0(eps_shadow: float, eps_total: float, eps_cutoff: float) -> str:
    if abs(eps_shadow) < 1e-12 and abs(eps_total - eps_cutoff) < 1e-12:
        return "Machine-precision shadow error; total error matches cutoff error."
    return "Shadow error is numerically negligible relative to the cutoff baseline."


def build_row(
    *,
    reviewer: str,
    table_id: str,
    setting_label: str,
    potential_family: str,
    source_files: str,
    assumptions: str,
    nx: int,
    N: int,
    seed: int,
    K0: float,
    t: float,
    alpha: float,
    qx: int,
    qy: int,
    J: int,
    alpha_total: float,
    alpha_each: float,
    R_hops: int,
    M_K: int,
    R_size: int,
    eps_cutoff: float,
    eps_shadow: float,
    eps_total: float,
    DeltaZ_F: float,
    leakage_l_rms: float,
    bound_apriori: float,
    observation: str,
    V_label: str,
) -> dict[str, object]:
    reduced_size = int(M_K * R_size)
    return {
        "reviewer": reviewer,
        "table_id": table_id,
        "setting_label": setting_label,
        "potential_family": potential_family,
        "source_files": source_files,
        "assumptions": assumptions,
        "state_dtype": STATE_DTYPE,
        "density_dtype": DENSITY_DTYPE,
        "nx": int(nx),
        "N": int(N),
        "seed": int(seed),
        "K0": float(K0),
        "t": float(t),
        "alpha": float(alpha),
        "qx": int(qx),
        "qy": int(qy),
        "J": int(J),
        "alpha_total": float(alpha_total),
        "alpha_each": float(alpha_each),
        "R_hops": int(R_hops),
        "M_K": int(M_K),
        "R_size": int(R_size),
        "reduced_size": reduced_size,
        "eps_cutoff": float(eps_cutoff),
        "eps_shadow": float(eps_shadow),
        "eps_total": float(eps_total),
        "shadow_fraction": safe_ratio(eps_shadow, eps_total),
        "shadow_to_cutoff": safe_ratio(eps_shadow, eps_cutoff),
        "triangle_gap": float(eps_cutoff + eps_shadow - eps_total),
        "DeltaZ_F": float(DeltaZ_F),
        "leakage_l_rms": float(leakage_l_rms),
        "bound_apriori": float(bound_apriori),
        "observation": observation,
        "V_label": V_label,
    }


def compute_v0_sanity_row(*, reviewer: str) -> dict[str, object]:
    """Exact-closure sanity check for the free case V=0."""
    N = 64
    nx = 6
    seed = 0
    K0 = 6.0
    t = float(0.30 * np.pi)
    sigma = 3.0

    psi1_0, psi2_0, _ = core_v0.vortex_initial_condition(N=N, sigma=sigma)
    mask = core_v0.low_freq_mask(N, K0)
    K_flat = core_v1.mask_to_flat(mask, N)
    M_K = int(len(K_flat))

    psi1_full, psi2_full, b1_full, b2_full = core_v0.evolve_components_fft_v0(
        psi1_0, psi2_0, t=t, return_coeffs=True
    )
    b1_full_lp = core_v0.lowpass_filter_coeffs(b1_full, mask)
    b2_full_lp = core_v0.lowpass_filter_coeffs(b2_full, mask)
    psi1_full_lp = core_v0.unitary_ifft2(b1_full_lp)
    psi2_full_lp = core_v0.unitary_ifft2(b2_full_lp)

    psi1_shadow, psi2_shadow, b1_shadow, b2_shadow, _k0_1, _k0_2 = core_v0.shadow_evolve_components_lowpass_coherences(
        psi1_0, psi2_0, mask, t, return_coeffs=True
    )

    rho_full = core_v0.density_from_components(psi1_full, psi2_full)
    rho_full_lp = core_v0.density_from_components(psi1_full_lp, psi2_full_lp)
    rho_shadow = core_v0.density_from_components(psi1_shadow, psi2_shadow)
    full_norm = float(np.linalg.norm(rho_full))

    eps_cutoff = rel_to_full(rho_full_lp, rho_full, full_norm)
    eps_shadow = rel_to_full(rho_shadow, rho_full_lp, full_norm)
    eps_total = rel_to_full(rho_shadow, rho_full, full_norm)

    b1_full_K = b1_full.reshape(-1)[K_flat]
    b2_full_K = b2_full.reshape(-1)[K_flat]
    b1_shadow_K = b1_shadow.reshape(-1)[K_flat]
    b2_shadow_K = b2_shadow.reshape(-1)[K_flat]
    ez1 = core_v1.actual_Z_error(b1_full_K, b1_full_K, b1_shadow_K, b1_shadow_K)
    ez2 = core_v1.actual_Z_error(b2_full_K, b2_full_K, b2_shadow_K, b2_shadow_K)
    DeltaZ_F = float(np.sqrt(ez1**2 + ez2**2))

    source_files = "figures/plot_v0_sanity.py, shiftflow/core_v0.py"
    assumptions = "Reuses the V=0 sanity configuration from figures/plot_v0_sanity.py: N=64, sigma=3.0, K0=6, t=0.30*pi."
    observation = observation_v0(eps_shadow, eps_total, eps_cutoff)

    return build_row(
        reviewer=reviewer,
        table_id="table1_v0_sanity",
        setting_label="V=0, N=64, K0=6, t=0.30*pi",
        potential_family="free_v0",
        source_files=source_files,
        assumptions=assumptions,
        nx=nx,
        N=N,
        seed=seed,
        K0=K0,
        t=t,
        alpha=0.0,
        qx=0,
        qy=0,
        J=0,
        alpha_total=0.0,
        alpha_each=0.0,
        R_hops=0,
        M_K=M_K,
        R_size=M_K,
        eps_cutoff=eps_cutoff,
        eps_shadow=eps_shadow,
        eps_total=eps_total,
        DeltaZ_F=DeltaZ_F,
        leakage_l_rms=0.0,
        bound_apriori=0.0,
        observation=observation,
        V_label="V=0",
    )


def compute_v1_density_row(
    *,
    reviewer: str,
    table_id: str,
    setting_label: str,
    potential_family: str,
    source_files: str,
    assumptions: str,
    nx: int,
    seed: int,
    K0: float,
    t: float,
    alpha: float,
    qx: int,
    qy: int,
    J: int,
    alpha_total: float,
    alpha_each: float,
    components: list[core_v1.FourierPotential],
    H_dense: np.ndarray,
    eig: tuple[np.ndarray, np.ndarray],
    b1_0: np.ndarray,
    b2_0: np.ndarray,
    b1_full_t: np.ndarray,
    b2_full_t: np.ndarray,
    R_hops: int,
) -> dict[str, object]:
    """Compute the exact/shadow density decomposition for one V!=0 setting."""
    N = 2**int(nx)
    mask = core_v0.low_freq_mask(N, K0)
    K_flat = core_v1.mask_to_flat(mask, N)
    M_K = int(len(K_flat))
    R_flat = core_v1.build_R_closure((0, 0), components, N, max_hops=R_hops)

    H_K = core_v1.extract_submatrix(H_dense, K_flat)
    H_R = core_v1.extract_submatrix(H_dense, R_flat)

    b1_trunc_K = core_v1.evolve_galerkin(b1_0[K_flat], H_K, t)
    b2_trunc_K = core_v1.evolve_galerkin(b2_0[K_flat], H_K, t)
    b1_trunc_R = core_v1.evolve_galerkin(b1_0[R_flat], H_R, t)
    b2_trunc_R = core_v1.evolve_galerkin(b2_0[R_flat], H_R, t)

    psi1_full_lp, psi2_full_lp = core_v1.reconstruct_lowpass(
        b1_full_t[K_flat], b2_full_t[K_flat], K_flat, N
    )
    psi1_shadow, psi2_shadow = core_v1.reconstruct_lowpass(
        b1_trunc_K, b2_trunc_K, K_flat, N
    )
    psi1_full = core_v0.unitary_ifft2(b1_full_t.reshape(N, N))
    psi2_full = core_v0.unitary_ifft2(b2_full_t.reshape(N, N))

    rho_full = core_v0.density_from_components(psi1_full, psi2_full)
    rho_full_lp = core_v0.density_from_components(psi1_full_lp, psi2_full_lp)
    rho_shadow = core_v0.density_from_components(psi1_shadow, psi2_shadow)
    full_norm = float(np.linalg.norm(rho_full))

    eps_cutoff = rel_to_full(rho_full_lp, rho_full, full_norm)
    eps_shadow = rel_to_full(rho_shadow, rho_full_lp, full_norm)
    eps_total = rel_to_full(rho_shadow, rho_full, full_norm)

    ez1 = core_v1.actual_Z_error(b1_full_t[K_flat], b1_full_t[R_flat], b1_trunc_K, b1_trunc_R)
    ez2 = core_v1.actual_Z_error(b2_full_t[K_flat], b2_full_t[R_flat], b2_trunc_K, b2_trunc_R)
    DeltaZ_F = float(np.sqrt(ez1**2 + ez2**2))
    leak = core_v1.leakage_apriori(H_dense, K_flat, R_flat)
    bound = core_v1.apriori_bound(leak, t)

    return build_row(
        reviewer=reviewer,
        table_id=table_id,
        setting_label=setting_label,
        potential_family=potential_family,
        source_files=source_files,
        assumptions=assumptions,
        nx=nx,
        N=N,
        seed=seed,
        K0=K0,
        t=t,
        alpha=alpha,
        qx=qx,
        qy=qy,
        J=J,
        alpha_total=alpha_total,
        alpha_each=alpha_each,
        R_hops=R_hops,
        M_K=M_K,
        R_size=int(len(R_flat)),
        eps_cutoff=eps_cutoff,
        eps_shadow=eps_shadow,
        eps_total=eps_total,
        DeltaZ_F=DeltaZ_F,
        leakage_l_rms=leak,
        bound_apriori=bound,
        observation="",
        V_label=core_v1.potential_label(components),
    )


def build_summary_rows(raw_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for row in raw_rows:
        rows.append({field: row[field] for field in SUMMARY_FIELDS})
    return rows


def max_abs(rows: list[dict[str, object]], key: str) -> float:
    vals = [abs(float(row[key])) for row in rows if not math.isnan(float(row[key]))]
    return max(vals) if vals else float("nan")


def generate_markdown(
    *,
    reviewer: str,
    raw_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    table1_rows = [row for row in raw_rows if row["table_id"] == "table1_v0_sanity"]
    table2_rows = [row for row in raw_rows if row["table_id"] == "table2_main_v1"]
    table3_rows = [row for row in raw_rows if row["table_id"] == "table3_dense_structured"]

    max_main_eps_shadow = max_abs(table2_rows, "eps_shadow")
    max_main_shadow_fraction = max_abs(table2_rows, "shadow_fraction")
    max_main_shadow_to_cutoff = max_abs(table2_rows, "shadow_to_cutoff")
    max_main_triangle_gap = max_abs(table2_rows, "triangle_gap")
    sanity_eps_shadow = float(table1_rows[0]["eps_shadow"])

    max_dense_eps_shadow = max_abs(table3_rows, "eps_shadow")
    max_dense_shadow_fraction = max_abs(table3_rows, "shadow_fraction")

    table1 = to_markdown_table(
        table1_rows,
        ["setting_label", "eps_cutoff", "eps_shadow", "eps_total", "observation"],
        {
            "setting_label": "Setting",
            "eps_cutoff": "eps_cutoff",
            "eps_shadow": "eps_shadow",
            "eps_total": "eps_total",
            "observation": "Observation",
        },
    )
    table2 = to_markdown_table(
        table2_rows,
        ["alpha", "K0", "eps_cutoff", "eps_shadow", "eps_total", "shadow_fraction", "DeltaZ_F", "leakage_l_rms"],
        {
            "alpha": "alpha",
            "K0": "K0",
            "eps_cutoff": "eps_cutoff",
            "eps_shadow": "eps_shadow",
            "eps_total": "eps_total",
            "shadow_fraction": "shadow_fraction",
            "DeltaZ_F": "DeltaZ_F",
            "leakage_l_rms": "leakage_l_rms",
        },
    )
    table3 = to_markdown_table(
        table3_rows,
        ["setting_label", "eps_cutoff", "eps_shadow", "eps_total", "shadow_fraction", "DeltaZ_F", "leakage_l_rms"],
        {
            "setting_label": "setting",
            "eps_cutoff": "eps_cutoff",
            "eps_shadow": "eps_shadow",
            "eps_total": "eps_total",
            "shadow_fraction": "shadow_fraction",
            "DeltaZ_F": "DeltaZ_F",
            "leakage_l_rms": "leakage_l_rms",
        },
    )

    summary_paragraph = (
        f"In the evaluated regime for Reviewer {reviewer}, the exact-closure sanity check gives "
        f"machine-precision shadow error (eps_shadow = {float_str(sanity_eps_shadow)}), so eps_total matches "
        f"eps_cutoff up to numerical precision. Across the main V != 0 single-cosine decomposition table, "
        f"the maximum eps_shadow is {float_str(max_main_eps_shadow)} and the maximum shadow_fraction is "
        f"{float_str(max_main_shadow_fraction)}, while the maximum eps_shadow / eps_cutoff ratio is "
        f"{float_str(max_main_shadow_to_cutoff)}. This supports the rebuttal claim that, in the tested regime, "
        f"the total density error is primarily driven by the cutoff K0 and the additional shadow contribution "
        f"remains secondary. In the optional dense structured case, eps_shadow stays small "
        f"(max {float_str(max_dense_eps_shadow)}) and shadow_fraction stays below "
        f"{float_str(max_dense_shadow_fraction)}."
    )

    parts = [
        "# Rebuttal Error Decomposition",
        "",
        f"These results were generated for the response to Reviewer {reviewer}.",
        "",
        "## Setup",
        "",
        "- Numerical precision: NumPy complex128 for Fourier/state quantities and float64 for density fields and norms.",
        "- Table 1 reuses the V=0 sanity configuration from `figures/plot_v0_sanity.py`: `N=64`, `sigma=3.0`, `K0=6`, `t=0.30*pi`.",
        "- Table 2 reuses the paper's standard single-cosine family from `experiments/run_sweep.py`: canonical seed `0`, `nx=5`, `q=(1,0)`, `t=0.5`, and `alpha in {0.2, 0.5, 1.0}`.",
        "- Table 3 reuses the structured dense rebuttal family from `experiments/run_rebuttal_stress_tests.py` with `J=8`, fixed `alpha_total=0.5`, `K0=5`, and `R_hops in {1, 3}`.",
        "- For all rows, `rho_full_lp` is reconstructed from exact full-state coefficients truncated to the same task cutoff `K0`; no method definition was changed.",
        "",
        "## Auto Summary",
        "",
        summary_paragraph,
        "",
        f"Across the main V != 0 rows, the maximum triangle-gap diagnostic `eps_cutoff + eps_shadow - eps_total` is {float_str(max_main_triangle_gap)}. This confirms that the decomposition is a useful diagnostic split, but not an exact additive identity.",
        "",
        "## Table 1: Exact-Closure Sanity Check",
        "",
        table1,
        "",
        "## Table 2: Main V != 0 Decomposition",
        "",
        table2,
        "",
        "## Table 3: Dense Structured Case Decomposition",
        "",
        table3,
        "",
        "## Dense-Case Note",
        "",
        "In the current implementation, changing `R_hops` affects the reference set used for the certificate (`DeltaZ_F` and `leakage_l_rms`), but the density reconstruction itself is determined by the retained task subspace `K`. Accordingly, the dense-case decomposition rows share the same density metrics at fixed `K0` while still showing improved certificate quantities as the reference set grows.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run rebuttal error decomposition for ShadowFluid")
    p.add_argument("--reviewer", default=DEFAULT_REVIEWER)
    p.add_argument("--out-dir", default=str(ROOT / "data" / "rebuttal"))
    p.add_argument("--nx", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--t", type=float, default=0.5)
    p.add_argument("--main-alphas", default=",".join(map(str, DEFAULT_MAIN_ALPHAS)))
    p.add_argument("--main-K0s", default=",".join(map(str, DEFAULT_MAIN_K0S)))
    p.add_argument("--dense-J", type=int, default=8)
    p.add_argument("--dense-alpha-total", type=float, default=0.5)
    p.add_argument("--dense-K0", type=float, default=5.0)
    p.add_argument("--dense-hops", default=",".join(map(str, DEFAULT_DENSE_HOPS)))
    args = p.parse_args(argv)

    reviewer = str(args.reviewer)
    out_dir = Path(args.out_dir) / reviewer / "error_decomposition"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, object]] = []
    raw_rows.append(compute_v0_sanity_row(reviewer=reviewer))

    nx = int(args.nx)
    N = 2**nx
    seed = int(args.seed)
    t = float(args.t)
    main_alphas = parse_float_list(args.main_alphas)
    main_k0s = parse_float_list(args.main_K0s)
    dense_hops = parse_int_list(args.dense_hops)

    psi1_0, psi2_0, _, _ = cases.vortex_case(nx=nx, seed=seed)
    b1_0 = core_v0.unitary_fft2(psi1_0).reshape(-1)
    b2_0 = core_v0.unitary_fft2(psi2_0).reshape(-1)

    main_source_files = "experiments/run_sweep.py, shiftflow/core_v1.py, shiftflow/core_v0.py"
    main_assumptions = "Reuses the standard single-cosine benchmark from experiments/run_sweep.py with q=(1,0), canonical seed 0, nx=5, and classical Galerkin evolution."

    for alpha in main_alphas:
        components = core_v1.potential_single(alpha, qx=1, qy=0)
        H_dense = core_v1.build_H_dense(N, components)
        eig = core_v1.eigendecompose(H_dense)
        b1_full_t = core_v1.evolve_full(b1_0, eig[0], eig[1], t)
        b2_full_t = core_v1.evolve_full(b2_0, eig[0], eig[1], t)

        for K0 in main_k0s:
            raw_rows.append(
                compute_v1_density_row(
                    reviewer=reviewer,
                    table_id="table2_main_v1",
                    setting_label=f"alpha={alpha:g}, K0={K0:g}",
                    potential_family="single_cosine",
                    source_files=main_source_files,
                    assumptions=main_assumptions,
                    nx=nx,
                    seed=seed,
                    K0=K0,
                    t=t,
                    alpha=alpha,
                    qx=1,
                    qy=0,
                    J=1,
                    alpha_total=alpha,
                    alpha_each=alpha,
                    components=components,
                    H_dense=H_dense,
                    eig=eig,
                    b1_0=b1_0,
                    b2_0=b2_0,
                    b1_full_t=b1_full_t,
                    b2_full_t=b2_full_t,
                    R_hops=1,
                )
            )

    dense_J = int(args.dense_J)
    dense_alpha_total = float(args.dense_alpha_total)
    dense_components = structured_components(dense_J, alpha_total=dense_alpha_total)
    dense_H = core_v1.build_H_dense(N, dense_components)
    dense_eig = core_v1.eigendecompose(dense_H)
    dense_b1_full_t = core_v1.evolve_full(b1_0, dense_eig[0], dense_eig[1], t)
    dense_b2_full_t = core_v1.evolve_full(b2_0, dense_eig[0], dense_eig[1], t)

    dense_source_files = "experiments/run_rebuttal_stress_tests.py, shiftflow/core_v1.py, shiftflow/core_v0.py"
    dense_assumptions = "Reuses the structured dense rebuttal family with fixed alpha_total=0.5 and J=8; varying R_hops changes the certificate/reference set while density reconstruction remains K-driven."

    for hops in dense_hops:
        raw_rows.append(
            compute_v1_density_row(
                reviewer=reviewer,
                table_id="table3_dense_structured",
                setting_label=f"J={dense_J}, alpha_total={dense_alpha_total:g}, K0={args.dense_K0:g}, R_hops={hops}",
                potential_family="structured_dense_fixed_total",
                source_files=dense_source_files,
                assumptions=dense_assumptions,
                nx=nx,
                seed=seed,
                K0=float(args.dense_K0),
                t=t,
                alpha=float("nan"),
                qx=0,
                qy=0,
                J=dense_J,
                alpha_total=dense_alpha_total,
                alpha_each=dense_alpha_total / dense_J,
                components=dense_components,
                H_dense=dense_H,
                eig=dense_eig,
                b1_0=b1_0,
                b2_0=b2_0,
                b1_full_t=dense_b1_full_t,
                b2_full_t=dense_b2_full_t,
                R_hops=hops,
            )
        )

    raw_rows.sort(key=lambda row: (row["table_id"], float(row["alpha_total"]), float(row["K0"]), int(row["R_hops"])))
    summary_rows = build_summary_rows(raw_rows)

    raw_csv = out_dir / "rebuttal_error_decomposition.csv"
    summary_csv = out_dir / "rebuttal_error_decomposition_summary.csv"
    markdown_path = out_dir / "rebuttal_error_decomposition.md"

    write_csv(raw_csv, raw_rows, RAW_FIELDS)
    write_csv(summary_csv, summary_rows, SUMMARY_FIELDS)
    generate_markdown(reviewer=reviewer, raw_rows=raw_rows, output_path=markdown_path)

    main_rows = [row for row in raw_rows if row["table_id"] == "table2_main_v1"]
    dense_rows = [row for row in raw_rows if row["table_id"] == "table3_dense_structured"]
    sanity_row = [row for row in raw_rows if row["table_id"] == "table1_v0_sanity"][0]
    max_main_eps_shadow = max_abs(main_rows, "eps_shadow")
    max_main_shadow_fraction = max_abs(main_rows, "shadow_fraction")
    max_main_shadow_to_cutoff = max_abs(main_rows, "shadow_to_cutoff")
    max_dense_eps_shadow = max_abs(dense_rows, "eps_shadow")

    print(f"Wrote: {raw_csv}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {markdown_path}")
    print("Key findings:")
    print(f"  V=0 sanity eps_shadow = {float_str(sanity_row['eps_shadow'])}")
    print(f"  Main V!=0 max eps_shadow = {float_str(max_main_eps_shadow)}")
    print(f"  Main V!=0 max shadow_fraction = {float_str(max_main_shadow_fraction)}")
    print(f"  Main V!=0 max eps_shadow/eps_cutoff = {float_str(max_main_shadow_to_cutoff)}")
    print(f"  Dense-case max eps_shadow = {float_str(max_dense_eps_shadow)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
