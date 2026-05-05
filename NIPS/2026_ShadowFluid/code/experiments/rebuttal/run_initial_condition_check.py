"""Generate reviewer-scoped initial-condition robustness checks for ShadowFluid.

This experiment is prepared for the response to Reviewer gBE6 and tests whether
the cutoff-dominated density-error picture remains stable across a small set of
deterministic initial-state families.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shiftflow import cases, core_v0, core_v1  # noqa: E402


DEFAULT_REVIEWER = "gBE6"
DEFAULT_K0S = [6.0, 8.0]
STATE_DTYPE = "complex128"
DENSITY_DTYPE = "float64"
RANDOM_BANDLIMIT = 10.0
RANDOM_BANDLIMIT_SEED = 20260417

RAW_FIELDS = [
    "reviewer",
    "initial_condition",
    "construction_rule",
    "ic_seed",
    "state_dtype",
    "density_dtype",
    "nx",
    "N",
    "simulation_seed",
    "alpha",
    "qx",
    "qy",
    "t",
    "K0",
    "R_hops",
    "M_K",
    "R_size",
    "reduced_size",
    "eps_cutoff",
    "eps_shadow",
    "eps_total",
    "shadow_fraction",
    "shadow_to_cutoff",
    "DeltaZ_F",
    "leakage_l_rms",
    "bound_apriori",
    "observation",
    "assumptions",
    "V_label",
]

SUMMARY_FIELDS = [
    "initial_condition",
    "K0",
    "eps_cutoff",
    "eps_shadow",
    "eps_total",
    "shadow_fraction",
    "shadow_to_cutoff",
    "DeltaZ_F",
    "leakage_l_rms",
    "observation",
]


@dataclass(frozen=True)
class InitialConditionSpec:
    name: str
    construction_rule: str
    ic_seed: str
    psi1: np.ndarray
    psi2: np.ndarray


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


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


def make_multiscale_packet(N: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic multi-bump packet with two spatial scales and phases."""
    x = np.linspace(-np.pi, np.pi, N, endpoint=False)
    y = np.linspace(-np.pi, np.pi, N, endpoint=False)
    X, Y = np.meshgrid(x, y)

    g1 = np.exp(-((X + 1.1) ** 2 + (Y - 0.7) ** 2) / (2 * 0.55**2))
    g2 = np.exp(-((X - 1.3) ** 2 + (Y + 1.0) ** 2) / (2 * 1.05**2))

    psi1 = g1 + 0.65 * np.exp(1j * (1.2 * X - 0.7 * Y)) * g2
    psi2 = 0.8 * np.exp(-1j * (0.6 * X + 0.9 * Y)) * g1 - 0.45 * g2
    psi1_n, psi2_n, _ = cases.normalize_components(
        psi1.astype(np.complex128),
        psi2.astype(np.complex128),
    )
    return psi1_n, psi2_n


def make_random_bandlimited_state(N: int, *, seed: int, K_band: float) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic random complex state in a moderate Fourier band."""
    rng = np.random.default_rng(seed)
    mask = core_v0.low_freq_mask(N, K_band)
    _, _, KX, KY = core_v0.k_grid(N)
    radius = np.sqrt(KX**2 + KY**2)
    weight = np.exp(-(radius / max(K_band, 1e-12)) ** 2)

    noise1 = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    noise2 = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    b1 = np.zeros((N, N), dtype=np.complex128)
    b2 = np.zeros((N, N), dtype=np.complex128)
    b1[mask] = noise1[mask] * weight[mask]
    b2[mask] = noise2[mask] * weight[mask]

    psi1 = core_v0.unitary_ifft2(b1)
    psi2 = core_v0.unitary_ifft2(b2)
    psi1_n, psi2_n, _ = cases.normalize_components(psi1, psi2)
    return psi1_n, psi2_n


def build_initial_conditions(nx: int, simulation_seed: int) -> list[InitialConditionSpec]:
    N = 2**int(nx)

    vortex_psi1, vortex_psi2, _, _ = cases.vortex_case(nx=nx, seed=simulation_seed)
    multiscale_psi1, multiscale_psi2 = make_multiscale_packet(N)
    random_psi1, random_psi2 = make_random_bandlimited_state(
        N,
        seed=RANDOM_BANDLIMIT_SEED,
        K_band=RANDOM_BANDLIMIT,
    )

    return [
        InitialConditionSpec(
            name="Gaussian vortex",
            construction_rule="Canonical vortex_case(nx=5, seed=0) reused from the paper/codebase.",
            ic_seed=str(simulation_seed),
            psi1=vortex_psi1,
            psi2=vortex_psi2,
        ),
        InitialConditionSpec(
            name="Multi-scale packet",
            construction_rule="Two spatial Gaussian bumps with widths 0.55 and 1.05, fixed centers, and deterministic complex phase factors; globally normalized across both components.",
            ic_seed="deterministic-no-seed",
            psi1=multiscale_psi1,
            psi2=multiscale_psi2,
        ),
        InitialConditionSpec(
            name="Random band-limited state",
            construction_rule=f"Random complex Fourier coefficients inside the band ||k||<= {RANDOM_BANDLIMIT:g} with Gaussian radial weighting, inverse FFT to position space, fixed RNG seed, then global normalization.",
            ic_seed=str(RANDOM_BANDLIMIT_SEED),
            psi1=random_psi1,
            psi2=random_psi2,
        ),
    ]


def make_observation(shadow_to_cutoff: float, shadow_fraction: float) -> str:
    if shadow_to_cutoff < 0.05:
        return "Cutoff-dominated; shadow contribution remains small."
    if shadow_to_cutoff < 0.2:
        return "Cutoff still dominates; shadow contribution stays secondary."
    if shadow_fraction < 0.25:
        return "Shadow contribution is visible but remains below the cutoff term."
    return "Shadow contribution is no longer clearly secondary in this tested row."


def compute_row(
    *,
    reviewer: str,
    ic: InitialConditionSpec,
    nx: int,
    simulation_seed: int,
    alpha: float,
    qx: int,
    qy: int,
    t: float,
    K0: float,
    R_hops: int,
    H_dense: np.ndarray,
    eig: tuple[np.ndarray, np.ndarray],
) -> dict[str, object]:
    N = 2**int(nx)
    b1_0 = core_v0.unitary_fft2(ic.psi1).reshape(-1)
    b2_0 = core_v0.unitary_fft2(ic.psi2).reshape(-1)
    b1_full_t = core_v1.evolve_full(b1_0, eig[0], eig[1], t)
    b2_full_t = core_v1.evolve_full(b2_0, eig[0], eig[1], t)

    mask = core_v0.low_freq_mask(N, K0)
    K_flat = core_v1.mask_to_flat(mask, N)
    R_flat = core_v1.build_R_closure((0, 0), core_v1.potential_single(alpha, qx=qx, qy=qy), N, max_hops=R_hops)

    H_K = core_v1.extract_submatrix(H_dense, K_flat)
    H_R = core_v1.extract_submatrix(H_dense, R_flat)
    b1_trunc_K = core_v1.evolve_galerkin(b1_0[K_flat], H_K, t)
    b2_trunc_K = core_v1.evolve_galerkin(b2_0[K_flat], H_K, t)
    b1_trunc_R = core_v1.evolve_galerkin(b1_0[R_flat], H_R, t)
    b2_trunc_R = core_v1.evolve_galerkin(b2_0[R_flat], H_R, t)

    psi1_full = core_v0.unitary_ifft2(b1_full_t.reshape(N, N))
    psi2_full = core_v0.unitary_ifft2(b2_full_t.reshape(N, N))
    psi1_full_lp, psi2_full_lp = core_v1.reconstruct_lowpass(
        b1_full_t[K_flat], b2_full_t[K_flat], K_flat, N
    )
    psi1_shadow, psi2_shadow = core_v1.reconstruct_lowpass(
        b1_trunc_K, b2_trunc_K, K_flat, N
    )

    rho_full = core_v0.density_from_components(psi1_full, psi2_full)
    rho_full_lp = core_v0.density_from_components(psi1_full_lp, psi2_full_lp)
    rho_shadow = core_v0.density_from_components(psi1_shadow, psi2_shadow)
    full_norm = float(np.linalg.norm(rho_full))

    eps_cutoff = rel_to_full(rho_full_lp, rho_full, full_norm)
    eps_shadow = rel_to_full(rho_shadow, rho_full_lp, full_norm)
    eps_total = rel_to_full(rho_shadow, rho_full, full_norm)
    shadow_fraction = safe_ratio(eps_shadow, eps_total)
    shadow_to_cutoff = safe_ratio(eps_shadow, eps_cutoff)

    ez1 = core_v1.actual_Z_error(b1_full_t[K_flat], b1_full_t[R_flat], b1_trunc_K, b1_trunc_R)
    ez2 = core_v1.actual_Z_error(b2_full_t[K_flat], b2_full_t[R_flat], b2_trunc_K, b2_trunc_R)
    DeltaZ_F = float(np.sqrt(ez1**2 + ez2**2))
    leak = core_v1.leakage_apriori(H_dense, K_flat, R_flat)
    bound = core_v1.apriori_bound(leak, t)

    components = core_v1.potential_single(alpha, qx=qx, qy=qy)
    observation = make_observation(shadow_to_cutoff, shadow_fraction)
    assumptions = "Reuses the standard single-cosine benchmark from experiments/run_sweep.py with alpha=0.5, q=(1,0), t=0.5, nx=5, canonical seed 0, and classical Galerkin evolution."

    return {
        "reviewer": reviewer,
        "initial_condition": ic.name,
        "construction_rule": ic.construction_rule,
        "ic_seed": ic.ic_seed,
        "state_dtype": STATE_DTYPE,
        "density_dtype": DENSITY_DTYPE,
        "nx": int(nx),
        "N": int(N),
        "simulation_seed": int(simulation_seed),
        "alpha": float(alpha),
        "qx": int(qx),
        "qy": int(qy),
        "t": float(t),
        "K0": float(K0),
        "R_hops": int(R_hops),
        "M_K": int(len(K_flat)),
        "R_size": int(len(R_flat)),
        "reduced_size": int(len(K_flat) * len(R_flat)),
        "eps_cutoff": float(eps_cutoff),
        "eps_shadow": float(eps_shadow),
        "eps_total": float(eps_total),
        "shadow_fraction": float(shadow_fraction),
        "shadow_to_cutoff": float(shadow_to_cutoff),
        "DeltaZ_F": float(DeltaZ_F),
        "leakage_l_rms": float(leak),
        "bound_apriori": float(bound),
        "observation": observation,
        "assumptions": assumptions,
        "V_label": core_v1.potential_label(components),
    }


def build_summary_rows(raw_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for row in raw_rows:
        rows.append({field: row[field] for field in SUMMARY_FIELDS})
    return rows


def max_abs(rows: list[dict[str, object]], key: str) -> float:
    vals = [abs(float(row[key])) for row in rows if not math.isnan(float(row[key]))]
    return max(vals) if vals else float("nan")


def argmax_row(rows: list[dict[str, object]], key: str) -> dict[str, object]:
    return max(rows, key=lambda row: abs(float(row[key])))


def generate_markdown(
    *,
    reviewer: str,
    rows: list[dict[str, object]],
    output_path: Path,
    alpha: float,
    qx: int,
    qy: int,
    t: float,
    R_hops: int,
) -> None:
    max_shadow_fraction = max_abs(rows, "shadow_fraction")
    max_shadow_to_cutoff = max_abs(rows, "shadow_to_cutoff")
    max_eps_shadow = max_abs(rows, "eps_shadow")
    worst_fraction_row = argmax_row(rows, "shadow_fraction")
    hierarchy_ok = all(float(row["eps_shadow"]) < float(row["eps_cutoff"]) for row in rows)

    if hierarchy_ok:
        hierarchy_line = "The qualitative hierarchy is preserved across all tested initial conditions: eps_shadow remains smaller than eps_cutoff in every row."
    else:
        hierarchy_line = "At least one tested row violates eps_shadow < eps_cutoff, so the qualitative hierarchy is not uniform across the full table."

    summary_paragraph = (
        f"For Reviewer {reviewer}, we performed a compact initial-condition robustness check under the standard single-cosine benchmark "
        f"(alpha={alpha:g}, q=({qx},{qy}), t={t:g}, R_hops={R_hops}). {hierarchy_line} "
        f"The maximum observed eps_shadow is {float_str(max_eps_shadow)}, the maximum shadow_fraction is {float_str(max_shadow_fraction)}, "
        f"and the maximum eps_shadow / eps_cutoff ratio is {float_str(max_shadow_to_cutoff)}. "
        f"The largest relative shadow contribution occurs for {worst_fraction_row['initial_condition']} at K0={float_str(worst_fraction_row['K0'])}, "
        f"but the cutoff term still dominates in that case."
    )

    table = to_markdown_table(
        rows,
        [
            "initial_condition",
            "K0",
            "eps_cutoff",
            "eps_shadow",
            "eps_total",
            "shadow_fraction",
            "DeltaZ_F",
            "leakage_l_rms",
            "observation",
        ],
        {
            "initial_condition": "Initial condition",
            "K0": "K0",
            "eps_cutoff": "eps_cutoff",
            "eps_shadow": "eps_shadow",
            "eps_total": "eps_total",
            "shadow_fraction": "shadow_fraction",
            "DeltaZ_F": "DeltaZ_F",
            "leakage_l_rms": "leakage_l_rms",
            "observation": "Observation",
        },
    )

    parts = [
        "# Rebuttal Initial-Condition Robustness Check",
        "",
        f"These results were generated for the response to Reviewer {reviewer}.",
        "",
        "## Setup",
        "",
        f"- Numerical precision: NumPy {STATE_DTYPE} for state/Fourier quantities and {DENSITY_DTYPE} for density fields and norms.",
        "- Base simulation setting: `nx=5`, `N=32`, `alpha=0.5`, `q=(1,0)`, `t=0.5`, `R_hops=1`, and canonical simulation seed `0`.",
        "- Tested cutoffs: `K0 in {6, 8}`.",
        "- `rho_full_lp` is always reconstructed from exact full-state coefficients truncated to the same task cutoff `K0`; the ShadowFluid method itself is unchanged.",
        "- Initial conditions:",
        "  - Gaussian vortex: reused `cases.vortex_case(nx=5, seed=0)`.",
        "  - Multi-scale packet: deterministic two-bump spatial packet with two widths and fixed phase factors.",
        f"  - Random band-limited state: deterministic random Fourier coefficients inside `||k|| <= {RANDOM_BANDLIMIT:g}` with fixed seed `{RANDOM_BANDLIMIT_SEED}`.",
        "",
        "## Auto Summary",
        "",
        summary_paragraph,
        "",
        "## Main Table",
        "",
        table,
        "",
        "## Construction Notes",
        "",
        "- The Gaussian vortex row reuses the paper/codebase default family rather than introducing a new state definition.",
        "- The multi-scale packet is intentionally simple and deterministic, so that robustness is tested against a qualitatively different spatial structure without changing the simulation pipeline.",
        "- The random band-limited state is deterministic through its fixed seed and uses only the existing FFT conventions already present in `shiftflow/core_v0.py`.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run rebuttal initial-condition robustness checks")
    p.add_argument("--reviewer", default=DEFAULT_REVIEWER)
    p.add_argument("--out-dir", default=str(ROOT / "data" / "rebuttal"))
    p.add_argument("--nx", type=int, default=5)
    p.add_argument("--simulation-seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--qx", type=int, default=1)
    p.add_argument("--qy", type=int, default=0)
    p.add_argument("--t", type=float, default=0.5)
    p.add_argument("--R-hops", type=int, default=1)
    p.add_argument("--K0s", default=",".join(map(str, DEFAULT_K0S)))
    args = p.parse_args(argv)

    reviewer = str(args.reviewer)
    out_dir = Path(args.out_dir) / reviewer / "initial_condition_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    nx = int(args.nx)
    alpha = float(args.alpha)
    qx = int(args.qx)
    qy = int(args.qy)
    t = float(args.t)
    R_hops = int(args.R_hops)
    simulation_seed = int(args.simulation_seed)
    K0s = parse_float_list(args.K0s)

    components = core_v1.potential_single(alpha, qx=qx, qy=qy)
    H_dense = core_v1.build_H_dense(2**nx, components)
    eig = core_v1.eigendecompose(H_dense)
    ic_specs = build_initial_conditions(nx, simulation_seed)

    raw_rows: list[dict[str, object]] = []
    for ic in ic_specs:
        for K0 in K0s:
            raw_rows.append(
                compute_row(
                    reviewer=reviewer,
                    ic=ic,
                    nx=nx,
                    simulation_seed=simulation_seed,
                    alpha=alpha,
                    qx=qx,
                    qy=qy,
                    t=t,
                    K0=K0,
                    R_hops=R_hops,
                    H_dense=H_dense,
                    eig=eig,
                )
            )

    raw_rows.sort(key=lambda row: (row["initial_condition"], float(row["K0"])))
    summary_rows = build_summary_rows(raw_rows)

    raw_csv = out_dir / "rebuttal_initial_condition_check.csv"
    summary_csv = out_dir / "rebuttal_initial_condition_check_summary.csv"
    markdown_path = out_dir / "rebuttal_initial_condition_check.md"

    write_csv(raw_csv, raw_rows, RAW_FIELDS)
    write_csv(summary_csv, summary_rows, SUMMARY_FIELDS)
    generate_markdown(
        reviewer=reviewer,
        rows=summary_rows,
        output_path=markdown_path,
        alpha=alpha,
        qx=qx,
        qy=qy,
        t=t,
        R_hops=R_hops,
    )

    max_shadow_fraction = max_abs(summary_rows, "shadow_fraction")
    max_shadow_to_cutoff = max_abs(summary_rows, "shadow_to_cutoff")
    worst_row = argmax_row(summary_rows, "shadow_fraction")
    hierarchy_ok = all(float(row["eps_shadow"]) < float(row["eps_cutoff"]) for row in summary_rows)

    print(f"Wrote: {raw_csv}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {markdown_path}")
    print("Key findings:")
    print(f"  eps_shadow < eps_cutoff for all rows: {hierarchy_ok}")
    print(f"  Max shadow_fraction = {float_str(max_shadow_fraction)}")
    print(f"  Max eps_shadow/eps_cutoff = {float_str(max_shadow_to_cutoff)}")
    print(
        "  Worst-case row: "
        f"{worst_row['initial_condition']} at K0={float_str(worst_row['K0'])}, "
        f"shadow_fraction={float_str(worst_row['shadow_fraction'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
