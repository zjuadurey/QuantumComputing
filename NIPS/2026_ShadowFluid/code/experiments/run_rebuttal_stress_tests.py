"""Generate rebuttal stress-test data for structured multi-component potentials.

This script packages the exploratory stress tests used in the rebuttal notes into
reproducible CSV files plus a compact Markdown summary.

These materials are specifically prepared for the response to Reviewer oB6r.

Experiments written by default:
1. Fixed total coupling strength, increasing graph density and closure depth.
2. Dense structured case, time sweep under fixed total coupling.
3. Fixed per-component coupling, increasing graph density.
4. Dense structured case, closure-depth sweep under fixed per-component coupling.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shiftflow import cases, core_v1  # noqa: E402


STRUCTURED_QS: list[tuple[int, int]] = [
    (1, 0),
    (0, 1),
    (1, 1),
    (1, -1),
    (2, 0),
    (0, 2),
    (2, 1),
    (1, 2),
]

DEFAULT_J_LIST = [1, 2, 4, 6, 8]
DEFAULT_HOPS_LIST = [1, 2, 3]
DEFAULT_T_LIST = [0.2, 0.5, 0.8, 1.0]
REVIEWER_LABEL = "Reviewer oB6r"

COMMON_FIELDS = [
    "section",
    "nx",
    "N",
    "seed",
    "K0",
    "t",
    "J",
    "alpha_total",
    "alpha_each",
    "R_hops",
    "M_K",
    "R_size",
    "reduced_size",
    "err_rho_vs_full",
    "err_rho_lp_vs_full",
    "extra_shadow_err",
    "err_E_LP",
    "err_Z_frob",
    "leakage_apriori",
    "bound_apriori",
    "V_label",
]


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def structured_components(
    J: int,
    *,
    alpha_total: float | None = None,
    alpha_each: float | None = None,
) -> list[core_v1.FourierPotential]:
    """Create a deterministic structured potential family for a given J."""
    J_i = int(J)
    if J_i <= 0:
        raise ValueError("J must be positive")
    if J_i > len(STRUCTURED_QS):
        raise ValueError(f"Requested J={J_i}, but only {len(STRUCTURED_QS)} structured components are defined")
    if (alpha_total is None) == (alpha_each is None):
        raise ValueError("Provide exactly one of alpha_total or alpha_each")

    if alpha_total is not None:
        alpha_i = float(alpha_total) / float(J_i)
    else:
        alpha_i = float(alpha_each)

    return [
        core_v1.FourierPotential(alpha=alpha_i, qx=qx, qy=qy)
        for qx, qy in STRUCTURED_QS[:J_i]
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows to CSV, preserving a stable and human-friendly field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COMMON_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def format_float(x: object, digits: int = 6) -> str:
    """Compact formatting for Markdown summary tables."""
    if isinstance(x, int):
        return str(x)
    xf = float(x)
    if abs(xf) >= 1e-2 and abs(xf) < 1e3:
        return f"{xf:.{digits}f}".rstrip("0").rstrip(".")
    return f"{xf:.{digits}e}"


def to_markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    """Render a simple Markdown table."""
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        vals = []
        for col in columns:
            value = row[col]
            if isinstance(value, str):
                vals.append(value)
            else:
                vals.append(format_float(value))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *body])


def result_row(
    section: str,
    *,
    nx: int,
    seed: int,
    K0: float,
    t: float,
    J: int,
    alpha_total: float,
    alpha_each: float,
    R_hops: int,
    result: core_v1.V1Result,
) -> dict[str, object]:
    """Standardize result rows across rebuttal experiments."""
    extra_shadow_err = float(result.err_rho_vs_full - result.err_rho_lp_vs_full)
    return {
        "section": section,
        "nx": int(nx),
        "N": int(result.N),
        "seed": int(seed),
        "K0": float(K0),
        "t": float(t),
        "J": int(J),
        "alpha_total": float(alpha_total),
        "alpha_each": float(alpha_each),
        "R_hops": int(R_hops),
        "M_K": int(result.M_K),
        "R_size": int(result.R_size),
        "reduced_size": int(result.M_K * result.R_size),
        "err_rho_vs_full": float(result.err_rho_vs_full),
        "err_rho_lp_vs_full": float(result.err_rho_lp_vs_full),
        "extra_shadow_err": extra_shadow_err,
        "err_E_LP": float(result.err_E_LP),
        "err_Z_frob": float(result.err_Z_frob),
        "leakage_apriori": float(result.leakage_apriori),
        "bound_apriori": float(result.bound_apriori),
        "V_label": str(result.V_label),
    }


def run_fixed_alpha_total(
    *,
    nx: int,
    seed: int,
    K0: float,
    t: float,
    J_list: list[int],
    hops_list: list[int],
    alpha_total: float,
) -> list[dict[str, object]]:
    """Sweep J and closure depth while keeping total coupling fixed."""
    N = 2**int(nx)
    psi1_0, psi2_0, _, _ = cases.vortex_case(nx=nx, seed=seed)
    rows: list[dict[str, object]] = []

    for J in J_list:
        comps = structured_components(J, alpha_total=alpha_total)
        H_dense = core_v1.build_H_dense(N, comps)
        eig = core_v1.eigendecompose(H_dense)
        alpha_each = float(alpha_total) / float(J)

        for hops in hops_list:
            result = core_v1.run_single(
                N=N,
                components=comps,
                K0=K0,
                t=t,
                psi1_0=psi1_0,
                psi2_0=psi2_0,
                R_hops=hops,
                H_dense=H_dense,
                eig=eig,
                use_qiskit=False,
            )
            rows.append(
                result_row(
                    "fixed_alpha_total",
                    nx=nx,
                    seed=seed,
                    K0=K0,
                    t=t,
                    J=J,
                    alpha_total=alpha_total,
                    alpha_each=alpha_each,
                    R_hops=hops,
                    result=result,
                )
            )
    return rows


def run_dense_time_sweep(
    *,
    nx: int,
    seed: int,
    K0: float,
    t_list: list[float],
    hops_list: list[int],
    J_dense: int,
    alpha_total: float,
) -> list[dict[str, object]]:
    """Sweep time for the densest structured case under fixed total coupling."""
    N = 2**int(nx)
    psi1_0, psi2_0, _, _ = cases.vortex_case(nx=nx, seed=seed)
    comps = structured_components(J_dense, alpha_total=alpha_total)
    H_dense = core_v1.build_H_dense(N, comps)
    eig = core_v1.eigendecompose(H_dense)
    alpha_each = float(alpha_total) / float(J_dense)

    rows: list[dict[str, object]] = []
    for hops in hops_list:
        for t in t_list:
            result = core_v1.run_single(
                N=N,
                components=comps,
                K0=K0,
                t=t,
                psi1_0=psi1_0,
                psi2_0=psi2_0,
                R_hops=hops,
                H_dense=H_dense,
                eig=eig,
                use_qiskit=False,
            )
            rows.append(
                result_row(
                    "dense_time_fixed_alpha_total",
                    nx=nx,
                    seed=seed,
                    K0=K0,
                    t=t,
                    J=J_dense,
                    alpha_total=alpha_total,
                    alpha_each=alpha_each,
                    R_hops=hops,
                    result=result,
                )
            )
    return rows


def run_fixed_alpha_each(
    *,
    nx: int,
    seed: int,
    K0: float,
    t: float,
    J_list: list[int],
    alpha_each: float,
) -> list[dict[str, object]]:
    """Sweep J while keeping each Fourier component at a fixed strength."""
    N = 2**int(nx)
    psi1_0, psi2_0, _, _ = cases.vortex_case(nx=nx, seed=seed)
    rows: list[dict[str, object]] = []

    for J in J_list:
        comps = structured_components(J, alpha_each=alpha_each)
        H_dense = core_v1.build_H_dense(N, comps)
        eig = core_v1.eigendecompose(H_dense)
        alpha_total = float(alpha_each) * float(J)

        result = core_v1.run_single(
            N=N,
            components=comps,
            K0=K0,
            t=t,
            psi1_0=psi1_0,
            psi2_0=psi2_0,
            R_hops=1,
            H_dense=H_dense,
            eig=eig,
            use_qiskit=False,
        )
        rows.append(
            result_row(
                "fixed_alpha_each",
                nx=nx,
                seed=seed,
                K0=K0,
                t=t,
                J=J,
                alpha_total=alpha_total,
                alpha_each=alpha_each,
                R_hops=1,
                result=result,
            )
        )
    return rows


def run_dense_hops_alpha_each(
    *,
    nx: int,
    seed: int,
    K0: float,
    t: float,
    hops_list: list[int],
    J_dense: int,
    alpha_each: float,
) -> list[dict[str, object]]:
    """Sweep closure depth for the densest structured case at fixed per-component strength."""
    N = 2**int(nx)
    psi1_0, psi2_0, _, _ = cases.vortex_case(nx=nx, seed=seed)
    comps = structured_components(J_dense, alpha_each=alpha_each)
    H_dense = core_v1.build_H_dense(N, comps)
    eig = core_v1.eigendecompose(H_dense)
    alpha_total = float(alpha_each) * float(J_dense)

    rows: list[dict[str, object]] = []
    for hops in hops_list:
        result = core_v1.run_single(
            N=N,
            components=comps,
            K0=K0,
            t=t,
            psi1_0=psi1_0,
            psi2_0=psi2_0,
            R_hops=hops,
            H_dense=H_dense,
            eig=eig,
            use_qiskit=False,
        )
        rows.append(
            result_row(
                "dense_hops_fixed_alpha_each",
                nx=nx,
                seed=seed,
                K0=K0,
                t=t,
                J=J_dense,
                alpha_total=alpha_total,
                alpha_each=alpha_each,
                R_hops=hops,
                result=result,
            )
        )
    return rows


def write_summary(
    path: Path,
    *,
    fixed_alpha_total_rows: list[dict[str, object]],
    dense_time_rows: list[dict[str, object]],
    fixed_alpha_each_rows: list[dict[str, object]],
    dense_hops_alpha_each_rows: list[dict[str, object]],
) -> None:
    """Write a compact Markdown summary next to the raw CSV outputs."""
    fixed_alpha_total_view = [
        row
        for row in fixed_alpha_total_rows
        if int(row["R_hops"]) == 1
    ]
    dense_time_view = dense_time_rows
    fixed_alpha_each_view = fixed_alpha_each_rows
    dense_hops_alpha_each_view = dense_hops_alpha_each_rows

    parts = [
        "# Rebuttal Stress Tests",
        "",
        f"These stress tests were prepared for the response to {REVIEWER_LABEL}.",
        "",
        "This directory contains reproducible stress tests for the rebuttal discussion on",
        "structured multi-component potentials and closure-depth tradeoffs.",
        "",
        "## Files",
        "",
        "- `fixed_alpha_total.csv`: sweep graph density `J` and closure depth `R_hops` with fixed total coupling.",
        "- `dense_time_fixed_alpha_total.csv`: time sweep for the densest structured case under fixed total coupling.",
        "- `fixed_alpha_each.csv`: sweep graph density `J` with fixed per-component coupling.",
        "- `dense_hops_fixed_alpha_each.csv`: closure-depth sweep for the densest case at fixed per-component coupling.",
        "",
        "## Snapshot: Fixed Total Coupling (`R_hops = 1`)",
        "",
        to_markdown_table(
            fixed_alpha_total_view,
            [
                "J",
                "alpha_total",
                "alpha_each",
                "R_size",
                "reduced_size",
                "err_rho_vs_full",
                "err_Z_frob",
                "leakage_apriori",
            ],
        ),
        "",
        "## Snapshot: Dense Case Time Sweep (Fixed Total Coupling)",
        "",
        to_markdown_table(
            dense_time_view,
            [
                "t",
                "R_hops",
                "R_size",
                "err_rho_vs_full",
                "err_Z_frob",
                "leakage_apriori",
                "bound_apriori",
            ],
        ),
        "",
        "## Snapshot: Fixed Per-Component Coupling (`R_hops = 1`)",
        "",
        to_markdown_table(
            fixed_alpha_each_view,
            [
                "J",
                "alpha_total",
                "alpha_each",
                "R_size",
                "reduced_size",
                "err_rho_vs_full",
                "err_Z_frob",
                "leakage_apriori",
            ],
        ),
        "",
        "## Snapshot: Dense Case Closure Sweep (Fixed Per-Component Coupling)",
        "",
        to_markdown_table(
            dense_hops_alpha_each_view,
            [
                "R_hops",
                "alpha_total",
                "R_size",
                "reduced_size",
                "err_rho_vs_full",
                "err_Z_frob",
                "leakage_apriori",
            ],
        ),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate rebuttal stress-test data")
    p.add_argument("--out-dir", default=str(ROOT / "data" / "rebuttal_stress_tests"))
    p.add_argument("--nx", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--K0", type=float, default=5.0)
    p.add_argument("--t", type=float, default=0.5, help="Default time for non-time-sweep sections")
    p.add_argument("--alpha-total", type=float, default=0.5)
    p.add_argument("--alpha-each", type=float, default=0.1)
    p.add_argument("--J-list", default=",".join(map(str, DEFAULT_J_LIST)))
    p.add_argument("--hops-list", default=",".join(map(str, DEFAULT_HOPS_LIST)))
    p.add_argument("--t-list", default=",".join(map(str, DEFAULT_T_LIST)))
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    J_list = _parse_int_list(args.J_list)
    hops_list = _parse_int_list(args.hops_list)
    t_list = _parse_float_list(args.t_list)
    J_dense = max(J_list)

    fixed_alpha_total_rows = run_fixed_alpha_total(
        nx=args.nx,
        seed=args.seed,
        K0=args.K0,
        t=args.t,
        J_list=J_list,
        hops_list=hops_list,
        alpha_total=args.alpha_total,
    )
    dense_time_rows = run_dense_time_sweep(
        nx=args.nx,
        seed=args.seed,
        K0=args.K0,
        t_list=t_list,
        hops_list=hops_list,
        J_dense=J_dense,
        alpha_total=args.alpha_total,
    )
    fixed_alpha_each_rows = run_fixed_alpha_each(
        nx=args.nx,
        seed=args.seed,
        K0=args.K0,
        t=args.t,
        J_list=J_list,
        alpha_each=args.alpha_each,
    )
    dense_hops_alpha_each_rows = run_dense_hops_alpha_each(
        nx=args.nx,
        seed=args.seed,
        K0=args.K0,
        t=args.t,
        hops_list=hops_list,
        J_dense=J_dense,
        alpha_each=args.alpha_each,
    )

    fixed_alpha_total_csv = out_dir / "fixed_alpha_total.csv"
    dense_time_csv = out_dir / "dense_time_fixed_alpha_total.csv"
    fixed_alpha_each_csv = out_dir / "fixed_alpha_each.csv"
    dense_hops_alpha_each_csv = out_dir / "dense_hops_fixed_alpha_each.csv"
    summary_md = out_dir / "summary.md"

    write_csv(fixed_alpha_total_csv, fixed_alpha_total_rows)
    write_csv(dense_time_csv, dense_time_rows)
    write_csv(fixed_alpha_each_csv, fixed_alpha_each_rows)
    write_csv(dense_hops_alpha_each_csv, dense_hops_alpha_each_rows)
    write_summary(
        summary_md,
        fixed_alpha_total_rows=fixed_alpha_total_rows,
        dense_time_rows=dense_time_rows,
        fixed_alpha_each_rows=fixed_alpha_each_rows,
        dense_hops_alpha_each_rows=dense_hops_alpha_each_rows,
    )

    print(f"Wrote: {fixed_alpha_total_csv}")
    print(f"Wrote: {dense_time_csv}")
    print(f"Wrote: {fixed_alpha_each_csv}")
    print(f"Wrote: {dense_hops_alpha_each_csv}")
    print(f"Wrote: {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
