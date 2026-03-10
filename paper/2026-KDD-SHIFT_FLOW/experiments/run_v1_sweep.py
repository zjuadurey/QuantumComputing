"""experiments/run_v1_sweep.py

V!=0 SHIFT-FLOW experiment sweep.

Runs full-state and Galerkin-truncated evolution for a difficulty ladder
of potentials, sweeping K0 and coupling strength.

Example:
  python experiments/run_v1_sweep.py
  python experiments/run_v1_sweep.py --nx 5 --alphas 0.1,0.5,1.0
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shiftflow import cases, core_v0, core_v1  # noqa: E402


DEFAULT_K0_LIST = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
DEFAULT_ALPHAS = [0.1, 0.5, 1.0, 2.0]
DEFAULT_T_LIST = [0.1, 0.3, 0.5, 0.7, 1.0]

FIELDNAMES = [
    "nx", "N", "K0", "M_K", "R_size", "t", "seed",
    "alpha", "J", "V_label", "R_hops",
    "err_b_K_rel", "err_rho_vs_full", "err_E_LP",
    "leakage_apriori", "bound_apriori", "err_Z_frob",
    "err_rho_lp_vs_full",
    "rt_build_H_s", "rt_eig_s", "rt_full_s", "rt_galerkin_s", "rt_total_s",
]


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="V!=0 SHIFT-FLOW sweep")
    p.add_argument("--out", default=str(ROOT / "results" / "sweep_v1.csv"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--nx", type=int, default=6)
    p.add_argument("--K0-list", default=",".join(map(str, DEFAULT_K0_LIST)))
    p.add_argument("--alphas", default=",".join(map(str, DEFAULT_ALPHAS)))
    p.add_argument("--t-list", default=",".join(map(str, DEFAULT_T_LIST)))
    p.add_argument("--seeds", default="0")
    p.add_argument("--R-hops", type=int, default=1)
    p.add_argument("--qx", type=int, default=1)
    p.add_argument("--qy", type=int, default=0)

    # Tier 2: multi-component
    p.add_argument(
        "--tier2-J-list", default="",
        help="comma-separated J values for Tier 2 (random multi-component V)"
    )
    p.add_argument("--tier2-alpha-scale", type=float, default=0.5)
    p.add_argument("--tier2-q-max", type=int, default=3)
    p.add_argument("--tier2-seed", type=int, default=42)

    # Qiskit shadow evolution
    p.add_argument(
        "--use-qiskit", action="store_true", default=True,
        help="Use Qiskit Aer for shadow (Galerkin) evolution (default: True)"
    )
    p.add_argument(
        "--no-qiskit", dest="use_qiskit", action="store_false",
        help="Use classical scipy.expm for shadow evolution"
    )

    args = p.parse_args(argv)

    nx = args.nx
    N = 2 ** nx
    K0_list = _parse_float_list(args.K0_list)
    alpha_list = _parse_float_list(args.alphas)
    t_list = _parse_float_list(args.t_list)
    seeds = _parse_int_list(args.seeds)
    R_hops = args.R_hops

    # Qiskit Aer simulator (shared instance)
    qiskit_sim = None
    if args.use_qiskit:
        from qiskit_aer import AerSimulator
        qiskit_sim = AerSimulator(method="statevector")
        print(f"Qiskit Aer shadow evolution: ENABLED")
    else:
        print(f"Qiskit Aer shadow evolution: DISABLED (classical scipy.expm)")

    # Build potential configurations
    configs: list[tuple[list[core_v1.FourierPotential], str]] = []

    # Tier 1: single-component, sweep alpha
    for alpha in alpha_list:
        comps = core_v1.potential_single(alpha, qx=args.qx, qy=args.qy)
        configs.append((comps, f"tier1_a{alpha}"))

    # Tier 2: multi-component
    if args.tier2_J_list:
        J_list = _parse_int_list(args.tier2_J_list)
        for J in J_list:
            comps = core_v1.potential_multi_random(
                J, alpha_scale=args.tier2_alpha_scale,
                q_max=args.tier2_q_max, seed=args.tier2_seed,
            )
            configs.append((comps, f"tier2_J{J}"))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "a"
    write_header = args.overwrite or not out_path.exists()

    total = len(configs) * len(K0_list) * len(t_list) * len(seeds)
    done = 0

    with out_path.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for comps, config_tag in configs:
            # Build H and eigendecompose once per potential config
            t0 = time.perf_counter()
            H = core_v1.build_H_dense(N, comps)
            rt_build = time.perf_counter() - t0

            t0 = time.perf_counter()
            eig = core_v1.eigendecompose(H)
            rt_eig = time.perf_counter() - t0

            alpha_total = sum(abs(c.alpha) for c in comps)
            V_label = core_v1.potential_label(comps)
            J = len(comps)
            print(f"\n--- {V_label} (J={J}) | H build: {rt_build:.2f}s, eig: {rt_eig:.2f}s ---")

            for seed in seeds:
                psi1_0, psi2_0, _, meta = cases.vortex_case(nx=nx, seed=seed)

                for K0 in K0_list:
                    for t_val in t_list:
                        t_total0 = time.perf_counter()

                        t0 = time.perf_counter()
                        result = core_v1.run_single(
                            N=N, components=comps, K0=K0, t=t_val,
                            psi1_0=psi1_0, psi2_0=psi2_0,
                            r0=(0, 0), R_hops=R_hops,
                            H_dense=H, eig=eig,
                            use_qiskit=args.use_qiskit,
                            qiskit_sim=qiskit_sim,
                        )
                        rt_point = time.perf_counter() - t0

                        rt_total = time.perf_counter() - t_total0

                        # Timing: Qiskit circuit time for shadow evolution
                        rt_galerkin = (
                            result.rt_qiskit_K_s + result.rt_qiskit_R_s
                            if args.use_qiskit
                            else float("nan")
                        )

                        row = {
                            "nx": nx, "N": N,
                            "K0": K0, "M_K": result.M_K,
                            "R_size": result.R_size,
                            "t": t_val, "seed": seed,
                            "alpha": alpha_total, "J": J,
                            "V_label": V_label, "R_hops": R_hops,
                            "err_b_K_rel": result.err_b_K_rel,
                            "err_rho_vs_full": result.err_rho_vs_full,
                            "err_E_LP": result.err_E_LP,
                            "leakage_apriori": result.leakage_apriori,
                            "bound_apriori": result.bound_apriori,
                            "err_Z_frob": result.err_Z_frob,
                            "err_rho_lp_vs_full": result.err_rho_lp_vs_full,
                            "rt_build_H_s": rt_build,
                            "rt_eig_s": rt_eig,
                            "rt_full_s": rt_point - rt_galerkin if args.use_qiskit else float("nan"),
                            "rt_galerkin_s": rt_galerkin,
                            "rt_total_s": rt_total,
                        }
                        writer.writerow(row)
                        f.flush()

                        done += 1
                        if done % 10 == 0 or done == total:
                            pct = 100.0 * done / total
                            print(
                                f"  [{done}/{total} {pct:5.1f}%] K0={K0} t={t_val}"
                                f" err_rho={result.err_rho_vs_full:.3e}"
                                f" leak={result.leakage_apriori:.3e}"
                                f" bound={result.bound_apriori:.3e}"
                                f" |Z|err={result.err_Z_frob:.3e}"
                            )

    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
