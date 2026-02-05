"""experiments/run_qiskit_spotcheck.py

Qiskit spot-check validation for the FFT baseline (V=0).

Purpose:
- Sweeps use the FFT baseline for speed.
- This script uses Qiskit statevector simulation (QFT -> phase -> iQFT) on a
  small subset of points to validate equivalence and improve experimental
  credibility.

This is NOT classical shadow tomography.

Examples:
  # validate a few random points from an existing sweep
  python experiments/run_qiskit_spotcheck.py --from-sweep results/sweep.csv --n 12 --overwrite

  # validate an explicit grid
  python experiments/run_qiskit_spotcheck.py --nx-list 5,6 --K0-list 2.5 --seeds 0 --t-list 0,0.3 --overwrite
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shiftflow import cases, core_v0, metrics  # noqa: E402


DEFAULT_NX_LIST = [5, 6, 7]
DEFAULT_SEEDS = [0]
DEFAULT_K0_LIST = [2.5]
DEFAULT_T_LIST = [0.0, 0.3, 1.0]


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _safe_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _load_points_from_sweep(path: Path) -> list[dict]:
    pts: list[dict] = []
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            pts.append(
                {
                    "nx": int(row["nx"]),
                    "K0": _safe_float(row["K0"]),
                    "t": _safe_float(row["t"]),
                    "seed": int(row["seed"]),
                }
            )
    return pts


def _sample_points(points: list[dict], n: int, sample_seed: int) -> list[dict]:
    if n <= 0 or n >= len(points):
        return points
    rng = np.random.default_rng(int(sample_seed))
    idx = rng.choice(len(points), size=int(n), replace=False)
    return [points[int(i)] for i in idx]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Qiskit spot-check for FFT baseline")
    p.add_argument("--out", default=str(ROOT / "results" / "qiskit_spotcheck.csv"))
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--from-sweep", default=None, help="path to sweep.csv to sample points from")
    p.add_argument("--n", type=int, default=12, help="#points to sample from sweep (0 => all)")
    p.add_argument("--sample-seed", type=int, default=0)

    p.add_argument("--nx-list", default=",".join(map(str, DEFAULT_NX_LIST)))
    p.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    p.add_argument("--K0-list", default=",".join(map(str, DEFAULT_K0_LIST)))
    p.add_argument("--t-list", default=",".join(map(str, DEFAULT_T_LIST)))

    p.add_argument("--no-omega", action="store_true")

    # case generation parameters (must match sweep if sampling from it)
    p.add_argument("--sigma-base", type=float, default=2.0)
    p.add_argument("--sigma-jitter", type=float, default=0.05)
    p.add_argument("--shift-max", type=int, default=2)
    p.add_argument("--noise-eps", type=float, default=5e-4)
    p.add_argument("--canonical-seed", type=int, default=0)

    args = p.parse_args(argv)

    # Require qiskit-aer at runtime
    try:
        import qiskit_aer  # noqa: F401
    except Exception as e:
        raise SystemExit(
            "qiskit-aer is required for Qiskit spot-checks. "
            "Install it (or run in your qiskit env) and rerun.\n" + str(e)
        )

    if args.from_sweep:
        sweep_path = (ROOT / args.from_sweep).resolve() if not Path(args.from_sweep).is_absolute() else Path(args.from_sweep)
        points_all = _load_points_from_sweep(sweep_path)
        points = _sample_points(points_all, int(args.n), int(args.sample_seed))
    else:
        nx_list = _parse_int_list(args.nx_list)
        seeds = _parse_int_list(args.seeds)
        K0_list = _parse_float_list(args.K0_list)
        t_list = _parse_float_list(args.t_list)
        points = []
        for nx in nx_list:
            for K0 in K0_list:
                for seed in seeds:
                    for t in t_list:
                        points.append({"nx": int(nx), "K0": float(K0), "seed": int(seed), "t": float(t)})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and (not args.overwrite):
        raise SystemExit(f"Refusing to overwrite existing file (use --overwrite): {out_path}")

    fieldnames = [
        "nx",
        "N",
        "K0",
        "t",
        "seed",
        "err_rho_fft_vs_qiskit",
        "err_momentum_fft_vs_qiskit",
        "err_omega_fft_vs_qiskit",
        "rt_fft_s",
        "rt_qiskit_s",
    ]

    # aggregate summary
    max_rho = 0.0
    max_mom = 0.0
    max_omg = 0.0

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for i, pt in enumerate(points, start=1):
            nx = int(pt["nx"])
            N = 2**nx
            K0 = float(pt["K0"])
            t = float(pt["t"])
            seed = int(pt["seed"])

            dx = 2.0 * math.pi / N
            dy = 2.0 * math.pi / N

            psi1_0, psi2_0, stacked, _meta = cases.vortex_case(
                nx=nx,
                seed=seed,
                sigma_base=float(args.sigma_base),
                sigma_jitter=float(args.sigma_jitter),
                shift_max=int(args.shift_max),
                noise_eps=float(args.noise_eps),
                canonical_seed=int(args.canonical_seed),
            )

            E = core_v0.energy_grid_free(N)

            # FFT baseline (full)
            t0 = time.perf_counter()
            psi1_fft, psi2_fft = core_v0.evolve_components_fft_v0(psi1_0, psi2_0, t=t, E=E)
            rt_fft = time.perf_counter() - t0

            # Qiskit baseline (full)
            t0 = time.perf_counter()
            sv = core_v0.evolve_statevector_v0(nx=nx, ny=nx, t=t, initial_state=stacked)
            psi1_q, psi2_q = core_v0.statevector_to_components(sv, N)
            rt_q = time.perf_counter() - t0

            err_rho = metrics.err_rho_from_components(psi1_pred=psi1_fft, psi2_pred=psi2_fft, psi1_ref=psi1_q, psi2_ref=psi2_q)
            err_mom = metrics.err_momentum_from_components(
                psi1_pred=psi1_fft,
                psi2_pred=psi2_fft,
                psi1_ref=psi1_q,
                psi2_ref=psi2_q,
                dx=dx,
                dy=dy,
            )
            if args.no_omega:
                err_omg = float("nan")
            else:
                err_omg = metrics.err_omega_from_components(
                    psi1_pred=psi1_fft,
                    psi2_pred=psi2_fft,
                    psi1_ref=psi1_q,
                    psi2_ref=psi2_q,
                    dx=dx,
                    dy=dy,
                )

            max_rho = max(max_rho, float(err_rho))
            max_mom = max(max_mom, float(err_mom))
            if not math.isnan(err_omg):
                max_omg = max(max_omg, float(err_omg))

            w.writerow(
                {
                    "nx": nx,
                    "N": N,
                    "K0": K0,
                    "t": t,
                    "seed": seed,
                    "err_rho_fft_vs_qiskit": float(err_rho),
                    "err_momentum_fft_vs_qiskit": float(err_mom),
                    "err_omega_fft_vs_qiskit": float(err_omg),
                    "rt_fft_s": float(rt_fft),
                    "rt_qiskit_s": float(rt_q),
                }
            )
            f.flush()

            print(
                f"[{i}/{len(points)}] nx={nx} K0={K0} seed={seed} t={t} "
                f"err_rho={err_rho:.2e} err_mom={err_mom:.2e} err_omg={err_omg:.2e}"
            )

    print(f"Wrote: {out_path}")
    print(f"Max errors: rho={max_rho:.3e} momentum={max_mom:.3e} omega={max_omg:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
