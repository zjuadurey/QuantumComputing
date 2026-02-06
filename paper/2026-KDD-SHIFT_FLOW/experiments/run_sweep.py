"""experiments/run_sweep.py

Run SHIFT-FLOW sweeps and write a single CSV.

Baseline (for sweeps): FFT-based free evolution (V=0) with unitary FFT
conventions matching `test/shadow_test_v4.py`.

SHIFT-FLOW: shadow / truncated-mode observable dynamics (coherence-based low-pass
reconstruction), NOT classical shadow tomography.

Example:
  python experiments/run_sweep.py --overwrite
  python experiments/run_sweep.py --K0-list 2.5,4.5,6.5 --nx-list 6 --seeds 0,1
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
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_K0_LIST = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _make_t_list(t0: float, t1: float, dt: float) -> list[float]:
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if t1 < t0:
        raise ValueError("t1 must be >= t0")

    n_steps = int(round((t1 - t0) / dt))
    ts = [t0 + i * dt for i in range(n_steps + 1)]
    # avoid ugly float strings like 0.30000000000004
    return [float(f"{t:.12g}") for t in ts]


def _safe_div(num: float, den: float) -> float:
    return num / den if den != 0.0 else float("nan")


def _vorticity_from_current_and_density(
    Jx: np.ndarray,
    Jy: np.ndarray,
    rho: np.ndarray,
    dx: float,
    dy: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute omega from (Jx,Jy,rho) using v4's periodic central differences."""
    rho_safe = np.where(rho > eps, rho, eps)
    ux = Jx / rho_safe
    uy = Jy / rho_safe
    duy_dx = (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)) / (2.0 * dx)
    dux_dy = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) / (2.0 * dy)
    return duy_dx - dux_dy


def _maybe_write_header(out_path: Path, overwrite: bool) -> bool:
    if overwrite:
        return True
    return not out_path.exists()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SHIFT-FLOW sweep runner")
    p.add_argument("--out", default=str(ROOT / "results" / "sweep.csv"), help="output CSV path")
    p.add_argument("--overwrite", action="store_true", help="overwrite output CSV")

    p.add_argument("--nx-list", default=",".join(map(str, DEFAULT_NX_LIST)), help="comma-separated nx values")
    p.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)), help="comma-separated seeds")
    p.add_argument(
        "--K0-list",
        default=",".join(map(str, DEFAULT_K0_LIST)),
        help="comma-separated K0 values",
    )

    p.add_argument("--t-list", default=None, help="comma-separated t values (overrides t0/t1/dt)")
    p.add_argument("--t0", type=float, default=0.0)
    p.add_argument("--t1", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=0.1)

    p.add_argument("--no-omega", action="store_true", help="skip vorticity error")

    # case generation parameters
    p.add_argument("--sigma-base", type=float, default=2.0)
    p.add_argument("--sigma-jitter", type=float, default=0.05)
    p.add_argument("--shift-max", type=int, default=2)
    p.add_argument("--noise-eps", type=float, default=5e-4)
    p.add_argument("--canonical-seed", type=int, default=0)

    p.add_argument("--progress-every", type=int, default=25, help="print progress every N points")

    args = p.parse_args(argv)

    nx_list = _parse_int_list(args.nx_list)
    seeds = _parse_int_list(args.seeds)
    K0_list = _parse_float_list(args.K0_list)
    if args.t_list:
        t_list = _parse_float_list(args.t_list)
    else:
        t_list = _make_t_list(args.t0, args.t1, args.dt)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "nx",
        "N",
        "K0",
        "M",
        "t",
        "seed",
        "sigma_base",
        "sigma_eff",
        "shift_x",
        "shift_y",
        "noise_eps",
        "k0_1_y",
        "k0_1_x",
        "k0_2_y",
        "k0_2_x",
        # shadow vs baseline low-pass (sanity)
        "err_rho",
        "err_momentum",
        "err_omega",
        # shadow vs baseline full (truncation-to-full)
        "err_rho_vs_full",
        "err_momentum_vs_full",
        "err_omega_vs_full",
        # baseline low-pass vs baseline full (best possible under truncation)
        "err_rho_lp_vs_full",
        "err_momentum_lp_vs_full",
        "err_omega_lp_vs_full",
        "E_LP_base",
        "E_LP_shadow",
        "err_E_LP",
        "q_base",
        "q_shift",
        "measurement_proxy",
        "postprocess_task_proxy",
        "postprocess_full_proxy",
        "rt_baseline_full_s",
        "rt_baseline_lp_s",
        "rt_shadow_s",
        "rt_metrics_s",
        "rt_total_s",
    ]

    mode = "w" if args.overwrite else "a"
    write_header = _maybe_write_header(out_path, overwrite=args.overwrite)
    if (not args.overwrite) and out_path.exists():
        # append mode if file exists
        mode = "a"

    total = len(nx_list) * len(K0_list) * len(seeds) * len(t_list)
    done = 0

    with out_path.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for nx in nx_list:
            N = 2**int(nx)
            dx = 2.0 * math.pi / N
            dy = 2.0 * math.pi / N
            E = core_v0.energy_grid_free(N)

            # Precompute full k-space phases for all t (shared across seeds/K0)
            phase_full_map: dict[float, np.ndarray] = {}
            for t in t_list:
                phase_full_map[float(t)] = np.exp(-1j * E * float(t))

            for K0 in K0_list:
                mask = core_v0.low_freq_mask(N, K0)
                cost = metrics.cost_proxies(nx=int(nx), mask=mask)

                # precompute mask indices for faster fills
                mask_idx = np.nonzero(mask)

                for seed in seeds:
                    psi1_0, psi2_0, _stacked, meta = cases.vortex_case(
                        nx=int(nx),
                        seed=int(seed),
                        sigma_base=float(args.sigma_base),
                        sigma_jitter=float(args.sigma_jitter),
                        shift_max=int(args.shift_max),
                        noise_eps=float(args.noise_eps),
                        canonical_seed=int(args.canonical_seed),
                    )

                    b1_0 = core_v0.unitary_fft2(psi1_0)
                    b2_0 = core_v0.unitary_fft2(psi2_0)

                    # mask slices (used for low-pass baseline and shadow)
                    b1_0_mask = b1_0[mask_idx]
                    b2_0_mask = b2_0[mask_idx]

                    # choose reference modes once per (seed, mask)
                    k0_1 = core_v0.choose_reference_mode(b1_0, mask, prefer=(0, 0), min_rel=1e-3)
                    k0_2 = core_v0.choose_reference_mode(b2_0, mask, prefer=(0, 0), min_rel=1e-3)

                    # precompute coherence seeds for shadow (mask only)
                    b1_k0 = b1_0[k0_1]
                    b2_k0 = b2_0[k0_2]
                    z1_0_mask = b1_0_mask * np.conj(b1_k0)
                    z2_0_mask = b2_0_mask * np.conj(b2_k0)
                    E0_1 = float(E[k0_1])
                    E0_2 = float(E[k0_2])

                    # task-only baseline E_LP is time-invariant for free evolution
                    E_LP_base = float(np.sum(np.abs(b1_0_mask) ** 2 + np.abs(b2_0_mask) ** 2))

                    # reusable coefficient buffers (filled per t)
                    b1_full = np.empty((N, N), dtype=np.complex128)
                    b2_full = np.empty((N, N), dtype=np.complex128)

                    b1_lp = np.zeros((N, N), dtype=np.complex128)
                    b2_lp = np.zeros((N, N), dtype=np.complex128)
                    b1_shadow = np.zeros((N, N), dtype=np.complex128)
                    b2_shadow = np.zeros((N, N), dtype=np.complex128)

                    for t in t_list:
                        t_total0 = time.perf_counter()

                        t_f = float(t)
                        phase_full = phase_full_map[t_f]

                        # -------- baseline full (FFT) --------
                        t0 = time.perf_counter()
                        np.multiply(b1_0, phase_full, out=b1_full)
                        np.multiply(b2_0, phase_full, out=b2_full)
                        psi1_full = core_v0.unitary_ifft2(b1_full)
                        psi2_full = core_v0.unitary_ifft2(b2_full)
                        rt_baseline_full_s = time.perf_counter() - t0

                        # -------- baseline low-pass (FFT) --------
                        t0 = time.perf_counter()
                        phase_mask = phase_full[mask_idx]
                        b1_lp.fill(0.0)
                        b2_lp.fill(0.0)
                        b1_lp[mask_idx] = b1_0_mask * phase_mask
                        b2_lp[mask_idx] = b2_0_mask * phase_mask
                        psi1_base_lp = core_v0.unitary_ifft2(b1_lp)
                        psi2_base_lp = core_v0.unitary_ifft2(b2_lp)
                        rt_baseline_lp_s = time.perf_counter() - t0

                        # -------- shadow low-pass (coherences) --------
                        t0 = time.perf_counter()

                        # component 1
                        exp_iE0_1t = np.exp(1j * E0_1 * t_f)
                        phase1_mask = phase_mask * exp_iE0_1t
                        b1_k0_t = b1_k0 * np.conj(exp_iE0_1t)
                        b1_shadow.fill(0.0)
                        b1_shadow[mask_idx] = (z1_0_mask * phase1_mask) / np.conj(b1_k0_t)
                        b1_shadow[k0_1] = b1_k0_t

                        # component 2
                        exp_iE0_2t = np.exp(1j * E0_2 * t_f)
                        phase2_mask = phase_mask * exp_iE0_2t
                        b2_k0_t = b2_k0 * np.conj(exp_iE0_2t)
                        b2_shadow.fill(0.0)
                        b2_shadow[mask_idx] = (z2_0_mask * phase2_mask) / np.conj(b2_k0_t)
                        b2_shadow[k0_2] = b2_k0_t

                        psi1_shadow = core_v0.unitary_ifft2(b1_shadow)
                        psi2_shadow = core_v0.unitary_ifft2(b2_shadow)
                        rt_shadow_s = time.perf_counter() - t0

                        # -------- metrics --------
                        t0 = time.perf_counter()
                        # fields
                        rho_full = core_v0.density_from_components(psi1_full, psi2_full)
                        rho_lp = core_v0.density_from_components(psi1_base_lp, psi2_base_lp)
                        rho_shadow = core_v0.density_from_components(psi1_shadow, psi2_shadow)

                        Jx_full, Jy_full = core_v0.current_from_components(psi1_full, psi2_full, dx=dx, dy=dy)
                        Jx_lp, Jy_lp = core_v0.current_from_components(psi1_base_lp, psi2_base_lp, dx=dx, dy=dy)
                        Jx_sh, Jy_sh = core_v0.current_from_components(psi1_shadow, psi2_shadow, dx=dx, dy=dy)

                        # shadow vs low-pass baseline (sanity)
                        err_rho = metrics.rel_l2(rho_shadow, rho_lp)
                        err_mom = metrics.rel_l2_vec(Jx_sh, Jy_sh, Jx_lp, Jy_lp)

                        # shadow vs full baseline (approx to full)
                        err_rho_vs_full = metrics.rel_l2(rho_shadow, rho_full)
                        err_mom_vs_full = metrics.rel_l2_vec(Jx_sh, Jy_sh, Jx_full, Jy_full)

                        # low-pass baseline vs full baseline (truncation error)
                        err_rho_lp_vs_full = metrics.rel_l2(rho_lp, rho_full)
                        err_mom_lp_vs_full = metrics.rel_l2_vec(Jx_lp, Jy_lp, Jx_full, Jy_full)

                        if args.no_omega:
                            err_omg = float("nan")
                            err_omg_vs_full = float("nan")
                            err_omg_lp_vs_full = float("nan")
                        else:
                            omg_full = _vorticity_from_current_and_density(Jx_full, Jy_full, rho_full, dx=dx, dy=dy)
                            omg_lp = _vorticity_from_current_and_density(Jx_lp, Jy_lp, rho_lp, dx=dx, dy=dy)
                            omg_sh = _vorticity_from_current_and_density(Jx_sh, Jy_sh, rho_shadow, dx=dx, dy=dy)
                            err_omg = metrics.rel_l2(omg_sh, omg_lp)
                            err_omg_vs_full = metrics.rel_l2(omg_sh, omg_full)
                            err_omg_lp_vs_full = metrics.rel_l2(omg_lp, omg_full)

                        b1_shadow_mask = b1_shadow[mask_idx]
                        b2_shadow_mask = b2_shadow[mask_idx]
                        E_LP_shadow = float(np.sum(np.abs(b1_shadow_mask) ** 2 + np.abs(b2_shadow_mask) ** 2))
                        err_E_LP = _safe_div(abs(E_LP_shadow - E_LP_base), abs(E_LP_base))

                        rt_metrics_s = time.perf_counter() - t0
                        rt_total_s = time.perf_counter() - t_total0

                        row = {
                            "nx": int(nx),
                            "N": int(N),
                            "K0": float(K0),
                            "M": int(cost.M),
                            "t": float(t),
                            "seed": int(seed),
                            "sigma_base": float(meta.sigma_base),
                            "sigma_eff": float(meta.sigma_eff),
                            "shift_x": int(meta.shift_x),
                            "shift_y": int(meta.shift_y),
                            "noise_eps": float(meta.noise_eps),
                            "k0_1_y": int(k0_1[0]),
                            "k0_1_x": int(k0_1[1]),
                            "k0_2_y": int(k0_2[0]),
                            "k0_2_x": int(k0_2[1]),
                            "err_rho": float(err_rho),
                            "err_momentum": float(err_mom),
                            "err_omega": float(err_omg),
                            "err_rho_vs_full": float(err_rho_vs_full),
                            "err_momentum_vs_full": float(err_mom_vs_full),
                            "err_omega_vs_full": float(err_omg_vs_full),
                            "err_rho_lp_vs_full": float(err_rho_lp_vs_full),
                            "err_momentum_lp_vs_full": float(err_mom_lp_vs_full),
                            "err_omega_lp_vs_full": float(err_omg_lp_vs_full),
                            "E_LP_base": float(E_LP_base),
                            "E_LP_shadow": float(E_LP_shadow),
                            "err_E_LP": float(err_E_LP),
                            "q_base": int(cost.q_base),
                            "q_shift": int(cost.q_shift),
                            "measurement_proxy": float(cost.measurement_proxy),
                            "postprocess_task_proxy": float(cost.postprocess_task_proxy),
                            "postprocess_full_proxy": float(cost.postprocess_full_proxy),
                            "rt_baseline_full_s": float(rt_baseline_full_s),
                            "rt_baseline_lp_s": float(rt_baseline_lp_s),
                            "rt_shadow_s": float(rt_shadow_s),
                            "rt_metrics_s": float(rt_metrics_s),
                            "rt_total_s": float(rt_total_s),
                        }

                        writer.writerow(row)
                        f.flush()

                        done += 1
                        if args.progress_every > 0 and (done % args.progress_every == 0 or done == total):
                            pct = 100.0 * done / total
                            print(f"[{done}/{total} {pct:5.1f}%] nx={nx} K0={K0} seed={seed} t={t}")

    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
