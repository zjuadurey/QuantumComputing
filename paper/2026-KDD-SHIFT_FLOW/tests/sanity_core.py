"""tests/sanity_core.py

Sanity check that the refactored core matches `test/shadow_test_v4.py`.

Run from repo root:
  python tests/sanity_core.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shiftflow import core_v0 as core


def _load_v4_module():
    root = Path(__file__).resolve().parents[1]
    v4_path = root / "test" / "shadow_test_v4.py"
    if not v4_path.exists():
        raise FileNotFoundError(f"Missing reference file: {v4_path}")

    spec = importlib.util.spec_from_file_location("shadow_test_v4_ref", v4_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for: {v4_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.linalg.norm(a - b))
    den = float(np.linalg.norm(b))
    return num / den if den != 0.0 else float("nan")


def main() -> int:
    v4 = _load_v4_module()

    # Fixed sanity setting
    nx = 6
    ny = 6
    N = 2**nx
    K0 = 2.5
    t = 0.3
    seed = 0
    sigma = 2.0

    # seed is currently unused for the v4 IC; keep for forward-compat
    _ = seed

    dx = 2.0 * np.pi / N
    dy = 2.0 * np.pi / N

    # ---- Initial condition (reference) ----
    psi1_0_ref, psi2_0_ref, initial_state_ref = v4.vortex_initial_condition(N=N, sigma=sigma)

    mask_ref = v4.low_freq_mask(N, K0)
    E_ref = v4.energy_grid_free(N)

    # ---- Baseline full (reference, FFT) ----
    phase_ref = np.exp(-1j * E_ref * t)
    b1_0_ref = v4.unitary_fft2(psi1_0_ref)
    b2_0_ref = v4.unitary_fft2(psi2_0_ref)
    b1_full_ref = b1_0_ref * phase_ref
    b2_full_ref = b2_0_ref * phase_ref
    psi1_full_ref = v4.unitary_ifft2(b1_full_ref)
    psi2_full_ref = v4.unitary_ifft2(b2_full_ref)

    # ---- Baseline low-pass (reference) ----
    b1_full_lp_ref = np.where(mask_ref, b1_full_ref, 0.0)
    b2_full_lp_ref = np.where(mask_ref, b2_full_ref, 0.0)
    psi1_full_lp_ref = v4.unitary_ifft2(b1_full_lp_ref)
    psi2_full_lp_ref = v4.unitary_ifft2(b2_full_lp_ref)

    # ---- Shadow (reference) ----
    k0_1_ref = v4.choose_reference_mode(b1_0_ref, mask_ref, prefer=(0, 0), min_rel=1e-3)
    k0_2_ref = v4.choose_reference_mode(b2_0_ref, mask_ref, prefer=(0, 0), min_rel=1e-3)
    b1_shadow_ref = v4.shadow_evolve_lowpass_from_coherences(b0=b1_0_ref, mask=mask_ref, t=t, k0_idx=k0_1_ref, E=E_ref)
    b2_shadow_ref = v4.shadow_evolve_lowpass_from_coherences(b0=b2_0_ref, mask=mask_ref, t=t, k0_idx=k0_2_ref, E=E_ref)
    psi1_shadow_ref = v4.unitary_ifft2(b1_shadow_ref)
    psi2_shadow_ref = v4.unitary_ifft2(b2_shadow_ref)

    # ---- Core: mask & shadow should exactly match v4 ----
    mask_core = core.low_freq_mask(N, K0)
    if not np.array_equal(mask_core, mask_ref):
        raise AssertionError("low_freq_mask mismatch between core_v0 and v4")

    E_core = core.energy_grid_free(N)
    if rel_l2(E_core, E_ref) != 0.0:
        raise AssertionError("energy_grid_free mismatch between core_v0 and v4")

    # Baseline full (core, FFT)
    psi1_full_core, psi2_full_core, b1_full_core, b2_full_core = core.evolve_components_fft_v0(
        psi1_0_ref,
        psi2_0_ref,
        t=t,
        E=E_core,
        return_coeffs=True,
    )

    # Baseline low-pass (core)
    b1_full_lp_core = np.where(mask_ref, b1_full_core, 0.0)
    b2_full_lp_core = np.where(mask_ref, b2_full_core, 0.0)
    psi1_full_lp_core = core.unitary_ifft2(b1_full_lp_core)
    psi2_full_lp_core = core.unitary_ifft2(b2_full_lp_core)

    # Shadow evolution (core)
    b1_0_core = core.unitary_fft2(psi1_0_ref)
    b2_0_core = core.unitary_fft2(psi2_0_ref)
    k0_1_core = core.choose_reference_mode(b1_0_core, mask_ref, prefer=(0, 0), min_rel=1e-3)
    k0_2_core = core.choose_reference_mode(b2_0_core, mask_ref, prefer=(0, 0), min_rel=1e-3)
    if k0_1_core != k0_1_ref or k0_2_core != k0_2_ref:
        raise AssertionError(f"k0 mismatch: core={(k0_1_core, k0_2_core)} v4={(k0_1_ref, k0_2_ref)}")

    b1_shadow_core = core.shadow_evolve_lowpass_from_coherences(b0=b1_0_core, mask=mask_ref, t=t, k0_idx=k0_1_core, E=E_core)
    b2_shadow_core = core.shadow_evolve_lowpass_from_coherences(b0=b2_0_core, mask=mask_ref, t=t, k0_idx=k0_2_core, E=E_core)
    psi1_shadow_core = core.unitary_ifft2(b1_shadow_core)
    psi2_shadow_core = core.unitary_ifft2(b2_shadow_core)

    # ---- Compare fields: rho + momentum (Jx,Jy) + optional omega ----
    tol = 1e-10

    def _diag(mod, psi1: np.ndarray, psi2: np.ndarray):
        rho = mod.density_from_components(psi1, psi2)
        Jx, Jy = mod.current_from_components(psi1, psi2, dx=dx, dy=dy)
        omg = mod.vorticity_from_components(psi1, psi2, dx=dx, dy=dy)
        return rho, Jx, Jy, omg

    def _check_fields(tag: str, psi1_core: np.ndarray, psi2_core: np.ndarray, psi1_ref: np.ndarray, psi2_ref: np.ndarray):
        rho_c, Jx_c, Jy_c, omg_c = _diag(core, psi1_core, psi2_core)
        rho_r, Jx_r, Jy_r, omg_r = _diag(v4, psi1_ref, psi2_ref)

        err_rho = rel_l2(rho_c, rho_r)
        err_Jx = rel_l2(Jx_c, Jx_r)
        err_Jy = rel_l2(Jy_c, Jy_r)
        err_omg = rel_l2(omg_c, omg_r)

        print(f"[{tag}] relL2 rho={err_rho:.3e}  Jx={err_Jx:.3e}  Jy={err_Jy:.3e}  omg={err_omg:.3e}")

        assert err_rho < tol, f"{tag}: rho relL2 {err_rho} >= {tol}"
        assert err_Jx < tol, f"{tag}: Jx relL2 {err_Jx} >= {tol}"
        assert err_Jy < tol, f"{tag}: Jy relL2 {err_Jy} >= {tol}"
        assert err_omg < tol, f"{tag}: omega relL2 {err_omg} >= {tol}"

    _check_fields("baseline_full_fft", psi1_full_core, psi2_full_core, psi1_full_ref, psi2_full_ref)
    _check_fields("baseline_lowpass", psi1_full_lp_core, psi2_full_lp_core, psi1_full_lp_ref, psi2_full_lp_ref)
    _check_fields("shadow", psi1_shadow_core, psi2_shadow_core, psi1_shadow_ref, psi2_shadow_ref)

    # ---- Optional spot-check: Qiskit baseline (if available) ----
    try:
        import qiskit_aer  # noqa: F401
    except Exception:
        print("[qiskit] qiskit-aer not installed; skipping Qiskit spot-check")
    else:
        sv_ref_q = v4.evolve_statevector_v0(nx=nx, ny=ny, t=t, initial_state=initial_state_ref)
        psi1_q, psi2_q = v4.statevector_to_components(sv_ref_q, N)
        _check_fields("baseline_full_qiskit_vs_fft", psi1_q, psi2_q, psi1_full_ref, psi2_full_ref)

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
