from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_aggregate(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[(row["split"], row["metric"])] = row
    return rows


def _read_density_metric(
    base_dir: Path,
    split_to_file: dict[str, str],
    aggregate_csv: Path | None = None,
) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    if aggregate_csv is not None and aggregate_csv.exists():
        agg = _read_aggregate(aggregate_csv)
        split_map = {
            "id": "test",
            "ood_alpha": "split_ood_alpha",
            "ood_structure": "split_ood_structure",
        }
        for key, split_name in split_map.items():
            row = agg.get((split_name, "density_rel_l2"))
            if row is None:
                continue
            out[key] = (float(row["mean"]), float(row["std"]))
        return out

    for key, rel in split_to_file.items():
        m = _read_json(base_dir / rel)
        out[key] = (float(m["density_rel_l2"]), 0.0)
    return out


def _read_long_final(base_dir: Path, rel: str) -> float:
    data = _read_json(base_dir / rel)
    curve = data.get("density_rel_l2_curve", [])
    if not curve:
        return float("nan")
    return float(curve[-1])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_exp32_main_table(results_root: Path) -> list[dict[str, Any]]:
    exp32 = results_root / "bench_exp32"
    budget_rows: list[dict[str, Any]] = []

    full_shadow = _read_density_metric(
        exp32 / "fixed_shadowfluid",
        {
            "id": "forecasting_metrics.json",
            "ood_alpha": "ood_split_ood_alpha.json",
            "ood_structure": "ood_split_ood_structure.json",
        },
    )
    full_shadow_long = _read_long_final(exp32 / "fixed_shadowfluid", "long_rollout_curves.json")
    budget_rows.append(
        {
            "group": "anchor",
            "method": "ShadowFluid (full)",
            "base_K0": 6,
            "id_density_rel_l2": full_shadow["id"][0],
            "id_density_rel_l2_std": full_shadow["id"][1],
            "ood_alpha_density_rel_l2": full_shadow["ood_alpha"][0],
            "ood_alpha_density_rel_l2_std": full_shadow["ood_alpha"][1],
            "ood_structure_density_rel_l2": full_shadow["ood_structure"][0],
            "ood_structure_density_rel_l2_std": full_shadow["ood_structure"][1],
            "long_rollout_final_density_rel_l2": full_shadow_long,
            "gain_vs_budgeted_base_id": "",
            "notes": "full reference solver",
        }
    )

    base_specs = [
        (
            4,
            "shadowfluid_basek4",
            None,
            "shadowfluid_residual_densityfirst",
            results_root / "bench_exp32" / "shadowfluid_residual_densityfirst_exp32_multiseed" / "aggregate_summary.csv",
        ),
        (
            3,
            "shadowfluid_basek3",
            None,
            "shadowfluid_residual_densityfirst_k3",
            results_root / "bench_exp32" / "shadowfluid_residual_densityfirst_k3_short_multiseed" / "aggregate_summary.csv",
        ),
        (
            2,
            "shadowfluid_basek2",
            None,
            "shadowfluid_residual_densityfirst_k2_short",
            results_root / "bench_exp32" / "shadowfluid_residual_densityfirst_k2_short_multiseed" / "aggregate_summary.csv",
        ),
    ]

    for base_k0, base_run, _, res_run, res_agg in base_specs:
        base = _read_density_metric(
            exp32 / base_run,
            {
                "id": "forecasting_metrics.json",
                "ood_alpha": "ood_split_ood_alpha.json",
                "ood_structure": "ood_split_ood_structure.json",
            },
        )
        base_long = _read_long_final(exp32 / base_run, "long_rollout_curves.json") if (exp32 / base_run / "long_rollout_curves.json").exists() else float("nan")
        budget_rows.append(
            {
                "group": "main",
                "method": f"ShadowFluid budgeted (K0={base_k0})",
                "base_K0": base_k0,
                "id_density_rel_l2": base["id"][0],
                "id_density_rel_l2_std": base["id"][1],
                "ood_alpha_density_rel_l2": base["ood_alpha"][0],
                "ood_alpha_density_rel_l2_std": base["ood_alpha"][1],
                "ood_structure_density_rel_l2": base["ood_structure"][0],
                "ood_structure_density_rel_l2_std": base["ood_structure"][1],
                "long_rollout_final_density_rel_l2": base_long,
                "gain_vs_budgeted_base_id": 0.0,
                "notes": "under-budget physics solver",
            }
        )

        res = _read_density_metric(
            exp32 / res_run,
            {
                "id": "test_metrics.json",
                "ood_alpha": "ood_split_ood_alpha.json",
                "ood_structure": "ood_split_ood_structure.json",
            },
            aggregate_csv=res_agg,
        )
        res_long = _read_long_final(exp32 / res_run, "long_rollout_curves.json") if (exp32 / res_run / "long_rollout_curves.json").exists() else float("nan")
        budget_rows.append(
            {
                "group": "main",
                "method": f"ShadowFluid + residual (K0={base_k0})",
                "base_K0": base_k0,
                "id_density_rel_l2": res["id"][0],
                "id_density_rel_l2_std": res["id"][1],
                "ood_alpha_density_rel_l2": res["ood_alpha"][0],
                "ood_alpha_density_rel_l2_std": res["ood_alpha"][1],
                "ood_structure_density_rel_l2": res["ood_structure"][0],
                "ood_structure_density_rel_l2_std": res["ood_structure"][1],
                "long_rollout_final_density_rel_l2": res_long,
                "gain_vs_budgeted_base_id": base["id"][0] - res["id"][0],
                "notes": "conservative density-first residual correction",
            }
        )
    return budget_rows


def build_reference_table(results_root: Path) -> list[dict[str, Any]]:
    exp32 = results_root / "bench_exp32"
    ref_rows: list[dict[str, Any]] = []
    specs = [
        (
            "ShadowFluid (full)",
            exp32 / "fixed_shadowfluid",
            None,
            {
                "id": "forecasting_metrics.json",
                "ood_alpha": "ood_split_ood_alpha.json",
                "ood_structure": "ood_split_ood_structure.json",
            },
            "long_rollout_curves.json",
        ),
        (
            "FNO",
            exp32 / "fno",
            results_root / "bench_exp32" / "fno_exp32_paper_ref_multiseed" / "aggregate_summary.csv",
            {
                "id": "forecasting_metrics.json",
                "ood_alpha": "ood_split_ood_alpha.json",
                "ood_structure": "ood_split_ood_structure.json",
            },
            "long_rollout_curves.json",
        ),
        (
            "NSO (learned dictionary)",
            exp32 / "nso_relaxed",
            results_root / "bench_exp32" / "nso_exp32_relaxed_paper_ref_multiseed" / "aggregate_summary.csv",
            {
                "id": "forecasting_metrics.json",
                "ood_alpha": "ood_split_ood_alpha.json",
                "ood_structure": "ood_split_ood_structure.json",
            },
            "long_rollout_curves.json",
        ),
        (
            "ShadowFluid budgeted (K0=3)",
            exp32 / "shadowfluid_basek3",
            None,
            {
                "id": "forecasting_metrics.json",
                "ood_alpha": "ood_split_ood_alpha.json",
                "ood_structure": "ood_split_ood_structure.json",
            },
            "long_rollout_curves.json",
        ),
        (
            "ShadowFluid + residual (K0=3)",
            exp32 / "shadowfluid_residual_densityfirst_k3",
            results_root / "bench_exp32" / "shadowfluid_residual_densityfirst_k3_short_multiseed" / "aggregate_summary.csv",
            {
                "id": "test_metrics.json",
                "ood_alpha": "ood_split_ood_alpha.json",
                "ood_structure": "ood_split_ood_structure.json",
            },
            "long_rollout_curves.json",
        ),
        (
            "ShadowFluid + residual (K0=2)",
            exp32 / "shadowfluid_residual_densityfirst_k2_short",
            results_root / "bench_exp32" / "shadowfluid_residual_densityfirst_k2_short_multiseed" / "aggregate_summary.csv",
            {
                "id": "test_metrics.json",
                "ood_alpha": "ood_split_ood_alpha.json",
                "ood_structure": "ood_split_ood_structure.json",
            },
            None,
        ),
    ]
    for label, base_dir, agg, split_files, long_name in specs:
        dens = _read_density_metric(base_dir, split_files, aggregate_csv=agg)
        long_final = _read_long_final(base_dir, long_name) if long_name else float("nan")
        ref_rows.append(
            {
                "method": label,
                "id_density_rel_l2": dens["id"][0],
                "id_density_rel_l2_std": dens["id"][1],
                "ood_alpha_density_rel_l2": dens["ood_alpha"][0],
                "ood_alpha_density_rel_l2_std": dens["ood_alpha"][1],
                "ood_structure_density_rel_l2": dens["ood_structure"][0],
                "ood_structure_density_rel_l2_std": dens["ood_structure"][1],
                "long_rollout_final_density_rel_l2": long_final,
            }
        )
    return ref_rows


def build_exp64_pilot_table(results_root: Path) -> list[dict[str, Any]]:
    exp64 = results_root / "bench_exp64_pilot"
    rows: list[dict[str, Any]] = []
    specs = [
        (
            3,
            "shadowfluid_basek3",
            None,
            {
                "id": "forecasting_metrics.json",
                "ood_alpha": "ood_split_ood_alpha.json",
                "ood_structure": "ood_split_ood_structure.json",
            },
            "long_rollout_curves.json",
            "ShadowFluid budgeted (K0=3)",
        ),
        (
            3,
            "shadowfluid_residual_densityfirst_k3",
            results_root / "bench_exp64_pilot" / "shadowfluid_residual_densityfirst_k3_exp64_pilot_multiseed" / "aggregate_summary.csv",
            {
                "id": "test_metrics.json",
                "ood_alpha": "ood_split_ood_alpha.json",
                "ood_structure": "ood_split_ood_structure.json",
            },
            "long_rollout_curves.json",
            "ShadowFluid + residual (K0=3)",
        ),
        (
            2,
            "shadowfluid_basek2",
            None,
            {
                "id": "forecasting_metrics.json",
                "ood_alpha": "ood_split_ood_alpha.json",
                "ood_structure": "ood_split_ood_structure.json",
            },
            "long_rollout_curves.json",
            "ShadowFluid budgeted (K0=2)",
        ),
        (
            2,
            "shadowfluid_residual_densityfirst_k2",
            results_root / "bench_exp64_pilot" / "shadowfluid_residual_densityfirst_k2_exp64_pilot_multiseed" / "aggregate_summary.csv",
            {
                "id": "test_metrics.json",
                "ood_alpha": "ood_split_ood_alpha.json",
                "ood_structure": "ood_split_ood_structure.json",
            },
            "long_rollout_curves.json",
            "ShadowFluid + residual (K0=2)",
        ),
    ]
    for k0, run_name, agg, split_files, long_name, label in specs:
        dens = _read_density_metric(exp64 / run_name, split_files, aggregate_csv=agg)
        long_final = _read_long_final(exp64 / run_name, long_name)
        rows.append(
            {
                "base_K0": k0,
                "method": label,
                "id_density_rel_l2": dens["id"][0],
                "id_density_rel_l2_std": dens["id"][1],
                "ood_alpha_density_rel_l2": dens["ood_alpha"][0],
                "ood_alpha_density_rel_l2_std": dens["ood_alpha"][1],
                "ood_structure_density_rel_l2": dens["ood_structure"][0],
                "ood_structure_density_rel_l2_std": dens["ood_structure"][1],
                "long_rollout_final_density_rel_l2": long_final,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build paper-facing main experiment tables")
    p.add_argument("--results-root", default=str(ROOT / "results"))
    p.add_argument("--out-dir", default=str(ROOT / "results" / "comparison_tables"))
    args = p.parse_args(argv)

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp32_rows = build_exp32_main_table(results_root)
    ref_rows = build_reference_table(results_root)
    exp64_rows = build_exp64_pilot_table(results_root)

    _write_csv(out_dir / "paper_main_exp32_budget.csv", exp32_rows)
    _write_csv(out_dir / "paper_reference_baselines_exp32.csv", ref_rows)
    _write_csv(out_dir / "paper_main_exp64_pilot.csv", exp64_rows)

    print(out_dir / "paper_main_exp32_budget.csv")
    print(out_dir / "paper_reference_baselines_exp32.csv")
    print(out_dir / "paper_main_exp64_pilot.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
