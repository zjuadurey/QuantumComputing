from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shiftflow.bench.runner import (  # noqa: E402
    build_overridden_config,
    run_evaluation_from_config,
    run_training_from_config,
)


DEFAULT_SEEDS = [20260505, 20260515, 20260525]
DEFAULT_OOD_COLUMNS = ["split_ood_alpha", "split_ood_structure"]


def _parse_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_strings(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _safe_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0 if values else float("nan")
    return float(statistics.pstdev(values))


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float))


def _aggregate_scalar_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    metric_keys: list[str] = []
    for key, value in rows[0].items():
        if _is_number(value):
            metric_keys.append(key)

    out: list[dict[str, Any]] = []
    for key in metric_keys:
        vals = [
            float(row[key])
            for row in rows
            if key in row and _is_number(row[key]) and not math.isnan(float(row[key]))
        ]
        out.append(
            {
                "metric": key,
                "mean": _safe_mean(vals),
                "std": _safe_std(vals),
                "num_seeds": len(vals),
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _seed_output_dir(base_output_dir: str, label: str, seed: int) -> str:
    base = Path(base_output_dir)
    return str((base.parent / f"{label}_multiseed" / f"seed_{seed}").resolve())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run one config across multiple random seeds and aggregate results")
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    p.add_argument("--label", default=None)
    p.add_argument("--ood-columns", default=",".join(DEFAULT_OOD_COLUMNS))
    p.add_argument("--evaluate-long-rollout", action="store_true")
    p.add_argument("--reuse-existing", action="store_true", default=True)
    args = p.parse_args(argv)

    seeds = _parse_ints(args.seeds)
    ood_columns = _parse_strings(args.ood_columns)
    base_cfg = build_overridden_config(args.config, {})
    label = args.label or Path(args.config).stem
    multiseed_root = Path(_seed_output_dir(base_cfg["output_dir"], label, seed=0)).parent
    multiseed_root.mkdir(parents=True, exist_ok=True)

    per_seed_rows: list[dict[str, Any]] = []
    evaluation_records: dict[str, list[dict[str, Any]]] = {"test": []}
    for col in ood_columns:
        evaluation_records[col] = []

    for seed in seeds:
        override = {
            "seed": int(seed),
            "output_dir": _seed_output_dir(base_cfg["output_dir"], label, seed),
        }
        cfg = build_overridden_config(args.config, override)
        out_dir = Path(cfg["output_dir"])
        test_path = out_dir / "test_metrics.json"
        if args.reuse_existing and test_path.exists():
            print(f"[multiseed] reuse seed={seed} -> {cfg['output_dir']}", flush=True)
            test_metrics = json.loads(test_path.read_text())
        else:
            print(f"[multiseed] train seed={seed} -> {cfg['output_dir']}", flush=True)
            test_metrics = run_training_from_config(cfg, config_path=args.config)
        per_seed_rows.append(
            {
                "seed": int(seed),
                "split": "test",
                **{k: v for k, v in test_metrics.items() if not isinstance(v, list)},
            }
        )
        evaluation_records["test"].append(test_metrics)

        for split_column in ood_columns:
            output_name = f"ood_{split_column}.json"
            eval_path = out_dir / output_name
            if args.reuse_existing and eval_path.exists():
                metrics = json.loads(eval_path.read_text())
            else:
                metrics = run_evaluation_from_config(
                    cfg,
                    config_path=args.config,
                    split_column=split_column,
                    split_values=["test"],
                    output_name=output_name,
                )
            per_seed_rows.append(
                {
                    "seed": int(seed),
                    "split": split_column,
                    **{k: v for k, v in metrics.items() if not isinstance(v, list)},
                }
            )
            evaluation_records[split_column].append(metrics)

        if args.evaluate_long_rollout:
            metrics = run_evaluation_from_config(
                cfg,
                config_path=args.config,
                split_column="split_id",
                split_values=["test"],
                output_name="long_rollout_metrics.json",
            )
            with (Path(cfg["output_dir"]) / "long_rollout_curves.json").open("w") as f:
                json.dump(
                    {
                        "density_rel_l2_curve": metrics.get("density_rel_l2_curve", []),
                        "lowfreq_spectral_rel_l2_curve": metrics.get("lowfreq_spectral_rel_l2_curve", []),
                    },
                    f,
                    indent=2,
                )

    per_seed_path = multiseed_root / "per_seed_summary.csv"
    _write_csv(per_seed_path, per_seed_rows)

    aggregate_rows: list[dict[str, Any]] = []
    aggregate_json: dict[str, Any] = {
        "config": str(args.config),
        "label": label,
        "seeds": seeds,
        "splits": {},
    }
    for split_name, rows in evaluation_records.items():
        agg = _aggregate_scalar_rows(
            [{k: v for k, v in row.items() if not isinstance(v, list)} for row in rows]
        )
        for row in agg:
            aggregate_rows.append({"split": split_name, **row})
        aggregate_json["splits"][split_name] = agg

    aggregate_csv_path = multiseed_root / "aggregate_summary.csv"
    aggregate_json_path = multiseed_root / "aggregate_summary.json"
    _write_csv(aggregate_csv_path, aggregate_rows)
    aggregate_json_path.write_text(json.dumps(aggregate_json, indent=2))

    manifest = {
        "config": str(args.config),
        "label": label,
        "seeds": seeds,
        "multiseed_root": str(multiseed_root),
        "per_seed_summary": str(per_seed_path),
        "aggregate_summary_csv": str(aggregate_csv_path),
        "aggregate_summary_json": str(aggregate_json_path),
    }
    (multiseed_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[multiseed] wrote aggregate summary: {aggregate_csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
