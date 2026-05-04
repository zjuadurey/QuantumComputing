#!/usr/bin/env python
"""Run P5a with external runtime trace import."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from failureops.data_model import (
    P4_BASELINE_FIELDS,
    P4_INTERVENTION_FIELDS,
    P4_PAIRED_EFFECT_FIELDS,
    P4_PAIRING_VALIDATION_FIELDS,
    P5_RUNTIME_TRACE_FIELDS,
)
from failureops.experiment_config import load_experiment_config
from failureops.io_utils import write_csv_rows
from failureops.manifest import write_manifest
from failureops.paired_metrics import summarize_paired_effects
from failureops.pairing import build_p4_intervention_row, event_record_hash, record_hash, validate_intervention_rows
from failureops.runtime_service import apply_p3_intervention, generate_p3_runs
from failureops.runtime_trace import (
    apply_trace_to_baseline_rows,
    export_runtime_trace_rows,
    load_runtime_trace,
    write_trace_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/p5a_trace_mode.yaml")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    outputs = config["outputs"]
    trace_config = config["runtime_trace"]
    baseline_rows = []
    for workload_id in config["workloads"]:
        for stress_level in config["stress_levels"]:
            for seed_index in range(config["num_seeds"]):
                seed = config["seed_start"] + seed_index * config["seed_stride"]
                baseline_rows.extend(
                    generate_p3_runs(
                        workload_id=workload_id,
                        stress_level=stress_level,
                        num_shots=config["num_shots_per_seed"],
                        seed=seed,
                        run_id=f"{config['experiment_id']}_{workload_id}_{stress_level}_{seed_index}",
                    )
                )

    trace_input = Path(str(trace_config.get("input") or trace_config["output"]))
    trace_output = Path(str(trace_config["output"]))
    if not trace_input.exists() and trace_config.get("export_proxy_trace_if_missing"):
        trace_rows = export_runtime_trace_rows(
            baseline_rows,
            trace_source="failureops_proxy_export_for_p5a",
        )
        write_trace_rows(trace_output, trace_rows, P5_RUNTIME_TRACE_FIELDS)
        trace_input = trace_output

    traces = load_runtime_trace(
        trace_input,
        ignore_run_id=bool(trace_config.get("ignore_run_id", False)),
    )
    baseline_rows, trace_stats = apply_trace_to_baseline_rows(
        baseline_rows,
        traces,
        missing_policy=str(trace_config.get("missing_policy", "error")),
        ignore_run_id=bool(trace_config.get("ignore_run_id", False)),
    )
    for row in baseline_rows:
        row["event_record_hash"] = event_record_hash(row)
        row["record_hash"] = record_hash(row)

    intervention_rows = []
    for baseline in baseline_rows:
        for intervention in config["interventions"]:
            intervened = apply_p3_intervention(baseline, intervention)
            intervention_rows.append(build_p4_intervention_row(baseline, intervened, intervention))

    validation_rows = validate_intervention_rows(intervention_rows)
    effect_rows = summarize_paired_effects(
        intervention_rows,
        num_bootstrap=int(config["bootstrap"]["num_resamples"]),
        bootstrap_seed=int(config["bootstrap"]["seed"]),
    )

    write_csv_rows(outputs["baseline"], baseline_rows, P4_BASELINE_FIELDS)
    write_csv_rows(outputs["interventions"], intervention_rows, P4_INTERVENTION_FIELDS)
    write_csv_rows(outputs["validation"], validation_rows, P4_PAIRING_VALIDATION_FIELDS)
    write_csv_rows(outputs["effects"], effect_rows, P4_PAIRED_EFFECT_FIELDS)
    row_counts = {
        "baseline": len(baseline_rows),
        "interventions": len(intervention_rows),
        "validation": len(validation_rows),
        "effects": len(effect_rows),
        **trace_stats,
    }
    manifest_outputs = dict(outputs)
    manifest_outputs["runtime_trace"] = str(trace_input)
    write_manifest(
        outputs["manifest"],
        config=config,
        command=sys.argv,
        outputs=manifest_outputs,
        row_counts=row_counts,
    )
    print(f"loaded runtime trace from {trace_input}")
    print(f"matched traces={trace_stats['matched_traces']} missing traces={trace_stats['missing_traces']}")
    print(f"wrote {len(baseline_rows)} P5a baseline rows to {outputs['baseline']}")
    print(f"wrote {len(intervention_rows)} P5a intervention rows to {outputs['interventions']}")
    print(f"wrote {len(validation_rows)} P5a validation rows to {outputs['validation']}")
    print(f"wrote {len(effect_rows)} P5a paired-effect rows to {outputs['effects']}")
    print(f"wrote P5a manifest to {outputs['manifest']}")


if __name__ == "__main__":
    main()

