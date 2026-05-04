#!/usr/bin/env python
"""Run P5b with layered event records for noise interventions."""

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
)
from failureops.event_layers import NOISE_LAYER_MAP, apply_layered_noise_intervention, attach_event_layers
from failureops.experiment_config import load_experiment_config
from failureops.io_utils import write_csv_rows
from failureops.manifest import write_manifest
from failureops.paired_metrics import summarize_paired_effects
from failureops.pairing import build_p4_intervention_row, event_record_hash, record_hash, validate_intervention_rows
from failureops.runtime_service import apply_p3_intervention, generate_p3_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/p5b_layered_events.yaml")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    outputs = config["outputs"]
    baseline_rows = []
    intervention_rows = []
    for workload_id in config["workloads"]:
        for stress_level in config["stress_levels"]:
            for seed_index in range(config["num_seeds"]):
                seed = config["seed_start"] + seed_index * config["seed_stride"]
                run_rows = generate_p3_runs(
                    workload_id=workload_id,
                    stress_level=stress_level,
                    num_shots=config["num_shots_per_seed"],
                    seed=seed,
                    run_id=f"{config['experiment_id']}_{workload_id}_{stress_level}_{seed_index}",
                )
                run_rows = [attach_event_layers(row) for row in run_rows]
                for row in run_rows:
                    row["event_record_hash"] = event_record_hash(row)
                    row["record_hash"] = record_hash(row)
                baseline_rows.extend(run_rows)
                for baseline in run_rows:
                    for intervention in config["interventions"]:
                        if intervention in NOISE_LAYER_MAP:
                            intervened = apply_layered_noise_intervention(baseline, intervention)
                        else:
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
    write_manifest(
        outputs["manifest"],
        config=config,
        command=sys.argv,
        outputs=outputs,
        row_counts={
            "baseline": len(baseline_rows),
            "interventions": len(intervention_rows),
            "validation": len(validation_rows),
            "effects": len(effect_rows),
        },
    )
    print(f"wrote {len(baseline_rows)} P5b baseline rows to {outputs['baseline']}")
    print(f"wrote {len(intervention_rows)} P5b intervention rows to {outputs['interventions']}")
    print(f"wrote {len(validation_rows)} P5b validation rows to {outputs['validation']}")
    print(f"wrote {len(effect_rows)} P5b paired-effect rows to {outputs['effects']}")
    print(f"wrote P5b manifest to {outputs['manifest']}")


if __name__ == "__main__":
    main()

