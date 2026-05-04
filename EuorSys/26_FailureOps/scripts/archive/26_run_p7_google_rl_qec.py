#!/usr/bin/env python
"""Run P7 real Google RL QEC decoder-pathway paired comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from failureops.data_model import (
    P4_PAIRED_EFFECT_FIELDS,
    P4_PAIRING_VALIDATION_FIELDS,
    P7_BASELINE_FIELDS,
    P7_DECODER_INTERVENTION_FIELDS,
)
from failureops.google_rl_qec_adapter import load_google_rl_qec_records
from failureops.io_utils import write_csv_rows
from failureops.paired_metrics import summarize_paired_effects
from failureops.pairing import validate_intervention_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=(
            "data/raw/google_rl_qec/google_reinforcement_learning_qec/"
            "surface_code_traditional_calibration/Z/r010"
        ),
    )
    parser.add_argument(
        "--baseline-decoder",
        default="correlated_matching_decoder_with_si1000_prior",
    )
    parser.add_argument(
        "--intervened-decoder",
        default="tesseract_decoder_with_si1000_prior",
    )
    parser.add_argument("--max-shots", type=int, default=10000)
    parser.add_argument("--run-id", default="p7_google_rl_qec_surface_Z_r010")
    parser.add_argument(
        "--baseline-output",
        default="data/results/p7_google_rl_qec_surface_Z_r010_baseline_runs.csv",
    )
    parser.add_argument(
        "--intervention-output",
        default="data/results/p7_google_rl_qec_surface_Z_r010_decoder_interventions.csv",
    )
    parser.add_argument(
        "--validation-output",
        default="data/results/p7_google_rl_qec_surface_Z_r010_pairing_validation.csv",
    )
    parser.add_argument(
        "--effects-output",
        default="data/results/p7_google_rl_qec_surface_Z_r010_decoder_paired_effects.csv",
    )
    parser.add_argument("--num-bootstrap", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()

    baseline_rows, intervention_rows = load_google_rl_qec_records(
        args.data_dir,
        baseline_decoder_pathway=args.baseline_decoder,
        intervened_decoder_pathway=args.intervened_decoder,
        max_shots=args.max_shots,
        run_id=args.run_id,
    )
    validation_rows = validate_intervention_rows(intervention_rows)
    effect_rows = summarize_paired_effects(
        intervention_rows,
        num_bootstrap=args.num_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )

    write_csv_rows(args.baseline_output, baseline_rows, P7_BASELINE_FIELDS)
    write_csv_rows(args.intervention_output, intervention_rows, P7_DECODER_INTERVENTION_FIELDS)
    write_csv_rows(args.validation_output, validation_rows, P4_PAIRING_VALIDATION_FIELDS)
    write_csv_rows(args.effects_output, effect_rows, P4_PAIRED_EFFECT_FIELDS)
    print(f"wrote {len(baseline_rows)} P7 baseline rows to {args.baseline_output}")
    print(f"wrote {len(intervention_rows)} P7 decoder-intervention rows to {args.intervention_output}")
    print(f"wrote {len(validation_rows)} P7 pairing-validation rows to {args.validation_output}")
    print(f"wrote {len(effect_rows)} P7 paired-effect rows to {args.effects_output}")


if __name__ == "__main__":
    main()
