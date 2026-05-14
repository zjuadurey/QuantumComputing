#!/usr/bin/env python
"""Regenerate per-shot paired decoder intervention CSVs for ALL 40 Google RL QEC conditions.

This reads the pre-computed .b8 decoder prediction files from Google's dataset
and writes per-shot paired transition records (rescued/induced labels per shot_id),
which are needed for the FailureOps-driven policy selection experiment (Step 2).

Output per condition: data/results/p7_per_shot/{condition}_decoder_interventions.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from failureops.data_model import P7_DECODER_INTERVENTION_FIELDS
from failureops.google_rl_qec_adapter import load_google_rl_qec_records
from failureops.io_utils import ensure_parent_dir, write_csv_rows

DATA_ROOT = Path("data/raw/google_rl_qec/google_reinforcement_learning_qec")
OUTPUT_DIR = Path("data/results/p7_per_shot")

CONTROL_MODES = ["reinforcement_learning", "traditional_calibration"]
BASES = ["X", "Z"]
CYCLES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

BASELINE_DECODER = "correlated_matching_decoder_with_si1000_prior"
INTERVENED_DECODER = "tesseract_decoder_with_si1000_prior"
MAX_SHOTS = 10000


def main() -> None:
    total = 0
    failed = []

    for mode in CONTROL_MODES:
        for basis in BASES:
            for cycles in CYCLES:
                data_dir = DATA_ROOT / f"surface_code_{mode}" / basis / f"r{cycles:03d}"
                condition_name = f"surface_code_{mode}_{basis}_r{cycles:03d}"
                run_id = f"p7_google_rl_qec_{condition_name}"

                output_path = OUTPUT_DIR / f"{condition_name}_decoder_interventions.csv"

                if output_path.exists():
                    print(f"  SKIP (exists): {condition_name}")
                    total += 1
                    continue

                if not (data_dir / "metadata.json").exists():
                    print(f"  SKIP (no data): {condition_name}")
                    failed.append(condition_name)
                    continue

                try:
                    baseline_rows, intervention_rows = load_google_rl_qec_records(
                        str(data_dir),
                        baseline_decoder_pathway=BASELINE_DECODER,
                        intervened_decoder_pathway=INTERVENED_DECODER,
                        max_shots=MAX_SHOTS,
                        run_id=run_id,
                    )
                except Exception as exc:
                    print(f"  FAIL {condition_name}: {exc}")
                    failed.append(condition_name)
                    continue

                ensure_parent_dir(output_path)
                write_csv_rows(str(output_path), intervention_rows, list(P7_DECODER_INTERVENTION_FIELDS))

                rescued = sum(1 for r in intervention_rows if r.get("rescued_failure") and str(r["rescued_failure"]) == "True")
                induced = sum(1 for r in intervention_rows if r.get("new_failure") and str(r["new_failure"]) == "True")
                n = len(intervention_rows)

                print(f"  OK   {condition_name}: {n} shots, rescued={rescued}, induced={induced}")
                total += 1

    print(f"\nDone: {total} conditions generated, {len(failed)} failed")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
