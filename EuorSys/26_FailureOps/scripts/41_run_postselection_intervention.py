#!/usr/bin/env python
"""Post-selection intervention: apply detector-burden percentile thresholds
and compute rescued/induced transition changes.

Uses per-condition aggregate data from p7_5_rescue_induction_features.csv
to estimate what happens when high-burden shots are discarded via
post-selection, using the per-transition-class mean detector burden
and shot counts.

Outputs:
  data/results/p10_postselection_intervention.csv  -- per-threshold results
  data/results/p10_postselection_summary.csv       -- summary table
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from failureops.io_utils import ensure_parent_dir, fmt_float, write_csv_rows


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    features_path = repo_root / "data/results/p7_5_rescue_induction_features.csv"
    output_path = repo_root / "data/results/p10_postselection_intervention.csv"
    summary_path = repo_root / "data/results/p10_postselection_summary.csv"

    with open(features_path) as f:
        rows = list(csv.DictReader(f))

    # Group by condition (workload_id) and transition_class
    by_condition = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        wid = r["workload_id"]
        tc = r["transition_class"]
        n = int(r["num_shots"])
        burden = float(r["mean_detector_event_count"])
        by_condition[wid][tc] = {"shots": n, "burden": burden}

    thresholds = [0.95, 0.90, 0.80]
    results = []

    for thresh in thresholds:
        total_discarded_rescued = 0
        total_discarded_unchanged = 0
        total_rescued = 0
        total_unchanged = 0
        total_induced = 0
        total_pairs = 0

        for wid, tcs in by_condition.items():
            rescued = tcs.get("rescued", {"shots": 0, "burden": 0})
            unchanged = tcs.get("unchanged_failure", {"shots": 0, "burden": 0})
            induced = tcs.get("induced", {"shots": 0, "burden": 0})
            success = tcs.get("unchanged_success", {"shots": 0, "burden": 0})

            total_rescued += rescued["shots"]
            total_unchanged += unchanged["shots"]
            total_induced += induced["shots"]
            total_pairs += rescued["shots"] + unchanged["shots"] + induced["shots"] + success["shots"]

            # Estimate global burden threshold using weighted average
            all_burdens = []
            all_weights = []
            for tc_data, tc_name in [(rescued, "rescued"), (unchanged, "unchanged"),
                                       (induced, "induced"), (success, "success")]:
                if tc_data["shots"] > 0:
                    all_burdens.append(tc_data["burden"])
                    all_weights.append(tc_data["shots"])

            if not all_burdens:
                continue

            # Compute the global thresh percentile burden across all shots
            # Using weighted percentile approximation
            sorted_pairs = sorted(zip(all_burdens, all_weights), key=lambda x: x[0])
            cumsum = 0
            total_w = sum(all_weights)
            thresh_burden = sorted_pairs[-1][0]  # default: max
            for b, w in sorted_pairs:
                cumsum += w
                if cumsum / total_w >= thresh:
                    thresh_burden = b
                    break

            # Shots with burden > threshold_burden are discarded
            # Use per-transition-class burden to estimate discard fraction
            # Conservative: use the threshold burden directly
            # Since we only have mean burden, not distribution, we estimate
            # discard rates assuming normal-ish distribution
            # For a cleaner estimate: use the weighted percentile directly
            # All shots whose mean class burden exceeds threshold are discarded
            for tc_data, tc_name in [(rescued, "rescued"), (unchanged, "unchanged_failure")]:
                if tc_data["burden"] > thresh_burden and tc_data["shots"] > 0:
                    # Estimate fraction discarded based on how far above threshold
                    # Conservative linear estimate
                    excess = tc_data["burden"] - thresh_burden
                    # If burden exceeds threshold, estimate ~50-90% discarded
                    discard_frac = min(0.95, 0.1 + excess / max(thresh_burden, 1) * 0.5)
                    if tc_name == "rescued":
                        total_discarded_rescued += int(tc_data["shots"] * discard_frac)
                    else:
                        total_discarded_unchanged += int(tc_data["shots"] * discard_frac)

        # Compute post-selection metrics
        retained_rescued = total_rescued - total_discarded_rescued
        retained_unchanged = total_unchanged - total_discarded_unchanged
        rescued_frac = retained_rescued / (retained_rescued + retained_unchanged) if (retained_rescued + retained_unchanged) > 0 else 0
        base_rescued_frac = total_rescued / (total_rescued + total_unchanged) if (total_rescued + total_unchanged) > 0 else 0

        results.append({
            "threshold_percentile": f"{int(thresh*100)}th",
            "discarded_rescued_pct": fmt_float(total_discarded_rescued / max(total_rescued, 1) * 100),
            "discarded_unchanged_pct": fmt_float(total_discarded_unchanged / max(total_unchanged, 1) * 100),
            "rescued_fraction_before": fmt_float(base_rescued_frac),
            "rescued_fraction_after": fmt_float(rescued_frac),
            "rescued_fraction_change": fmt_float(rescued_frac - base_rescued_frac),
            "total_rescued": str(total_rescued),
            "total_unchanged_failure": str(total_unchanged),
            "total_pairs": str(total_pairs),
        })

    write_csv_rows(output_path, results, list(results[0].keys()) if results else [])
    print(f"Wrote {output_path}")

    # Summary
    summary = [
        {"threshold": r["threshold_percentile"],
         "discarded_rescued": r["discarded_rescued_pct"],
         "discarded_unchanged": r["discarded_unchanged_pct"],
         "rescued_fraction_before": r["rescued_fraction_before"],
         "rescued_fraction_after": r["rescued_fraction_after"]}
        for r in results
    ]
    write_csv_rows(summary_path, summary, list(summary[0].keys()) if summary else [])
    print(f"Wrote {summary_path}")

    # Print summary
    print("\nPost-Selection Results:")
    for r in results:
        print(f"  {r['threshold_percentile']}: discarded rescued={r['discarded_rescued_pct']}%, "
              f"unchanged={r['discarded_unchanged_pct']}%, "
              f"rescued_frac: {r['rescued_fraction_before']} -> {r['rescued_fraction_after']}")


if __name__ == "__main__":
    main()
