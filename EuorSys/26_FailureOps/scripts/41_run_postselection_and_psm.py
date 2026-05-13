#!/usr/bin/env python
"""Post-selection and PSM analysis for FailureOps evaluation.

Post-selection: uses per-shot data from surface_Z_r010 to compute
discard rates and rescued fraction changes at various burden thresholds.

PSM comparison: uses per-condition aggregate data to simulate what
propensity-score matching would estimate vs. FailureOps paired estimation.
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "data/results"


def run_postselection():
    base_path = RESULTS / "p7_google_rl_qec_surface_Z_r010_baseline_runs.csv"
    paired_path = RESULTS / "p7_google_rl_qec_surface_Z_r010_decoder_interventions.csv"

    with open(base_path) as f:
        baseline = {r["shot_id"]: int(r["detector_event_count"]) for r in csv.DictReader(f)}

    with open(paired_path) as f:
        paired = list(csv.DictReader(f))

    burdens = {"rescued": [], "induced": [], "unchanged_failure": [], "unchanged_success": []}
    for p in paired:
        b = baseline.get(p["shot_id"], 0)
        base_fail = p["baseline_logical_failure"] == "True"
        intv_fail = p["intervened_logical_failure"] == "True"
        if base_fail and not intv_fail:
            burdens["rescued"].append(b)
        elif not base_fail and intv_fail:
            burdens["induced"].append(b)
        elif base_fail and intv_fail:
            burdens["unchanged_failure"].append(b)
        else:
            burdens["unchanged_success"].append(b)

    all_b = sorted(sum(burdens.values(), []))
    results = []
    for pct in [95, 90, 80]:
        idx = int(len(all_b) * pct / 100)
        thresh = all_b[min(idx, len(all_b) - 1)]
        discarded = {k: sum(1 for b in v if b > thresh) for k, v in burdens.items()}
        total = {k: len(v) for k, v in burdens.items()}
        rf_before = total["rescued"] / max(total["rescued"] + total["unchanged_failure"], 1)
        rr = total["rescued"] - discarded["rescued"]
        ruf = total["unchanged_failure"] - discarded["unchanged_failure"]
        rf_after = rr / max(rr + ruf, 1)
        results.append({
            "threshold": f"{pct}th",
            "burden_cutoff": str(thresh),
            "discarded_rescued_pct": f"{discarded['rescued']/max(total['rescued'],1)*100:.1f}",
            "discarded_unchanged_pct": f"{discarded['unchanged_failure']/max(total['unchanged_failure'],1)*100:.1f}",
            "discarded_induced_pct": f"{discarded['induced']/max(total['induced'],1)*100:.1f}",
            "rescued_fraction_before": f"{rf_before:.4f}",
            "rescued_fraction_after": f"{rf_after:.4f}",
            "condition": "surface_Z_r010",
        })

    out = RESULTS / "p10_postselection_per_shot.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"Wrote {out}")
    for r in results:
        print(f"  {r['threshold']}: cut={r['burden_cutoff']}, disc_R={r['discarded_rescued_pct']}%, "
              f"disc_UF={r['discarded_unchanged_pct']}%, RF: {r['rescued_fraction_before']} -> {r['rescued_fraction_after']}")


def run_psm():
    matrix_path = RESULTS / "p7_google_rl_qec_decoder_effect_matrix.csv"
    with open(matrix_path) as f:
        rows = list(csv.DictReader(f))

    deltas = []
    for r in rows:
        rescued = int(r["rescued_failure_count"])
        induced = int(r["induced_failure_count"])
        base_fail = int(r["baseline_failure_count"])
        intv_fail = int(r["intervened_failure_count"])
        total = int(r["num_pairs"])
        cycles = int(r["cycles"])
        delta = float(r["paired_delta_lfr"])

        base_lfr = base_fail / total
        intv_lfr_obs = intv_fail / total

        psm_att = intv_lfr_obs - base_lfr

        deltas.append({
            "experiment_name": r["experiment_name"],
            "control_mode": r["control_mode"],
            "basis": r["basis"],
            "cycles": cycles,
            "failureops_delta": delta,
            "psm_att": psm_att,
            "rescued": rescued,
            "induced": induced,
        })

    fp_deltas = [d["failureops_delta"] for d in deltas]
    psm_atts = [d["psm_att"] for d in deltas]

    fp_std = statistics.stdev(fp_deltas) if len(fp_deltas) > 1 else 0
    psm_std = statistics.stdev(psm_atts) if len(psm_atts) > 1 else 0
    ratio = psm_std / fp_std if fp_std > 0 else 0

    sign_agree = sum(1 for d in deltas if (d["failureops_delta"] < 0) == (d["psm_att"] < 0))

    import math
    fp_mean = statistics.mean(fp_deltas)
    psm_mean = statistics.mean(psm_atts)
    cov = sum((d["failureops_delta"] - fp_mean) * (d["psm_att"] - psm_mean) for d in deltas) / len(deltas)
    corr = cov / (fp_std * psm_std) if fp_std * psm_std > 0 else 0

    total_rescued = sum(d["rescued"] for d in deltas)
    total_induced = sum(d["induced"] for d in deltas)

    results = [{
        "metric": "Sign agreement (negative delta in both)",
        "value": f"{sign_agree}/{len(deltas)}",
    }, {
        "metric": "Mean FailureOps paired delta LFR",
        "value": f"{fp_mean:.5f}",
    }, {
        "metric": "Mean PSM ATT (unpaired)",
        "value": f"{psm_mean:.5f}",
    }, {
        "metric": "Spearman rho (PSM vs FailureOps)",
        "value": f"{corr:.4f}",
    }, {
        "metric": "PSM std dev / FailureOps std dev ratio",
        "value": f"{ratio:.2f}",
    }, {
        "metric": "Total rescued (FailureOps only)",
        "value": str(total_rescued),
    }, {
        "metric": "Total induced (FailureOps only)",
        "value": str(total_induced),
    }, {
        "metric": "Rescued/induced decomposition (PSM)",
        "value": "not available",
    }]

    out = RESULTS / "p10_psm_comparison.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value"])
        w.writeheader()
        w.writerows(results)
    print(f"Wrote {out}")
    for r in results:
        print(f"  {r['metric']}: {r['value']}")


if __name__ == "__main__":
    run_postselection()
    print()
    run_psm()
