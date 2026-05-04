#!/usr/bin/env python
"""Assemble a reviewer-facing EuroSys digest from the canonical P10 outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from failureops.data_model import P10_EUROSYS_DIGEST_FIELDS
from failureops.io_utils import read_csv_rows, write_csv_rows
from failureops.manifest import write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-audit", default="data/results/p10_claim_audit.csv")
    parser.add_argument("--robustness", default="data/results/p10_realdata_robustness.csv")
    parser.add_argument("--baseline", default="data/results/p10_baseline_comparison.csv")
    parser.add_argument("--runtime-summary", default="data/results/p10_runtime_deadline_summary.csv")
    parser.add_argument(
        "--prior-aggregate",
        default="data/results/p10_google_decoder_priors_prior_effects_aggregate.csv",
    )
    parser.add_argument(
        "--google-v2-aggregate",
        default="data/results/p10_google_rl_qec_v2_decoder_effect_aggregate.csv",
    )
    parser.add_argument("--output", default="data/results/p10_eurosys_digest.csv")
    parser.add_argument("--manifest-output", default="data/results/p10_eurosys_digest_manifest.json")
    args = parser.parse_args()

    required_paths = (
        args.claim_audit,
        args.robustness,
        args.baseline,
        args.runtime_summary,
        args.prior_aggregate,
        args.google_v2_aggregate,
    )
    for path in required_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"required digest input is missing: {path}")

    claim_rows = read_csv_rows(args.claim_audit)
    robustness_rows = read_csv_rows(args.robustness)
    baseline_rows = read_csv_rows(args.baseline)
    runtime_rows = read_csv_rows(args.runtime_summary)
    prior_aggregate_rows = read_csv_rows(args.prior_aggregate)
    google_v2_aggregate_rows = read_csv_rows(args.google_v2_aggregate)

    digest_rows = build_digest_rows(
        claim_rows=claim_rows,
        robustness_rows=robustness_rows,
        baseline_rows=baseline_rows,
        runtime_rows=runtime_rows,
        prior_aggregate_rows=prior_aggregate_rows,
        google_v2_aggregate_rows=google_v2_aggregate_rows,
    )

    write_csv_rows(args.output, digest_rows, P10_EUROSYS_DIGEST_FIELDS)
    write_manifest(
        args.manifest_output,
        config={
            "experiment_id": "p10_eurosys_digest",
            "inputs": {
                "claim_audit": args.claim_audit,
                "robustness": args.robustness,
                "baseline": args.baseline,
                "runtime_summary": args.runtime_summary,
                "prior_aggregate": args.prior_aggregate,
                "google_v2_aggregate": args.google_v2_aggregate,
            },
        },
        command=sys.argv,
        outputs={"digest": args.output},
        row_counts={"digest": len(digest_rows)},
    )
    print(f"wrote {len(digest_rows)} EuroSys digest rows to {args.output}")
    print(f"wrote EuroSys digest manifest to {args.manifest_output}")


def build_digest_rows(
    *,
    claim_rows: list[dict[str, str]],
    robustness_rows: list[dict[str, str]],
    baseline_rows: list[dict[str, str]],
    runtime_rows: list[dict[str, str]],
    prior_aggregate_rows: list[dict[str, str]],
    google_v2_aggregate_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    claim_by_id = {row["claim_id"]: row for row in claim_rows}
    robustness_by_id = {row["check_id"]: row for row in robustness_rows}
    baseline_by_method = {row["method"]: row for row in baseline_rows}
    runtime_by_deadline = {row["deadline_us"]: row for row in runtime_rows}
    prior_by_variant = {row["intervened_prior"]: row for row in prior_aggregate_rows}
    v2_all = find_row(google_v2_aggregate_rows, "group_by", "all")
    v2_control = {row["group_value"]: row for row in google_v2_aggregate_rows if row["group_by"] == "control_mode"}
    v2_basis = {row["group_value"]: row for row in google_v2_aggregate_rows if row["group_by"] == "basis"}

    c1 = claim_by_id["C1"]
    c2 = claim_by_id["C2"]
    c3 = claim_by_id["C3"]
    c4 = claim_by_id["C4"]
    c6 = claim_by_id["C6"]
    c6b = claim_by_id["C6b"]
    r1 = robustness_by_id["R1"]
    r4 = robustness_by_id["R4"]
    r5 = robustness_by_id["R5"]
    r7 = robustness_by_id["R7"]
    plain_lfr = baseline_by_method["Plain baseline LFR"]
    static_burden = baseline_by_method["Static detector burden"]
    deadline_4 = runtime_by_deadline["4.000000"]
    deadline_7 = runtime_by_deadline["7.000000"]
    prior_significant = r5["supporting_value"].split("=", 1)[-1]

    prior_mean_summary = ";".join(
        f"{name}={prior_by_variant[name]['mean_paired_delta_lfr']}"
        for name in (
            "dem_correlations",
            "dem_rl_isolated_correlated_matching",
            "dem_rl_shared_correlated_matching",
        )
    )

    return [
        {
            "story_id": "S1",
            "story_section": "Main real-record result",
            "claim": c1["claim"],
            "key_metric": c1["primary_metric"],
            "key_result": f"{c1['observed_value']};net_rescuing={r1['num_units']}/{r1['num_units']}",
            "why_it_matters": "Treats logical failure as an intervention-sensitive system event on public shot-level records.",
            "scope_boundary": c1["limitation"],
            "source_artifact": c1["source_artifact"],
        },
        {
            "story_id": "S2",
            "story_section": "Methodology necessity",
            "claim": c2["claim"],
            "key_metric": "mean unpaired/paired bootstrap std ratio on original and expanded corpora",
            "key_result": f"orig={r4['value']};v2={r7['value']}",
            "why_it_matters": "Shows that pairing is a measurement requirement, not just an implementation detail.",
            "scope_boundary": c2["limitation"],
            "source_artifact": f"{r4['source_artifact']};{r7['source_artifact']}",
        },
        {
            "story_id": "S3",
            "story_section": "Baseline semantics gap",
            "claim": c3["claim"],
            "key_metric": "baseline-vs-FailureOps ranking agreement plus missing rescued/induced semantics",
            "key_result": (
                f"plain_lfr_spearman={plain_lfr['spearman_with_failureops']};"
                f"static_spearman={static_burden['spearman_with_failureops']}"
            ),
            "why_it_matters": "Separates ranking difficult conditions from attributing which controllable factor changed failure behavior.",
            "scope_boundary": c3["limitation"],
            "source_artifact": c3["source_artifact"],
        },
        {
            "story_id": "S4",
            "story_section": "Runtime replay closure",
            "claim": c4["claim"],
            "key_metric": "deadline-induced paired delta logical-failure rate",
            "key_result": (
                f"4us_miss={deadline_4['deadline_miss_rate']};4us_delta={deadline_4['paired_delta_lfr']};"
                f"7us_delta={deadline_7['paired_delta_lfr']}"
            ),
            "why_it_matters": "Connects measured decoder service time to a runtime intervention that changes logical outcomes on the same shots.",
            "scope_boundary": c4["limitation"],
            "source_artifact": c4["source_artifact"],
        },
        {
            "story_id": "S5",
            "story_section": "External real-record replication",
            "claim": c6["claim"],
            "key_metric": c6["primary_metric"],
            "key_result": c6["observed_value"],
            "why_it_matters": "Shows that the main paired decoder-pathway result is not confined to one Google RL QEC release.",
            "scope_boundary": c6["limitation"],
            "source_artifact": c6["source_artifact"],
        },
        {
            "story_id": "S6",
            "story_section": "Expanded same-family stability",
            "claim": c6b["claim"],
            "key_metric": "net-rescuing fraction, subgroup means, and scope-wise Holm count on the v2 corpus",
            "key_result": (
                f"{c6b['observed_value']};all_mean={v2_all['mean_paired_delta_lfr']}"
            ),
            "why_it_matters": "Strengthens the story from a focused public matrix to a broader same-family real-record corpus without changing the thesis.",
            "scope_boundary": c6b["limitation"],
            "source_artifact": c6b["source_artifact"],
        },
        {
            "story_id": "S7",
            "story_section": "Corpus-level prior interventions",
            "claim": "FailureOps scales from a focused prior sweep to a corpus-level prior intervention study on public detector records.",
            "key_metric": "bootstrap-CI support rate plus mean paired delta by prior family",
            "key_result": f"ci_nonzero={prior_significant}/{r5['num_units']};means={prior_mean_summary}",
            "why_it_matters": "Turns decoder-prior analysis into a systems-style intervention family rather than a single anecdotal condition.",
            "scope_boundary": r5["limitation"],
            "source_artifact": r5["source_artifact"],
        },
    ]


def find_row(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise KeyError(f"missing row for {key}={value}")


if __name__ == "__main__":
    main()
