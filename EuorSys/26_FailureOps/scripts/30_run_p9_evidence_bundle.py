#!/usr/bin/env python
"""Build a paper-facing evidence bundle from P7.5 real-data and P8 runtime outputs."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from failureops.data_model import (
    P9_CLAIM_TABLE_FIELDS,
    P9_EVIDENCE_SUMMARY_FIELDS,
    P9_PAIRING_VALIDITY_FIELDS,
    P9_REALNESS_TABLE_FIELDS,
)
from failureops.io_utils import ensure_parent_dir, fmt_float, parse_int, read_csv_rows, write_csv_rows
from failureops.manifest import write_manifest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/failureops-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--effect-matrix",
        default="data/results/p7_google_rl_qec_decoder_effect_matrix.csv",
    )
    parser.add_argument(
        "--variance",
        default="data/results/p7_5_paired_vs_unpaired_variance.csv",
    )
    parser.add_argument(
        "--features",
        default="data/results/p7_5_rescue_induction_features.csv",
    )
    parser.add_argument(
        "--prior-effects",
        default="data/results/p7_5_decoder_prior_interventions.csv",
    )
    parser.add_argument(
        "--p8-runtime-summary",
        default="data/results/p8_decoder_runtime_summary.csv",
    )
    parser.add_argument("--claim-output", default="data/results/p9_claim_table.csv")
    parser.add_argument("--realness-output", default="data/results/p9_realness_table.csv")
    parser.add_argument("--pairing-output", default="data/results/p9_pairing_validity_table.csv")
    parser.add_argument("--summary-output", default="data/results/p9_evidence_summary.csv")
    parser.add_argument("--manifest-output", default="data/results/p9_evidence_bundle_manifest.json")
    parser.add_argument("--figure-output", default="figures/p9_evidence_bundle.png")
    args = parser.parse_args()

    required = [
        args.effect_matrix,
        args.variance,
        args.features,
        args.prior_effects,
        args.p8_runtime_summary,
    ]
    for path in required:
        if not Path(path).exists():
            raise FileNotFoundError(f"required P9 input is missing: {path}")

    effect_rows = read_csv_rows(args.effect_matrix)
    variance_rows = read_csv_rows(args.variance)
    feature_rows = read_csv_rows(args.features)
    prior_rows = read_csv_rows(args.prior_effects)
    p8_rows = read_csv_rows(args.p8_runtime_summary)

    effect_summary = summarize_effect_matrix(effect_rows)
    variance_summary = summarize_variance(variance_rows)
    feature_summary = summarize_features(feature_rows)
    prior_summary = summarize_prior_effects(prior_rows)
    p8_summary = summarize_p8(p8_rows)

    summary_rows = build_evidence_summary(
        effect_summary=effect_summary,
        variance_summary=variance_summary,
        p8_summary=p8_summary,
        args=args,
    )
    claim_rows = build_claim_table(
        effect_summary=effect_summary,
        variance_summary=variance_summary,
        feature_summary=feature_summary,
        prior_summary=prior_summary,
        p8_summary=p8_summary,
        args=args,
    )
    realness_rows = build_realness_table(args)
    pairing_rows = build_pairing_table(effect_summary, p8_summary, args)

    write_csv_rows(args.claim_output, claim_rows, P9_CLAIM_TABLE_FIELDS)
    write_csv_rows(args.realness_output, realness_rows, P9_REALNESS_TABLE_FIELDS)
    write_csv_rows(args.pairing_output, pairing_rows, P9_PAIRING_VALIDITY_FIELDS)
    write_csv_rows(args.summary_output, summary_rows, P9_EVIDENCE_SUMMARY_FIELDS)
    plot_evidence_bundle(
        effect_rows=effect_rows,
        variance_rows=variance_rows,
        feature_summary=feature_summary,
        p8_summary=p8_summary,
        output=args.figure_output,
    )
    write_manifest(
        args.manifest_output,
        config={
            "experiment_id": "p9_evidence_bundle",
            "inputs": {
                "effect_matrix": args.effect_matrix,
                "variance": args.variance,
                "features": args.features,
                "prior_effects": args.prior_effects,
                "p8_runtime_summary": args.p8_runtime_summary,
            },
        },
        command=sys.argv,
        outputs={
            "claims": args.claim_output,
            "realness": args.realness_output,
            "pairing": args.pairing_output,
            "summary": args.summary_output,
            "figure": args.figure_output,
        },
        row_counts={
            "claims": len(claim_rows),
            "realness": len(realness_rows),
            "pairing": len(pairing_rows),
            "summary": len(summary_rows),
        },
    )

    print(f"wrote {len(claim_rows)} P9 claim rows to {args.claim_output}")
    print(f"wrote {len(realness_rows)} P9 realness rows to {args.realness_output}")
    print(f"wrote {len(pairing_rows)} P9 pairing-validity rows to {args.pairing_output}")
    print(f"wrote {len(summary_rows)} P9 summary rows to {args.summary_output}")
    print(f"wrote P9 evidence figure to {args.figure_output}")
    print(f"wrote P9 manifest to {args.manifest_output}")


def summarize_effect_matrix(rows: list[dict[str, str]]) -> dict[str, object]:
    total_pairs = sum(parse_int(row["valid_pairs"]) for row in rows)
    baseline_failures = sum(parse_int(row["baseline_failure_count"]) for row in rows)
    intervened_failures = sum(parse_int(row["intervened_failure_count"]) for row in rows)
    rescued = sum(parse_int(row["rescued_failure_count"]) for row in rows)
    induced = sum(parse_int(row["induced_failure_count"]) for row in rows)
    invalid = sum(parse_int(row["invalid_pairs"]) for row in rows)
    strongest = min(rows, key=lambda row: float(row["paired_delta_lfr"])) if rows else {}
    deltas = [float(row["paired_delta_lfr"]) for row in rows]
    return {
        "num_conditions": len(rows),
        "num_pairs": total_pairs,
        "baseline_lfr": baseline_failures / total_pairs if total_pairs else 0.0,
        "intervened_lfr": intervened_failures / total_pairs if total_pairs else 0.0,
        "paired_delta_lfr": (induced - rescued) / total_pairs if total_pairs else 0.0,
        "rescued_failure_count": rescued,
        "induced_failure_count": induced,
        "invalid_pairs": invalid,
        "violation_rate": invalid / (total_pairs + invalid) if total_pairs + invalid else 0.0,
        "strongest_condition": strongest.get("workload_id", ""),
        "strongest_paired_delta_lfr": float(strongest.get("paired_delta_lfr", 0.0) or 0.0),
        "min_paired_delta_lfr": min(deltas) if deltas else 0.0,
        "max_paired_delta_lfr": max(deltas) if deltas else 0.0,
    }


def summarize_variance(rows: list[dict[str, str]]) -> dict[str, object]:
    ratios = [float(row["std_ratio_unpaired_over_paired"]) for row in rows]
    paired_stds = [float(row["paired_bootstrap_std"]) for row in rows]
    unpaired_stds = [float(row["unpaired_bootstrap_std"]) for row in rows]
    return {
        "num_conditions": len(rows),
        "mean_ratio": statistics.fmean(ratios) if ratios else 0.0,
        "min_ratio": min(ratios) if ratios else 0.0,
        "max_ratio": max(ratios) if ratios else 0.0,
        "mean_paired_std": statistics.fmean(paired_stds) if paired_stds else 0.0,
        "mean_unpaired_std": statistics.fmean(unpaired_stds) if unpaired_stds else 0.0,
    }


def summarize_features(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for transition in ("rescued", "induced", "unchanged_failure", "unchanged_success"):
        group = [row for row in rows if row["transition_class"] == transition]
        weight = sum(parse_int(row["num_shots"]) for row in group)
        out[transition] = {
            "num_shots": float(weight),
            "mean_detector_event_count": weighted_mean(group, "mean_detector_event_count"),
            "burst_fraction": weighted_mean(group, "burst_fraction"),
        }
    return out


def summarize_prior_effects(rows: list[dict[str, str]]) -> dict[str, object]:
    if not rows:
        return {
            "num_interventions": 0,
            "strongest_prior": "",
            "strongest_delta": 0.0,
            "largest_regression_prior": "",
            "largest_regression_delta": 0.0,
        }
    strongest = min(rows, key=lambda row: float(row["paired_delta_lfr"]))
    regression = max(rows, key=lambda row: float(row["paired_delta_lfr"]))
    return {
        "num_interventions": len(rows),
        "strongest_prior": strongest["intervened_prior"],
        "strongest_delta": float(strongest["paired_delta_lfr"]),
        "largest_regression_prior": regression["intervened_prior"],
        "largest_regression_delta": float(regression["paired_delta_lfr"]),
    }


def summarize_p8(rows: list[dict[str, str]]) -> dict[str, object]:
    if not rows:
        return {}
    row = rows[0]
    return {
        "num_shots": parse_int(row["num_shots"]),
        "mean_decode_us": float(row["mean_per_shot_decode_time_us"]),
        "p95_decode_us": float(row["p95_per_shot_decode_time_us"]),
        "max_decode_us": float(row["max_per_shot_decode_time_us"]),
        "throughput": float(row["throughput_shots_per_second"]),
        "prediction_mismatch_rate": float(row["prediction_mismatch_rate"]),
        "source_data_dir": row["source_data_dir"],
        "decoder_pathway": row["decoder_pathway"],
    }


def build_evidence_summary(
    *,
    effect_summary: dict[str, object],
    variance_summary: dict[str, object],
    p8_summary: dict[str, object],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    return [
        {
            "summary_id": "p9_real_record_decoder_pathway",
            "evidence_source": "P7/P7.5 real detector records",
            "num_conditions": effect_summary["num_conditions"],
            "num_pairs": effect_summary["num_pairs"],
            "baseline_lfr": fmt_float(float(effect_summary["baseline_lfr"])),
            "intervened_lfr": fmt_float(float(effect_summary["intervened_lfr"])),
            "paired_delta_lfr": fmt_float(float(effect_summary["paired_delta_lfr"])),
            "rescued_failure_count": effect_summary["rescued_failure_count"],
            "induced_failure_count": effect_summary["induced_failure_count"],
            "strongest_condition": effect_summary["strongest_condition"],
            "strongest_paired_delta_lfr": fmt_float(float(effect_summary["strongest_paired_delta_lfr"])),
            "mean_unpaired_over_paired_std_ratio": fmt_float(float(variance_summary["mean_ratio"])),
            "p8_mean_decode_us": "",
            "p8_p95_decode_us": "",
            "p8_throughput_shots_per_second": "",
            "source_artifacts": f"{args.effect_matrix}|{args.variance}",
        },
        {
            "summary_id": "p9_measured_decoder_runtime_replay",
            "evidence_source": "P8 measured decoder replay",
            "num_conditions": 1,
            "num_pairs": p8_summary.get("num_shots", 0),
            "baseline_lfr": "",
            "intervened_lfr": "",
            "paired_delta_lfr": "",
            "rescued_failure_count": "",
            "induced_failure_count": "",
            "strongest_condition": p8_summary.get("source_data_dir", ""),
            "strongest_paired_delta_lfr": "",
            "mean_unpaired_over_paired_std_ratio": "",
            "p8_mean_decode_us": fmt_float(float(p8_summary.get("mean_decode_us", 0.0))),
            "p8_p95_decode_us": fmt_float(float(p8_summary.get("p95_decode_us", 0.0))),
            "p8_throughput_shots_per_second": fmt_float(float(p8_summary.get("throughput", 0.0))),
            "source_artifacts": args.p8_runtime_summary,
        },
    ]


def build_claim_table(
    *,
    effect_summary: dict[str, object],
    variance_summary: dict[str, object],
    feature_summary: dict[str, dict[str, float]],
    prior_summary: dict[str, object],
    p8_summary: dict[str, object],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    rescued_features = feature_summary["rescued"]
    induced_features = feature_summary["induced"]
    return [
        {
            "claim_id": "C1",
            "paper_claim": "FailureOps computes paired counterfactual decoder-pathway sensitivity on real QEC detector records.",
            "evidence_source": "P7/P7.5 real detector records",
            "metric": "mean paired_delta_lfr over all conditions",
            "value": fmt_float(float(effect_summary["paired_delta_lfr"])),
            "interpretation": (
                f"{effect_summary['num_conditions']} real-data conditions and "
                f"{effect_summary['num_pairs']} valid pairs show intervention-sensitive logical failure behavior."
            ),
            "limitation": "This is decoder-pathway sensitivity, not live hardware runtime attribution.",
            "citation_artifact": args.effect_matrix,
        },
        {
            "claim_id": "C2",
            "paper_claim": "Paired counterfactual evaluation reduces estimator variance relative to unpaired resampling.",
            "evidence_source": "P7.5 paired-vs-unpaired bootstrap",
            "metric": "mean std_ratio_unpaired_over_paired",
            "value": fmt_float(float(variance_summary["mean_ratio"])),
            "interpretation": "Unpaired bootstrap estimates have higher delta-LFR variance on the same real records.",
            "limitation": "Bootstrap variance is not a formal end-to-end causal proof.",
            "citation_artifact": args.variance,
        },
        {
            "claim_id": "C3",
            "paper_claim": "Rescued and induced shots have measurable detector-record profiles.",
            "evidence_source": "P7.5 rescue/induction feature summary",
            "metric": "rescued_vs_induced_detector_count",
            "value": (
                f"rescued={fmt_float(rescued_features['mean_detector_event_count'])};"
                f"induced={fmt_float(induced_features['mean_detector_event_count'])}"
            ),
            "interpretation": "Transition classes can be described by detector-event count and burst features.",
            "limitation": "These are descriptive shot profiles, not standalone attribution without intervention.",
            "citation_artifact": args.features,
        },
        {
            "claim_id": "C4",
            "paper_claim": "Decoder-prior changes can be represented as paired interventions over the same real detector records.",
            "evidence_source": "P7.5 decoder-prior interventions",
            "metric": "largest prior-induced delta_lfr",
            "value": fmt_float(float(prior_summary["largest_regression_delta"])),
            "interpretation": f"Largest observed prior regression: {prior_summary['largest_regression_prior']}.",
            "limitation": "Current prior sweep is a focused single-condition analysis.",
            "citation_artifact": args.prior_effects,
        },
        {
            "claim_id": "C5",
            "paper_claim": "FailureOps can ingest measured decoder-runtime replay evidence without a live device.",
            "evidence_source": "P8 measured PyMatching replay",
            "metric": "mean and p95 measured per-shot decode time",
            "value": (
                f"mean_us={fmt_float(float(p8_summary.get('mean_decode_us', 0.0)))};"
                f"p95_us={fmt_float(float(p8_summary.get('p95_decode_us', 0.0)))}"
            ),
            "interpretation": "Runtime evidence is measured decoder service time on real detector records.",
            "limitation": "This is offline decoder replay, not live control-stack timing.",
            "citation_artifact": args.p8_runtime_summary,
        },
    ]


def build_realness_table(args: argparse.Namespace) -> list[dict[str, object]]:
    return [
        {
            "evidence_mode": "P7/P7.5 real detector-record attribution",
            "qec_record_source": "Google RL QEC real-device detector_events.b8 and obs_flips_actual.b8",
            "decoder_output_source": "Dataset decoder prediction files plus local PyMatching prior variants",
            "runtime_source": "none",
            "supports_claim": "paired intervention sensitivity on real QEC shot records",
            "does_not_support_claim": "live runtime queueing, deadline, or hardware idle attribution",
            "primary_artifacts": f"{args.effect_matrix}|{args.variance}|{args.features}|{args.prior_effects}",
        },
        {
            "evidence_mode": "P8 measured decoder-runtime replay",
            "qec_record_source": "same real detector records as P7",
            "decoder_output_source": "local PyMatching decode from dataset DEM",
            "runtime_source": "measured wall-clock decoder replay with time.perf_counter_ns",
            "supports_claim": "measured decoder service-time trace can enter FailureOps artifacts",
            "does_not_support_claim": "live quantum control-stack feedback delay or real hardware idle exposure",
            "primary_artifacts": args.p8_runtime_summary,
        },
        {
            "evidence_mode": "P3-P6 proxy/runtime modes",
            "qec_record_source": "generated Stim/PyMatching records",
            "decoder_output_source": "local PyMatching",
            "runtime_source": "explicit proxy model or imported trace path",
            "supports_claim": "controlled intervention engine, pairing contracts, and runtime-factor experiments",
            "does_not_support_claim": "real-device runtime behavior",
            "primary_artifacts": "data/results/p4_paired_effects.csv|data/results/p6_mode_summary.csv",
        },
    ]


def build_pairing_table(
    effect_summary: dict[str, object],
    p8_summary: dict[str, object],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    p8_units = int(p8_summary.get("num_shots", 0))
    p8_mismatch_rate = float(p8_summary.get("prediction_mismatch_rate", 0.0))
    p8_mismatches = round(p8_units * p8_mismatch_rate)
    return [
        {
            "evidence_source": "P7/P7.5 decoder-pathway pairs",
            "num_units": effect_summary["num_pairs"],
            "valid_units": int(effect_summary["num_pairs"]) - int(effect_summary["invalid_pairs"]),
            "invalid_units": effect_summary["invalid_pairs"],
            "violation_rate": fmt_float(float(effect_summary["violation_rate"])),
            "validation_note": "same shot identity, detector record, and observable flip preserved",
            "source_artifact": args.effect_matrix,
        },
        {
            "evidence_source": "P8 local replay vs stored decoder output",
            "num_units": p8_units,
            "valid_units": p8_units - p8_mismatches,
            "invalid_units": p8_mismatches,
            "violation_rate": fmt_float(p8_mismatch_rate),
            "validation_note": "agreement check only; local DEM replay is not assumed identical to stored decoder pathway",
            "source_artifact": args.p8_runtime_summary,
        },
    ]


def plot_evidence_bundle(
    *,
    effect_rows: list[dict[str, str]],
    variance_rows: list[dict[str, str]],
    feature_summary: dict[str, dict[str, float]],
    p8_summary: dict[str, object],
    output: str,
) -> None:
    ensure_parent_dir(output)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    plot_delta_by_cycles(axes[0][0], effect_rows)
    plot_variance_ratio(axes[0][1], variance_rows)
    plot_transition_features(axes[1][0], feature_summary)
    plot_p8_runtime(axes[1][1], p8_summary)
    fig.suptitle("P9 FailureOps evidence bundle", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_delta_by_cycles(ax, rows: list[dict[str, str]]) -> None:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[parse_int(row["cycles"])].append(float(row["paired_delta_lfr"]))
    cycles = sorted(grouped)
    means = [statistics.fmean(grouped[cycle]) for cycle in cycles]
    lows = [min(grouped[cycle]) for cycle in cycles]
    highs = [max(grouped[cycle]) for cycle in cycles]
    ax.fill_between(cycles, lows, highs, alpha=0.18, color="#4b78a8", label="condition range")
    ax.plot(cycles, means, marker="o", color="#2f5f8f", label="mean")
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_title("Real-record decoder sensitivity")
    ax.set_xlabel("QEC cycles")
    ax.set_ylabel("paired delta LFR")
    ax.legend(fontsize=8)


def plot_variance_ratio(ax, rows: list[dict[str, str]]) -> None:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[parse_int(row["cycles"])].append(float(row["std_ratio_unpaired_over_paired"]))
    cycles = sorted(grouped)
    ratios = [statistics.fmean(grouped[cycle]) for cycle in cycles]
    ax.plot(cycles, ratios, marker="o", color="#855c9c")
    ax.axhline(1.0, color="#333333", linewidth=0.8)
    ax.set_title("Paired vs unpaired variance")
    ax.set_xlabel("QEC cycles")
    ax.set_ylabel("std ratio")


def plot_transition_features(ax, feature_summary: dict[str, dict[str, float]]) -> None:
    transitions = ["rescued", "induced", "unchanged_failure"]
    counts = [feature_summary[name]["mean_detector_event_count"] for name in transitions]
    bursts = [feature_summary[name]["burst_fraction"] for name in transitions]
    x = list(range(len(transitions)))
    ax.bar([item - 0.18 for item in x], counts, width=0.36, color="#3f7fb5", label="detector count")
    ax2 = ax.twinx()
    ax2.bar([item + 0.18 for item in x], bursts, width=0.36, color="#b65f35", label="burst fraction")
    ax.set_title("Transition shot profiles")
    ax.set_xticks(x, transitions, rotation=15)
    ax.set_ylabel("mean detector count")
    ax2.set_ylabel("burst fraction")


def plot_p8_runtime(ax, p8_summary: dict[str, object]) -> None:
    labels = ["mean", "p95", "max"]
    values = [
        float(p8_summary.get("mean_decode_us", 0.0)),
        float(p8_summary.get("p95_decode_us", 0.0)),
        float(p8_summary.get("max_decode_us", 0.0)),
    ]
    ax.bar(labels, values, color="#5f7f4f")
    ax.set_title("Measured decoder replay")
    ax.set_ylabel("per-shot decode time (us)")


def weighted_mean(rows: list[dict[str, str]], field: str) -> float:
    total_weight = sum(parse_int(row["num_shots"]) for row in rows)
    if not total_weight:
        return 0.0
    return sum(float(row[field]) * parse_int(row["num_shots"]) for row in rows) / total_weight


if __name__ == "__main__":
    main()
