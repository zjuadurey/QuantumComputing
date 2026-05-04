#!/usr/bin/env python
"""Run paired decoder-prior evidence over the Google decoder-priors corpus."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from failureops.data_model import P7_5_PRIOR_EFFECT_FIELDS
from failureops.google_decoder_priors_adapter import (
    DEFAULT_BASELINE_PRIOR,
    discover_google_decoder_prior_data_dirs,
    load_google_decoder_prior_records,
    recommended_intervened_priors,
)
from failureops.io_utils import ensure_parent_dir, fmt_float, parse_int, write_csv_rows
from failureops.manifest import write_manifest
from failureops.paired_metrics import summarize_paired_effects

os.environ.setdefault("MPLCONFIGDIR", "/tmp/failureops-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRIOR_AGGREGATE_FIELDS = [
    "decoder_backend",
    "baseline_prior",
    "intervened_prior",
    "num_conditions",
    "total_pairs",
    "mean_paired_delta_lfr",
    "min_paired_delta_lfr",
    "max_paired_delta_lfr",
    "negative_delta_fraction",
    "mean_net_rescue_rate",
    "strongest_condition",
    "strongest_paired_delta_lfr",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="data/raw/google_decoder_priors")
    parser.add_argument("--decoder-backend", default="correlated_matching")
    parser.add_argument("--baseline-prior", default=DEFAULT_BASELINE_PRIOR)
    parser.add_argument(
        "--intervened-priors",
        default="auto",
        help="Comma-separated prior variants, or 'auto' for backend-matched defaults.",
    )
    parser.add_argument("--max-shots", type=int, default=10000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-bootstrap", type=int, default=100)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument(
        "--output",
        default="data/results/p10_google_decoder_priors_prior_effects.csv",
    )
    parser.add_argument(
        "--aggregate-output",
        default="data/results/p10_google_decoder_priors_prior_effects_aggregate.csv",
    )
    parser.add_argument(
        "--figure-output",
        default="figures/p10_google_decoder_priors_prior_effects.png",
    )
    parser.add_argument(
        "--manifest-output",
        default="data/results/p10_google_decoder_priors_manifest.json",
    )
    args = parser.parse_args()

    intervened_priors = parse_prior_list(args.intervened_priors, decoder_backend=args.decoder_backend)
    required_priors = (args.baseline_prior, *intervened_priors)
    data_dirs = discover_google_decoder_prior_data_dirs(
        args.dataset_root,
        decoder_backend=args.decoder_backend,
        required_prior_variants=required_priors,
    )
    if args.limit:
        data_dirs = data_dirs[: args.limit]
    if not data_dirs:
        raise ValueError(
            f"no Google decoder-prior directories under {args.dataset_root} "
            f"for backend={args.decoder_backend!r} priors={required_priors!r}"
        )

    rows = []
    for index, data_dir in enumerate(data_dirs):
        for prior_offset, intervened_prior in enumerate(intervened_priors):
            run_id = f"p10_google_decoder_priors_{index:04d}_{prior_offset:02d}"
            baseline_rows, intervention_rows = load_google_decoder_prior_records(
                data_dir,
                baseline_prior=args.baseline_prior,
                intervened_prior=intervened_prior,
                decoder_backend=args.decoder_backend,
                max_shots=args.max_shots,
                run_id=run_id,
            )
            effect = summarize_paired_effects(
                intervention_rows,
                num_bootstrap=args.num_bootstrap,
                bootstrap_seed=args.bootstrap_seed + index * 100 + prior_offset,
            )[0]
            rows.append(
                {
                    "source_data_dir": str(data_dir),
                    "baseline_prior": args.baseline_prior,
                    "intervened_prior": intervened_prior,
                    **effect,
                }
            )
            print(
                f"[{index + 1}/{len(data_dirs)}] {rows[-1]['workload_id']} "
                f"{args.baseline_prior}->{intervened_prior} delta={rows[-1]['paired_delta_lfr']}"
            )

    aggregate_rows = summarize_prior_groups(
        rows,
        decoder_backend=args.decoder_backend,
        baseline_prior=args.baseline_prior,
    )
    write_csv_rows(args.output, rows, P7_5_PRIOR_EFFECT_FIELDS)
    write_csv_rows(args.aggregate_output, aggregate_rows, PRIOR_AGGREGATE_FIELDS)
    plot_prior_sweep(aggregate_rows, args.figure_output)
    write_manifest(
        args.manifest_output,
        config={
            "experiment_id": "p10_google_decoder_priors",
            "dataset_root": args.dataset_root,
            "decoder_backend": args.decoder_backend,
            "baseline_prior": args.baseline_prior,
            "intervened_priors": list(intervened_priors),
            "max_shots": args.max_shots,
            "limit": args.limit,
            "num_bootstrap": args.num_bootstrap,
            "bootstrap_seed": args.bootstrap_seed,
        },
        command=sys.argv,
        outputs={
            "prior_effects": args.output,
            "aggregate": args.aggregate_output,
            "figure": args.figure_output,
        },
        row_counts={
            "prior_effects": len(rows),
            "aggregate": len(aggregate_rows),
        },
    )
    print(f"wrote {len(rows)} Google decoder-prior effect rows to {args.output}")
    print(f"wrote {len(aggregate_rows)} Google decoder-prior aggregate rows to {args.aggregate_output}")
    print(f"wrote Google decoder-prior evidence figure to {args.figure_output}")
    print(f"wrote Google decoder-prior manifest to {args.manifest_output}")


def parse_prior_list(value: str, *, decoder_backend: str) -> tuple[str, ...]:
    if value.strip() == "auto":
        return recommended_intervened_priors(decoder_backend)
    priors = tuple(item.strip() for item in value.split(",") if item.strip())
    if not priors:
        raise ValueError("--intervened-priors must contain at least one prior or 'auto'")
    return priors


def summarize_prior_groups(
    rows: list[dict[str, object]],
    *,
    decoder_backend: str,
    baseline_prior: str,
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["intervened_prior"])].append(row)
    out = []
    for intervened_prior in sorted(grouped):
        group = grouped[intervened_prior]
        deltas = [float(row["paired_delta_lfr"]) for row in group]
        net_rescue = [float(row["net_rescue_rate"]) for row in group]
        strongest = min(group, key=lambda row: float(row["paired_delta_lfr"]))
        out.append(
            {
                "decoder_backend": decoder_backend,
                "baseline_prior": baseline_prior,
                "intervened_prior": intervened_prior,
                "num_conditions": len(group),
                "total_pairs": sum(parse_int(row["num_pairs"]) for row in group),
                "mean_paired_delta_lfr": fmt_float(statistics.fmean(deltas) if deltas else 0.0),
                "min_paired_delta_lfr": fmt_float(min(deltas) if deltas else 0.0),
                "max_paired_delta_lfr": fmt_float(max(deltas) if deltas else 0.0),
                "negative_delta_fraction": fmt_float(sum(delta < 0.0 for delta in deltas) / len(deltas) if deltas else 0.0),
                "mean_net_rescue_rate": fmt_float(statistics.fmean(net_rescue) if net_rescue else 0.0),
                "strongest_condition": str(strongest["workload_id"]),
                "strongest_paired_delta_lfr": strongest["paired_delta_lfr"],
            }
        )
    return out


def plot_prior_sweep(rows: list[dict[str, object]], output: str) -> None:
    labels = [str(row["intervened_prior"]).replace("dem_", "") for row in rows]
    mean_delta = [float(row["mean_paired_delta_lfr"]) for row in rows]
    negative_fraction = [float(row["negative_delta_fraction"]) for row in rows]
    x = list(range(len(rows)))
    fig, ax1 = plt.subplots(figsize=(9.5, 5.5))
    ax1.bar(x, mean_delta, color="#4b78a8")
    ax1.axhline(0.0, color="#333333", linewidth=0.8)
    ax1.set_xticks(x, labels)
    ax1.set_ylabel("Mean paired delta LFR")
    ax1.set_title("Google decoder-prior interventions on paired real detector records")
    ax1.tick_params(axis="x", labelrotation=18)
    ax2 = ax1.twinx()
    ax2.plot(x, negative_fraction, color="#b65f35", marker="o", linewidth=1.6)
    ax2.set_ylim(0.0, 1.05)
    ax2.set_ylabel("Negative-delta fraction")
    fig.tight_layout()
    ensure_parent_dir(output)
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
