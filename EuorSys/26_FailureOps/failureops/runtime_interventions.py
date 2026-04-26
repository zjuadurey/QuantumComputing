"""P2 runtime/system interventions.

These interventions are strictly paired with the baseline QEC record. They reuse
the same detector events, observable, and decoder prediction, then change only
runtime policy knobs such as timeout policy, backlog, capacity, or slack.
"""

from __future__ import annotations

from failureops.qec_backend import apply_runtime_intervention
from failureops.qec_interventions import NOISE_INTERVENTIONS, compare_qec_records

P2_RUNTIME_INTERVENTIONS = [
    "remove_decoder_timeout",
    "relax_timeout_policy",
    "reduce_decoder_delay_50pct",
    "reduce_idle_exposure_50pct",
    "increase_decoder_capacity_2x",
    "eliminate_decoder_backlog",
    "increase_synchronization_slack",
]

P2_INTERVENTIONS = [
    "remove_data_noise",
    "weaken_data_noise_50pct",
    "remove_measurement_noise",
    "weaken_measurement_noise_50pct",
    "remove_idle_noise",
    "weaken_idle_noise_50pct",
    *P2_RUNTIME_INTERVENTIONS,
]


def generate_p2_runtime_rows(
    *,
    baseline_rows: list[dict[str, object]],
    intervention: str,
) -> list[dict[str, object]]:
    if intervention not in P2_RUNTIME_INTERVENTIONS:
        raise ValueError(f"not a P2 runtime intervention: {intervention}")
    return [
        compare_qec_records(baseline, apply_runtime_intervention(baseline, intervention), intervention)
        for baseline in baseline_rows
    ]


def classify_intervention(intervention: str) -> str:
    if intervention in P2_RUNTIME_INTERVENTIONS:
        return "runtime"
    if intervention in NOISE_INTERVENTIONS:
        return "noise"
    raise ValueError(f"unknown P2 intervention: {intervention}")
