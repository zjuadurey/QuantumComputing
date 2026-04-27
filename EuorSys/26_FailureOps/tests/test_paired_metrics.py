from failureops.bootstrap import paired_delta_lfr, paired_counts
from failureops.paired_metrics import summarize_paired_effects


def row(baseline, intervened):
    return {
        "workload_id": "memory_x",
        "stress_level": "high",
        "intervention": "remove_decoder_queueing",
        "intervention_class": "runtime",
        "pairing_valid": True,
        "baseline_logical_failure": baseline,
        "intervened_logical_failure": intervened,
    }


def test_paired_delta_equals_induced_minus_rescued():
    rows = [
        row(True, False),
        row(True, True),
        row(False, True),
        row(False, False),
        row(True, False),
    ]
    counts = paired_counts(rows)
    assert counts["rescued_failure_count"] == 2
    assert counts["induced_failure_count"] == 1
    assert paired_delta_lfr(rows) == (1 - 2) / 5


def test_summarize_paired_effects_reports_net_rescue_rate():
    rows = [
        row(True, False),
        row(True, True),
        row(False, True),
        row(False, False),
        row(True, False),
    ]
    summary = summarize_paired_effects(rows, num_bootstrap=10, bootstrap_seed=1)[0]
    assert summary["net_rescue_count"] == 1
    assert summary["net_rescue_rate"] == "0.200000"
    assert summary["paired_delta_lfr"] == "-0.200000"

