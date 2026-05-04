import importlib.util
from pathlib import Path


def load_digest_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "37_build_p10_eurosys_digest.py"
    spec = importlib.util.spec_from_file_location("p10_eurosys_digest", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_digest_rows_emits_expected_story_units():
    module = load_digest_module()
    rows = module.build_digest_rows(
        claim_rows=[
            {
                "claim_id": "C1",
                "claim": "main claim",
                "primary_metric": "holm count",
                "observed_value": "39/40",
                "limitation": "family boundary",
                "source_artifact": "claim.csv",
            },
            {
                "claim_id": "C2",
                "claim": "pairing claim",
                "primary_metric": "ratio",
                "observed_value": "1.75",
                "limitation": "bootstrap boundary",
                "source_artifact": "claim.csv",
            },
            {
                "claim_id": "C3",
                "claim": "baseline claim",
                "primary_metric": "spearman",
                "observed_value": "0.95/0.96",
                "limitation": "correlation caveat",
                "source_artifact": "baseline.csv",
            },
            {
                "claim_id": "C4",
                "claim": "runtime claim",
                "primary_metric": "delta",
                "observed_value": "0.307200",
                "limitation": "replay only",
                "source_artifact": "runtime.csv",
            },
            {
                "claim_id": "C6",
                "claim": "replication claim",
                "primary_metric": "23/26",
                "observed_value": "23/26",
                "limitation": "google external only",
                "source_artifact": "qec3v5.csv",
            },
            {
                "claim_id": "C6b",
                "claim": "v2 claim",
                "primary_metric": "v2 summary",
                "observed_value": "net_rescuing=461/496;holm_significant=273/496",
                "limitation": "same-family only",
                "source_artifact": "v2.csv",
            },
        ],
        robustness_rows=[
            {
                "check_id": "R1",
                "num_units": "40",
            },
            {
                "check_id": "R4",
                "value": "1.754761",
                "source_artifact": "variance.csv",
            },
            {
                "check_id": "R5",
                "num_units": "5724",
                "supporting_value": "significant=4879",
                "limitation": "same-family prior only",
                "source_artifact": "prior.csv",
            },
            {
                "check_id": "R7",
                "value": "1.783838",
                "source_artifact": "v2_variance.csv",
            },
        ],
        baseline_rows=[
            {
                "method": "Plain baseline LFR",
                "spearman_with_failureops": "0.962664",
            },
            {
                "method": "Static detector burden",
                "spearman_with_failureops": "0.949906",
            },
        ],
        runtime_rows=[
            {
                "deadline_us": "4.000000",
                "deadline_miss_rate": "0.936000",
                "paired_delta_lfr": "0.307200",
            },
            {
                "deadline_us": "7.000000",
                "deadline_miss_rate": "0.000000",
                "paired_delta_lfr": "0.000000",
            },
        ],
        prior_aggregate_rows=[
            {"intervened_prior": "dem_correlations", "mean_paired_delta_lfr": "-0.013326"},
            {"intervened_prior": "dem_rl_isolated_correlated_matching", "mean_paired_delta_lfr": "-0.021166"},
            {"intervened_prior": "dem_rl_shared_correlated_matching", "mean_paired_delta_lfr": "-0.018025"},
        ],
        google_v2_aggregate_rows=[
            {"group_by": "all", "group_value": "all", "mean_paired_delta_lfr": "-0.013661"},
            {
                "group_by": "control_mode",
                "group_value": "traditional_calibration",
                "mean_paired_delta_lfr": "-0.014388",
            },
            {
                "group_by": "control_mode",
                "group_value": "traditional_calibration_and_rl_fine_tuning",
                "mean_paired_delta_lfr": "-0.012934",
            },
            {"group_by": "basis", "group_value": "X", "mean_paired_delta_lfr": "-0.011489"},
            {"group_by": "basis", "group_value": "Z", "mean_paired_delta_lfr": "-0.015833"},
        ],
    )

    by_id = {row["story_id"]: row for row in rows}
    assert set(by_id) == {"S1", "S2", "S3", "S4", "S5", "S6", "S7"}
    assert by_id["S2"]["key_result"] == "orig=1.754761;v2=1.783838"
    assert "4us_delta=0.307200" in by_id["S4"]["key_result"]
    assert "ci_nonzero=4879/5724" in by_id["S7"]["key_result"]
