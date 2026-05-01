from pathlib import Path

from failureops.google_rl_qec_adapter import (
    build_baseline_row,
    build_decoder_intervention_row,
    build_intervened_decoder_row,
    discover_google_rl_qec_data_dirs,
    summarize_p7_sweep_groups,
)


def test_decoder_pathway_switch_preserves_shot_identity_and_detector_record():
    baseline = build_baseline_row(
        data_dir=Path("dataset/surface_code_traditional_calibration/Z/r010"),
        metadata={
            "experiment_name": "surface_code_traditional_calibration",
            "basis": "Z",
            "cycle_dir": "r010",
            "rounds": 10,
        },
        run_id="p7_test",
        shot_id=3,
        detector_count=30,
        detector_events=[1, 7, 11],
        observable_flip=True,
        decoder_pathway="correlated_matching_decoder_with_si1000_prior",
        decoder_prediction=False,
    )
    intervened = build_intervened_decoder_row(
        baseline,
        decoder_pathway="tesseract_decoder_with_si1000_prior",
        decoder_prediction=True,
    )
    row = build_decoder_intervention_row(baseline, intervened)

    assert row["pairing_valid"]
    assert row["intervention"] == "switch_decoder_pathway"
    assert row["baseline_logical_failure"]
    assert not row["intervened_logical_failure"]
    assert row["rescued_failure"]
    assert row["baseline_decoder_pathway"] == "correlated_matching_decoder_with_si1000_prior"
    assert row["intervened_decoder_pathway"] == "tesseract_decoder_with_si1000_prior"


def test_discover_google_rl_qec_data_dirs_requires_decoder_predictions(tmp_path):
    data_dir = tmp_path / "surface_code_traditional_calibration" / "Z" / "r010"
    data_dir.mkdir(parents=True)
    for name in ("circuit_ideal.stim", "detection_events.b8", "obs_flips_actual.b8", "metadata.json"):
        (data_dir / name).write_text("{}")
    for pathway in (
        "correlated_matching_decoder_with_si1000_prior",
        "tesseract_decoder_with_si1000_prior",
    ):
        decoder_dir = data_dir / "decoding_results" / pathway
        decoder_dir.mkdir(parents=True)
        (decoder_dir / "obs_flips_predicted.b8").write_text("")

    assert discover_google_rl_qec_data_dirs(tmp_path) == [data_dir]


def test_summarize_p7_sweep_groups_reports_strongest_condition():
    rows = [
        {
            "code_family": "surface_code_memory",
            "control_mode": "traditional_calibration",
            "basis": "Z",
            "cycles": 10,
            "num_pairs": 100,
            "baseline_lfr": "0.100000",
            "intervened_lfr": "0.080000",
            "paired_delta_lfr": "-0.020000",
            "net_rescue_rate": "0.020000",
            "workload_id": "a",
        },
        {
            "code_family": "surface_code_memory",
            "control_mode": "reinforcement_learning",
            "basis": "Z",
            "cycles": 20,
            "num_pairs": 100,
            "baseline_lfr": "0.120000",
            "intervened_lfr": "0.070000",
            "paired_delta_lfr": "-0.050000",
            "net_rescue_rate": "0.050000",
            "workload_id": "b",
        },
    ]
    all_row = [row for row in summarize_p7_sweep_groups(rows) if row["group_by"] == "all"][0]
    assert all_row["num_conditions"] == 2
    assert all_row["strongest_condition"] == "b"
    assert all_row["mean_paired_delta_lfr"] == "-0.035000"
