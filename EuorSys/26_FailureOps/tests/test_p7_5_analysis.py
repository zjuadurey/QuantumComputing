from failureops.p7_5_analysis import (
    bootstrap_delta_values,
    detector_features,
    transform_dem_text,
    transform_probability,
)


def test_detector_features_splits_detector_record():
    features = detector_features(
        {
            "detector_events": "[0,5,10,20,29]",
            "detector_count": 30,
            "cycles": 10,
        }
    )
    assert features["count"] == 5
    assert features["early"] == 2
    assert features["mid"] == 1
    assert features["late"] == 2
    assert features["is_burst"] == 1.0


def test_bootstrap_delta_values_keeps_paired_samples_together():
    rows = [
        {"baseline_logical_failure": True, "intervened_logical_failure": False},
        {"baseline_logical_failure": False, "intervened_logical_failure": True},
        {"baseline_logical_failure": True, "intervened_logical_failure": True},
        {"baseline_logical_failure": False, "intervened_logical_failure": False},
    ]
    paired, unpaired = bootstrap_delta_values(rows, num_bootstrap=20, seed=1)
    assert len(paired) == 20
    assert len(unpaired) == 20
    assert all(-1.0 <= value <= 1.0 for value in paired)


def test_dem_probability_transforms_are_valid():
    assert transform_probability(0.001, "probability_floor_1e-3", 0.002) == 0.001
    assert transform_probability(0.0001, "probability_floor_1e-3", 0.002) == 0.001
    assert transform_probability(0.02, "probability_ceiling_1e-2", 0.002) == 0.01
    transformed = transform_dem_text("error(0.001) D0\nerror(0.01) D1 L0\n", "uniform_median_probability")
    assert "error(" in transformed
    assert "D0" in transformed
