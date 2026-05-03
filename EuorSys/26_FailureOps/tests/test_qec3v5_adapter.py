from failureops.qec3v5_adapter import parse_qec3v5_dir_name, summarize_qec3v5_groups


def test_parse_qec3v5_dir_name_extracts_surface_metadata():
    metadata = parse_qec3v5_dir_name("surface_code_bZ_d5_r25_center_5_5")

    assert metadata["code"] == "surface_code"
    assert metadata["basis"] == "Z"
    assert metadata["distance"] == 5
    assert metadata["rounds"] == 25
    assert metadata["center_row"] == 5
    assert metadata["center_col"] == 5


def test_qec3v5_group_summary_reports_strongest_condition():
    rows = [
        {
            "basis": "X",
            "cycles": 5,
            "distance": 5,
            "num_pairs": 100,
            "baseline_lfr": "0.100000",
            "intervened_lfr": "0.070000",
            "paired_delta_lfr": "-0.030000",
            "workload_id": "x5",
        },
        {
            "basis": "Z",
            "cycles": 5,
            "distance": 5,
            "num_pairs": 100,
            "baseline_lfr": "0.120000",
            "intervened_lfr": "0.110000",
            "paired_delta_lfr": "-0.010000",
            "workload_id": "z5",
        },
    ]

    summary = summarize_qec3v5_groups(rows)
    all_row = [row for row in summary if row["group_by"] == "all"][0]

    assert all_row["strongest_condition"] == "x5"
    assert all_row["mean_paired_delta_lfr"] == "-0.020000"
