from failureops.p6_unified import parse_mode_spec, summarize_intervention_stability, summarize_mode_rank_stability


def test_parse_mode_spec():
    mode = parse_mode_spec(
        "repetition_proxy_layered,repetition,proxy,layered,memory_x|idle_heavy_memory,low|high"
    )
    assert mode.mode_id == "repetition_proxy_layered"
    assert mode.backend == "repetition"
    assert mode.runtime_source == "proxy"
    assert mode.event_model == "layered"
    assert mode.workloads == ("memory_x", "idle_heavy_memory")
    assert mode.stress_levels == ("low", "high")


def test_intervention_stability_groups_modes():
    rows = [
        {
            "mode_id": "a",
            "intervention": "remove_decoder_queueing",
            "paired_delta_lfr": "-0.10",
            "net_rescue_rate": "0.10",
        },
        {
            "mode_id": "b",
            "intervention": "remove_decoder_queueing",
            "paired_delta_lfr": "-0.20",
            "net_rescue_rate": "0.20",
        },
    ]
    summary = summarize_intervention_stability(rows)[0]
    assert summary["num_modes"] == 2
    assert summary["mean_paired_delta_lfr"] == "-0.150000"
    assert summary["top_mode"] == "b"


def test_mode_rank_stability_reports_overlap():
    summary = summarize_mode_rank_stability(
        {
            "a": ["x", "y", "z"],
            "b": ["x", "z", "w"],
        }
    )
    assert summary["num_modes"] == 2
    assert summary["mean_top3_overlap"] == "0.666667"

