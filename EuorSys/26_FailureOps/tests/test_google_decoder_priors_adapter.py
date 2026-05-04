from failureops.google_decoder_priors_adapter import (
    available_decoder_backends,
    available_prior_variants,
    discover_google_decoder_prior_data_dirs,
    read_google_decoder_prior_metadata,
    recommended_intervened_priors,
)


def test_discover_google_decoder_prior_data_dirs_requires_prior_predictions(tmp_path):
    data_dir = (
        tmp_path
        / "sycamore_surface_code_d3_d5"
        / "sample_01"
        / "d3_at_q6_3"
        / "X"
        / "r05"
    )
    data_dir.mkdir(parents=True)
    for name in ("circuit_ideal.stim", "detection_events.b8", "obs_flips_actual.b8"):
        (data_dir / name).write_text("")
    (data_dir / "metadata.json").write_text('{"basis":"X","rounds":5}')
    for prior in ("dem_simple", "dem_correlations"):
        prior_dir = data_dir / "obs_flips_predicted" / prior
        prior_dir.mkdir(parents=True)
        (prior_dir / "correlated_matching.b8").write_text("")
        (prior_dir / "harmony.b8").write_text("")

    assert discover_google_decoder_prior_data_dirs(
        tmp_path,
        decoder_backend="correlated_matching",
        required_prior_variants=("dem_simple", "dem_correlations"),
    ) == [data_dir]


def test_read_google_decoder_prior_metadata_infers_context(tmp_path):
    data_dir = (
        tmp_path
        / "sycamore_surface_code_d3_d5"
        / "sample_19"
        / "d5_at_q6_5"
        / "X"
        / "r25"
    )
    data_dir.mkdir(parents=True)
    (data_dir / "metadata.json").write_text('{"basis":"X","rounds":25,"distance":5}')

    metadata = read_google_decoder_prior_metadata(data_dir)

    assert metadata["experiment_name"] == "sycamore_surface_code_d3_d5"
    assert metadata["condition_id"] == "sample_19_d5_at_q6_5"
    assert metadata["sample_id"] == "sample_19"
    assert metadata["placement_id"] == "d5_at_q6_5"
    assert metadata["basis"] == "X"
    assert metadata["cycle_dir"] == "r25"


def test_available_prior_variants_and_decoder_backends_are_discovered(tmp_path):
    data_dir = (
        tmp_path
        / "sycamore_surface_code_d3_d5"
        / "sample_01"
        / "d3_at_q6_3"
        / "X"
        / "r05"
    )
    (data_dir / "obs_flips_predicted" / "dem_simple").mkdir(parents=True)
    (data_dir / "obs_flips_predicted" / "dem_correlations").mkdir(parents=True)
    for name in ("correlated_matching.b8", "belief_matching.b8"):
        (data_dir / "obs_flips_predicted" / "dem_simple" / name).write_text("")
    (data_dir / "obs_flips_predicted" / "dem_correlations" / "correlated_matching.b8").write_text("")

    assert available_prior_variants(data_dir) == ("dem_correlations", "dem_simple")
    assert available_prior_variants(data_dir, decoder_backend="belief_matching") == ("dem_simple",)
    assert available_decoder_backends(data_dir) == ("belief_matching", "correlated_matching")


def test_recommended_intervened_priors_match_decoder_backend():
    assert recommended_intervened_priors("correlated_matching") == (
        "dem_correlations",
        "dem_rl_isolated_correlated_matching",
        "dem_rl_shared_correlated_matching",
    )
