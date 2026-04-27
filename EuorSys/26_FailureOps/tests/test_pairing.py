from copy import deepcopy

from failureops.intervention_registry import get_intervention_spec
from failureops.pairing import build_p4_intervention_row, event_record_hash, validate_pair
from failureops.runtime_service import apply_p3_intervention, recompute_p3_failure_behavior


def sample_record():
    row = {
        "run_id": "test_run",
        "workload_id": "memory_x",
        "circuit_id": "test_circuit",
        "backend": "test",
        "code_family": "repetition_code_memory",
        "distance": 5,
        "num_rounds": 8,
        "shot_id": 7,
        "seed": 1234,
        "stress_level": "high",
        "data_error_rate": 0.03,
        "measurement_error_rate": 0.02,
        "idle_error_rate": 0.002,
        "decoder_timeout_base_rate": 0.04,
        "detector_load_scale": 1.5,
        "idle_window_scale": 1.0,
        "deadline_scale": 1.0,
        "decoder_workers": 1,
        "decoder_service_rate": 2.4,
        "decoder_queue_depth": 4,
        "decoder_deadline": 0.42,
        "detector_count": 32,
        "detector_event_count": 8,
        "detector_events": "[0,1,2,3,4,5,6,7]",
        "observable_flip": True,
        "decoder_prediction": False,
        "decoder_arrival_time": "0.000000",
        "decoder_start_time": "0.000000",
        "decoder_finish_time": "0.000000",
        "decoder_latency": "0.000000",
        "decoder_deadline_miss": False,
        "decoder_queue_overflow": False,
        "decoder_timeout": False,
        "decoder_backlog": "0.000000",
        "runtime_stall_rounds": "0.000000",
        "idle_exposure": "0.000000",
        "runtime_idle_flip": False,
        "qec_decoder_failure": True,
    }
    return recompute_p3_failure_behavior(row)


def test_runtime_intervention_preserves_event_hash():
    baseline = sample_record()
    intervened = apply_p3_intervention(baseline, "remove_decoder_queueing")
    row = build_p4_intervention_row(baseline, intervened, "remove_decoder_queueing")
    assert row["baseline_event_record_hash"] == row["intervened_event_record_hash"]
    assert row["pairing_valid"]


def test_noise_intervention_allowed_to_change_event_hash():
    baseline = sample_record()
    intervened = apply_p3_intervention(baseline, "remove_data_noise")
    row = build_p4_intervention_row(baseline, intervened, "remove_data_noise")
    assert row["pairing_valid"]
    assert "event_record_changed_when_forbidden" not in row["pairing_violations"]
    assert event_record_hash(baseline)


def test_pairing_validator_catches_seed_mismatch():
    baseline = sample_record()
    intervened = deepcopy(baseline)
    intervened["seed"] = 9999
    spec = get_intervention_spec("remove_decoder_queueing")
    validation = validate_pair(baseline, intervened, spec)
    assert not validation["valid"]
    assert "seed_changed" in validation["violations"]

