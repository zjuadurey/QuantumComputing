from failureops.event_layers import (
    apply_layered_noise_intervention,
    attach_event_layers,
    event_layer_hash,
    parse_event_layers,
)
from failureops.pairing import build_p4_intervention_row
from failureops.runtime_service import apply_p3_intervention, recompute_p3_failure_behavior


def sample_record():
    row = {
        "run_id": "layer_test",
        "workload_id": "memory_x",
        "circuit_id": "test",
        "backend": "test",
        "code_family": "repetition_code_memory",
        "distance": 5,
        "num_rounds": 8,
        "shot_id": 1,
        "seed": 101,
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
        "decoder_arrival_time": "0.100000",
        "decoder_start_time": "0.300000",
        "decoder_finish_time": "0.800000",
        "decoder_latency": "0.700000",
        "decoder_deadline_miss": True,
        "decoder_queue_overflow": False,
        "decoder_timeout": False,
        "decoder_backlog": "2.000000",
        "runtime_stall_rounds": "0.500000",
        "idle_exposure": "0.900000",
        "runtime_idle_flip": False,
        "detector_count": 32,
        "detector_event_count": 9,
        "detector_events": "[0,1,2,3,4,5,6,7,8]",
        "observable_flip": True,
        "decoder_prediction": False,
        "qec_decoder_failure": True,
    }
    return attach_event_layers(recompute_p3_failure_behavior(row))


def test_attach_event_layers_splits_detector_events():
    row = sample_record()
    layers = parse_event_layers(row["event_layers"])
    assert len(layers["data_events"]) == 3
    assert len(layers["measurement_events"]) == 3
    assert len(layers["idle_events"]) == 3


def test_layered_noise_intervention_changes_event_layer_hash():
    baseline = sample_record()
    intervened = apply_layered_noise_intervention(baseline, "remove_data_noise")
    assert event_layer_hash(intervened) != event_layer_hash(baseline)
    p4_row = build_p4_intervention_row(baseline, intervened, "remove_data_noise")
    assert p4_row["pairing_valid"]


def test_runtime_intervention_preserves_layered_event_record():
    baseline = sample_record()
    intervened = apply_p3_intervention(baseline, "remove_decoder_queueing")
    p4_row = build_p4_intervention_row(baseline, intervened, "remove_decoder_queueing")
    assert p4_row["baseline_event_record_hash"] == p4_row["intervened_event_record_hash"]
    assert p4_row["pairing_valid"]

