from failureops.pairing import event_record_hash
from failureops.runtime_service import recompute_p3_failure_behavior
from failureops.runtime_trace import (
    apply_runtime_trace_to_record,
    export_runtime_trace_rows,
    normalize_trace_row,
)


def sample_record():
    row = {
        "run_id": "trace_test",
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
        "detector_event_count": 4,
        "detector_events": "[0,1,2,3]",
        "observable_flip": True,
        "decoder_prediction": False,
        "qec_decoder_failure": True,
    }
    return recompute_p3_failure_behavior(row)


def test_trace_normalization_derives_queue_and_service_time():
    trace = normalize_trace_row(
        {
            "run_id": "trace_test",
            "workload_id": "memory_x",
            "stress_level": "high",
            "seed": 101,
            "shot_id": 1,
            "decoder_arrival_time": "0.1",
            "decoder_start_time": "0.3",
            "decoder_finish_time": "0.8",
        }
    )
    assert trace["decoder_queue_wait"] == "0.200000"
    assert trace["decoder_service_time"] == "0.500000"
    assert trace["decoder_latency"] == "0.700000"


def test_apply_runtime_trace_preserves_event_record_hash():
    baseline = sample_record()
    before = event_record_hash(baseline)
    traced = apply_runtime_trace_to_record(
        baseline,
        normalize_trace_row(
            {
                "run_id": "trace_test",
                "workload_id": "memory_x",
                "stress_level": "high",
                "seed": 101,
                "shot_id": 1,
                "decoder_arrival_time": "0.1",
                "decoder_start_time": "0.1",
                "decoder_finish_time": "0.2",
                "decoder_backlog": "0.0",
                "decoder_timeout": False,
                "decoder_deadline_miss": False,
                "idle_exposure": "0.1",
                "runtime_idle_flip": False,
            }
        ),
    )
    assert event_record_hash(traced) == before
    assert traced["decoder_latency"] == "0.100000"


def test_export_runtime_trace_rows_contains_observability_fields():
    exported = export_runtime_trace_rows([sample_record()])
    assert exported[0]["decoder_queue_wait"] == "0.200000"
    assert exported[0]["decoder_service_time"] == "0.500000"
    assert exported[0]["trace_source"] == "failureops_proxy_export"

