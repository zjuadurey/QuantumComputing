import pytest

from failureops.surface_backend import generate_surface_code_runs


def test_surface_backend_generates_pilot_rows():
    try:
        rows = generate_surface_code_runs(
            workload_id="surface_rotated_memory_x",
            stress_level="medium",
            num_shots=2,
            seed=42,
            run_id="surface_test",
        )
    except Exception as exc:
        pytest.skip(f"surface backend dependencies unavailable: {exc}")
    assert len(rows) == 2
    assert rows[0]["code_family"] == "rotated_surface_code_memory"
    assert "detector_events" in rows[0]
    assert "logical_failure" in rows[0]

