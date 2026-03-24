import pytest

from qiskit_examples import (
    build_drive_and_acquire,
    build_single_drive_with_virtual_z,
    build_two_drive_channels,
)
from pulse_frontends.qiskit_pulse import (
    translate_qiskit_scheduleblock,
    verify_qiskit_scheduleblock,
)


pytest.importorskip("qiskit")


def test_translate_single_drive_with_virtual_z_example():
    sched = build_single_drive_with_virtual_z()

    translation = translate_qiskit_scheduleblock(sched)

    assert [type(stmt).__name__ for stmt in translation.program] == [
        "Play",
        "Delay",
        "ShiftPhase",
        "Play",
    ]
    assert translation.config.frames == frozenset({"d0"})


def test_translate_drive_and_acquire_example():
    sched = build_drive_and_acquire()

    translation = translate_qiskit_scheduleblock(sched)

    assert {type(stmt).__name__ for stmt in translation.program} == {
        "Play",
        "Acquire",
        "Delay",
        "ShiftPhase",
    }
    assert translation.config.frames == frozenset({"d0", "m0"})


def test_verify_qiskit_examples_enter_existing_pipeline():
    examples = [
        build_single_drive_with_virtual_z(),
        build_drive_and_acquire(),
        build_two_drive_channels(),
    ]

    for sched in examples:
        _, report = verify_qiskit_scheduleblock(sched)
        assert report.well_formed
        assert report.overall_ok
        assert report.port_exclusive
        assert report.feedback_causal
        assert report.frame_consistent
