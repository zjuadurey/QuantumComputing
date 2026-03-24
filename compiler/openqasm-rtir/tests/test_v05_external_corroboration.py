import math
import warnings

import pytest

from pulse_ir.ir import Config, Waveform, Play, ShiftPhase, Delay, Acquire, IfBit
from pulse_lowering.schedule import PulseEvent
from pulse_lowering.reconstruct import reconstruct_state
from pulse_lowering.lower_to_schedule import lower_to_schedule
from pulse_lowering.buggy_variants import (
    lower_buggy_drop_phase,
    lower_buggy_early_feedback,
    lower_buggy_ignore_shared_port,
    lower_buggy_reorder_ports,
)
from pulse_checks.feedback_causality import check_schedule_causality
from pulse_checks.frame_consistency import check_frame_consistency
from pulse_checks.port_exclusivity import diagnose_port_exclusivity
from pulse_checks.feedback_causality import diagnose_schedule_causality
from pulse_checks.frame_consistency import diagnose_frame_consistency
from pulse_external.qiskit_dynamics import (
    compare_schedule_lowerings,
    compare_single_frame_lowerings,
)
from pulse_frontends.qiskit_pulse import (
    translate_qiskit_scheduleblock,
    verify_qiskit_scheduleblock,
)


def _config_1frame_with_shift() -> Config:
    return Config(
        frames=frozenset(["d0"]),
        ports=frozenset(["p0"]),
        port_of={"d0": "p0"},
        init_freq={"d0": 5.0e-3},
        init_phase={"d0": 0.0},
    )


def _program_with_shift() -> list:
    return [
        Play("d0", Waveform("g160", 160)),
        ShiftPhase("d0", math.pi / 2),
        Play("d0", Waveform("g160", 160)),
    ]


def _config_2frame() -> Config:
    return Config(
        frames=frozenset(["d0", "m0"]),
        ports=frozenset(["p_drive", "p_meas"]),
        port_of={"d0": "p_drive", "m0": "p_meas"},
        init_freq={"d0": 5.0e-3, "m0": 7.0e-3},
        init_phase={"d0": 0.0, "m0": 0.0},
    )


def _program_with_multi_delay_feedback() -> list:
    return [
        Play("d0", Waveform("x_pulse", 160)),
        Acquire("m0", duration=1000, cbit="c0"),
        Delay(duration=400, frame="d0"),
        Delay(duration=600, frame="d0"),
        IfBit("c0", Play("d0", Waveform("x_pulse", 160))),
    ]


def _config_shared_port() -> Config:
    return Config(
        frames=frozenset(["d0", "d1"]),
        ports=frozenset(["p0"]),
        port_of={"d0": "p0", "d1": "p0"},
        init_freq={"d0": 0.0, "d1": 0.0},
        init_phase={"d0": 0.0, "d1": 0.0},
    )


def _program_shared_port_phase_diverse() -> list:
    return [
        Play("d0", Waveform("g160", 160)),
        ShiftPhase("d1", math.pi / 2),
        Play("d1", Waveform("g200", 200)),
    ]


def test_reconstruct_state_uses_latest_event_not_list_order():
    cfg = Config(
        frames=frozenset(["d0"]),
        ports=frozenset(["p0"]),
        port_of={"d0": "p0"},
        init_freq={"d0": 0.0},
        init_phase={"d0": 0.0},
    )
    events = [
        PulseEvent(1, "play", "d0", "p0", 10, 20, 1.0, 2.0),
        PulseEvent(0, "play", "d0", "p0", 0, 10, 0.0, 1.0),
    ]

    state = reconstruct_state(events, cfg)

    assert state.time["d0"] == 20
    assert state.phase["d0"] == pytest.approx(2.0)


def test_early_feedback_hoists_across_delay_block():
    cfg = _config_2frame()
    prog = _program_with_multi_delay_feedback()

    events = lower_buggy_early_feedback(prog, cfg)
    state = reconstruct_state(events, cfg)

    ok_fc, errors = check_schedule_causality(events)
    assert not ok_fc
    assert any("not ready" in e for e in errors)

    ok_fr, _ = check_frame_consistency(state, prog, cfg)
    assert ok_fr


def test_qiskit_dynamics_identical_lowerings_have_unit_fidelity():
    cfg = _config_1frame_with_shift()
    prog = _program_with_shift()

    result = compare_single_frame_lowerings(
        prog,
        cfg,
        frame="d0",
        lower_candidate=lower_to_schedule,
        drive_scale=0.02,
    )

    assert result.fidelity == pytest.approx(1.0, abs=1e-9)


def test_qiskit_dynamics_observes_drop_phase_deviation():
    cfg = _config_1frame_with_shift()
    prog = _program_with_shift()

    result = compare_single_frame_lowerings(
        prog,
        cfg,
        frame="d0",
        lower_candidate=lower_buggy_drop_phase,
        drive_scale=0.02,
    )

    assert result.fidelity < 0.999


def test_qiskit_dynamics_observes_ignore_shared_port_deviation():
    cfg = _config_shared_port()
    prog = _program_shared_port_phase_diverse()

    result = compare_schedule_lowerings(
        prog,
        cfg,
        scope="shared-port",
        lower_candidate=lower_buggy_ignore_shared_port,
        drive_scale=0.02,
    )

    assert result.fidelity < 0.999


def test_qiskit_dynamics_observes_reorder_ports_deviation():
    cfg = _config_shared_port()
    prog = _program_shared_port_phase_diverse()

    result = compare_schedule_lowerings(
        prog,
        cfg,
        scope="shared-port",
        lower_candidate=lower_buggy_reorder_ports,
        drive_scale=0.02,
    )

    assert result.fidelity < 0.999


def test_qiskit_dynamics_observes_early_feedback_timing_deviation():
    cfg = _config_2frame()
    prog = _program_with_multi_delay_feedback()

    result = compare_schedule_lowerings(
        prog,
        cfg,
        scope="d0-timing",
        frames=["d0"],
        lower_candidate=lower_buggy_early_feedback,
        drive_scale=0.02,
        static_drift=0.01,
    )

    assert result.fidelity < 0.999


def test_frame_consistency_diagnostics_quantify_phase_and_time_drift():
    cfg = _config_1frame_with_shift()
    prog = _program_with_shift()

    state = reconstruct_state(lower_buggy_drop_phase(prog, cfg), cfg)
    diagnostics = diagnose_frame_consistency(state, prog, cfg)

    assert diagnostics.num_phase_mismatches == 1
    assert diagnostics.max_abs_phase_diff == pytest.approx(math.pi / 2)
    assert diagnostics.num_time_mismatches == 0


def test_port_exclusivity_diagnostics_quantify_overlap():
    cfg = _config_shared_port()
    prog = _program_shared_port_phase_diverse()

    state = reconstruct_state(lower_buggy_ignore_shared_port(prog, cfg), cfg)
    diagnostics = diagnose_port_exclusivity(state)

    assert diagnostics.num_conflicts == 1
    assert diagnostics.max_overlap_dt == 160
    assert diagnostics.total_overlap_dt == 160


def test_feedback_causality_diagnostics_quantify_earliness():
    cfg = _config_2frame()
    prog = _program_with_multi_delay_feedback()

    events = lower_buggy_early_feedback(prog, cfg)
    diagnostics = diagnose_schedule_causality(events)

    assert diagnostics.num_violations == 1
    assert diagnostics.num_missing_cbits == 0
    assert diagnostics.max_earliness_dt == 840
    assert diagnostics.total_earliness_dt == 840


def _build_qiskit_scheduleblock():
    pytest.importorskip("qiskit")

    from qiskit import pulse
    from qiskit.pulse import AcquireChannel, DriveChannel, MemorySlot
    from qiskit.pulse.library import Constant

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pulse.build(name="adapter-demo") as sched:
            d0 = DriveChannel(0)
            a0 = AcquireChannel(0)
            pulse.play(Constant(160, 0.2), d0)
            pulse.delay(32, d0)
            pulse.shift_phase(math.pi / 2, d0)
            pulse.acquire(1000, a0, MemorySlot(0))
    return sched


def test_qiskit_scheduleblock_translates_to_core_ir_subset():
    sched = _build_qiskit_scheduleblock()

    translation = translate_qiskit_scheduleblock(sched)

    assert translation.program == [
        Play("d0", Waveform("Constant_160", 160)),
        Acquire("m0", 1000, "c0"),
        Delay(32, "d0"),
        ShiftPhase("d0", math.pi / 2),
    ]
    assert translation.config.frames == frozenset({"d0", "m0"})
    assert translation.config.port_of == {"d0": "d0", "m0": "m0"}


def test_qiskit_scheduleblock_adapter_enters_existing_verification_pipeline():
    sched = _build_qiskit_scheduleblock()

    _, report = verify_qiskit_scheduleblock(sched)

    assert report.well_formed
    assert report.overall_ok
    assert report.port_exclusive
    assert report.feedback_causal
    assert report.frame_consistent
