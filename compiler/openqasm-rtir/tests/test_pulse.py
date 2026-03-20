"""Tests for pulse-level IR, reference semantics, and checkers.

Covers all 6 examples (3 correct, 3 violations) and additional unit tests.
"""

import math
import pytest

from pulse_ir.ir import Config, FrameState, Waveform, Play, Acquire, ShiftPhase, Delay, IfBit
from pulse_ir.ref_semantics import step, run
from pulse_checks.port_exclusivity import check_port_exclusivity
from pulse_checks.feedback_causality import check_feedback_causality
from pulse_checks.frame_consistency import check_frame_consistency

TWO_PI = 2.0 * math.pi


# ---- helpers ----

def _simple_config_1frame() -> Config:
    return Config(
        frames=frozenset(["d0"]),
        ports=frozenset(["p0"]),
        port_of={"d0": "p0"},
        init_freq={"d0": 5.0e-3},
        init_phase={"d0": 0.0},
    )


# ===========================================================================
# 1. Reference semantics unit tests
# ===========================================================================

class TestStep:
    def test_play_advances_time_and_phase(self):
        cfg = _simple_config_1frame()
        s0 = FrameState.initial(cfg)
        s1 = step(s0, Play("d0", Waveform("g160", 160)), cfg)
        assert s1.time["d0"] == 160
        assert s1.phase["d0"] == pytest.approx(TWO_PI * 5.0e-3 * 160)
        assert s1.occupancy["p0"] == [(0, 160)]

    def test_acquire_sets_cbit_ready(self):
        cfg = Config(
            frames=frozenset(["m0"]),
            ports=frozenset(["p_meas"]),
            port_of={"m0": "p_meas"},
            init_freq={"m0": 7.0e-3},
            init_phase={"m0": 0.0},
        )
        s0 = FrameState.initial(cfg)
        s1 = step(s0, Acquire("m0", 1000, "c0"), cfg)
        assert s1.time["m0"] == 1000
        assert s1.cbit_ready["c0"] == 1000
        assert s1.occupancy["p_meas"] == [(0, 1000)]

    def test_shift_phase_zero_duration(self):
        cfg = _simple_config_1frame()
        s0 = FrameState.initial(cfg)
        s1 = step(s0, ShiftPhase("d0", math.pi / 2), cfg)
        assert s1.time["d0"] == 0  # no time advance
        assert s1.phase["d0"] == pytest.approx(math.pi / 2)

    def test_delay_no_port_occupancy(self):
        cfg = _simple_config_1frame()
        s0 = FrameState.initial(cfg)
        s1 = step(s0, Delay(500, "d0"), cfg)
        assert s1.time["d0"] == 500
        assert s1.phase["d0"] == pytest.approx(TWO_PI * 5.0e-3 * 500)
        assert s1.occupancy["p0"] == []  # delay is silence


# ===========================================================================
# 2. Correct examples — all checks PASS
# ===========================================================================

class TestCorrectExamples:
    def test_single_play(self):
        from pulse_examples.correct_single_play import config, program
        state = run(program, config)
        ok_pe, _ = check_port_exclusivity(state)
        ok_fc, _ = check_feedback_causality(program, config)
        ok_fr, _ = check_frame_consistency(state, program, config)
        assert ok_pe
        assert ok_fc
        assert ok_fr

    def test_measure_feedback(self):
        from pulse_examples.correct_measure_feedback import config, program
        state = run(program, config)
        ok_pe, _ = check_port_exclusivity(state)
        ok_fc, _ = check_feedback_causality(program, config)
        ok_fr, _ = check_frame_consistency(state, program, config)
        assert ok_pe
        assert ok_fc
        assert ok_fr

    def test_multi_frame(self):
        from pulse_examples.correct_multi_frame import config, program
        state = run(program, config)
        ok_pe, _ = check_port_exclusivity(state)
        ok_fc, _ = check_feedback_causality(program, config)
        ok_fr, _ = check_frame_consistency(state, program, config)
        assert ok_pe
        assert ok_fc
        assert ok_fr


# ===========================================================================
# 3. Violation examples — specific checks FAIL
# ===========================================================================

class TestViolationExamples:
    def test_port_conflict(self):
        from pulse_examples.violation_port_conflict import config, program
        state = run(program, config)
        ok, errors = check_port_exclusivity(state)
        assert not ok
        assert any("overlap" in e for e in errors)

    def test_causality_violation(self):
        from pulse_examples.violation_causality import config, program
        ok, errors = check_feedback_causality(program, config)
        assert not ok
        assert any("not ready" in e for e in errors)

    def test_phase_corruption(self):
        from pulse_examples.violation_phase import config, program, phase_corruption
        state = run(program, config)
        # Simulate compiler bug: corrupt the phase
        state.phase["d0"] += phase_corruption
        ok, errors = check_frame_consistency(state, program, config)
        assert not ok
        assert any("expected phase" in e for e in errors)

    def test_time_drift_self_consistent(self):
        """Codex-reported bug: a compiled state with extra time but self-consistent
        phase should FAIL correspondence check, not pass silently.

        Source program is empty → expected time=0, phase=0.
        Fake compiled state has time=100, phase=2π·freq·100 (self-consistent
        but diverged from source).
        """
        cfg = _simple_config_1frame()
        freq = cfg.init_freq["d0"]
        # Fabricate a "compiled" state that is internally consistent
        # but does NOT match the empty source program
        fake_state = FrameState(
            time={"d0": 100},
            phase={"d0": TWO_PI * freq * 100},
            cbit={},
            cbit_ready={},
            occupancy={"p0": []},
        )
        ok, errors = check_frame_consistency(fake_state, [], cfg)
        assert not ok
        assert any("expected time" in e for e in errors)


# ===========================================================================
# 4. Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_empty_program(self):
        cfg = _simple_config_1frame()
        state = run([], cfg)
        assert state.time["d0"] == 0
        ok_pe, _ = check_port_exclusivity(state)
        ok_fr, _ = check_frame_consistency(state, [], cfg)
        assert ok_pe
        assert ok_fr

    def test_multiple_shifts_accumulate(self):
        cfg = _simple_config_1frame()
        program = [
            ShiftPhase("d0", math.pi / 4),
            ShiftPhase("d0", math.pi / 4),
            ShiftPhase("d0", math.pi / 4),
        ]
        state = run(program, cfg)
        assert state.phase["d0"] == pytest.approx(3 * math.pi / 4)
        ok, _ = check_frame_consistency(state, program, cfg)
        assert ok

    def test_ifbit_always_executes_in_oracle(self):
        """Oracle executes IfBit body unconditionally for conservative timing."""
        cfg = Config(
            frames=frozenset(["d0", "m0"]),
            ports=frozenset(["p0", "p1"]),
            port_of={"d0": "p0", "m0": "p1"},
            init_freq={"d0": 5.0e-3, "m0": 7.0e-3},
            init_phase={"d0": 0.0, "m0": 0.0},
        )
        program = [
            Acquire("m0", 1000, "c0"),
            Delay(1000, "d0"),
            IfBit("c0", Play("d0", Waveform("x", 160))),
        ]
        state = run(program, cfg)
        # IfBit body should have been executed
        assert state.time["d0"] == 1160
        assert len(state.occupancy["p0"]) == 1
