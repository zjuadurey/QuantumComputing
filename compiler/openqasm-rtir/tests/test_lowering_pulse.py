"""End-to-end tests: source → lowering → reconstruct → checker.

Tests the full correspondence verification pipeline:
1. Correct lowering: all checkers PASS on reconstructed state
2. Buggy lowering variants: targeted checker FAILS, ALL non-target checkers PASS

This is the core evidence for the paper's correspondence property claim.
"""

import math
import pytest

from pulse_ir.ir import Config, Waveform, Play, Acquire, ShiftPhase, Delay, IfBit
from pulse_ir.ref_semantics import run
from pulse_lowering.lower_to_schedule import lower_to_schedule
from pulse_lowering.reconstruct import reconstruct_state
from pulse_lowering.buggy_variants import (
    lower_buggy_drop_phase,
    lower_buggy_extra_delay,
    lower_buggy_reorder_ports,
    lower_buggy_early_feedback,
)
from pulse_checks.port_exclusivity import check_port_exclusivity
from pulse_checks.feedback_causality import check_feedback_causality
from pulse_checks.frame_consistency import check_frame_consistency


# ---- shared fixtures ----

def _config_2frame() -> Config:
    """Two frames on separate ports (drive + measure)."""
    return Config(
        frames=frozenset(["d0", "m0"]),
        ports=frozenset(["p_drive", "p_meas"]),
        port_of={"d0": "p_drive", "m0": "p_meas"},
        init_freq={"d0": 5.0e-3, "m0": 7.0e-3},
        init_phase={"d0": 0.0, "m0": 0.0},
    )


def _program_with_feedback() -> list:
    """Play → Acquire → Delay (wait) → IfBit(conditional play)."""
    return [
        Play("d0", Waveform("x_pulse", 160)),
        Acquire("m0", duration=1000, cbit="c0"),
        Delay(duration=1000, frame="d0"),
        IfBit("c0", Play("d0", Waveform("x_pulse", 160))),
    ]


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


def _config_shared_port() -> Config:
    """Two frames sharing a single port."""
    return Config(
        frames=frozenset(["d0", "d1"]),
        ports=frozenset(["p0"]),
        port_of={"d0": "p0", "d1": "p0"},
        init_freq={"d0": 5.0e-3, "d1": 5.1e-3},
        init_phase={"d0": 0.0, "d1": 0.0},
    )


def _program_shared_port() -> list:
    """Two plays on different frames but same port — sequential, no conflict."""
    return [
        Play("d0", Waveform("g160", 160)),
        Play("d1", Waveform("g200", 200)),
    ]


# ===========================================================================
# 1. Correct lowering — all checks PASS (including compiled-mode causality)
# ===========================================================================

class TestCorrectLowering:
    def test_feedback_program_all_pass(self):
        """All 3 checks pass on correctly lowered feedback program."""
        cfg = _config_2frame()
        prog = _program_with_feedback()
        events = lower_to_schedule(prog, cfg)
        state = reconstruct_state(events, cfg)
        ok_pe, _ = check_port_exclusivity(state)
        ok_fc, _ = check_feedback_causality(prog, cfg, compiled_events=events)
        ok_fr, _ = check_frame_consistency(state, prog, cfg)
        assert ok_pe
        assert ok_fc
        assert ok_fr

    def test_shift_program_all_pass(self):
        cfg = _config_1frame_with_shift()
        prog = _program_with_shift()
        events = lower_to_schedule(prog, cfg)
        state = reconstruct_state(events, cfg)
        ok_pe, _ = check_port_exclusivity(state)
        ok_fr, _ = check_frame_consistency(state, prog, cfg)
        assert ok_pe
        assert ok_fr

    def test_lowering_matches_oracle(self):
        """Correct lowering should produce the same final state as oracle."""
        cfg = _config_1frame_with_shift()
        prog = _program_with_shift()
        oracle_state = run(prog, cfg)
        compiled_state = reconstruct_state(lower_to_schedule(prog, cfg), cfg)
        assert compiled_state.time == oracle_state.time
        for f in cfg.frames:
            assert compiled_state.phase[f] == pytest.approx(oracle_state.phase[f])
        assert compiled_state.occupancy == oracle_state.occupancy


# ===========================================================================
# 2. Buggy lowering — targeted checks FAIL, ALL non-target checks PASS
# ===========================================================================

class TestBuggyLowering:
    def test_drop_phase_caught_by_frame_consist(self):
        """Compiler drops ShiftPhase → FrameConsist FAILS.
        Non-target: PortExcl PASSES, FeedbackCausal PASSES (no IfBit in program).
        """
        cfg = _config_1frame_with_shift()
        prog = _program_with_shift()
        events = lower_buggy_drop_phase(prog, cfg)
        state = reconstruct_state(events, cfg)
        # Target: FrameConsist FAILS
        ok_fr, errors = check_frame_consistency(state, prog, cfg)
        assert not ok_fr
        assert any("expected phase" in e for e in errors)
        # Non-target: PortExcl PASSES
        ok_pe, _ = check_port_exclusivity(state)
        assert ok_pe
        # Non-target: FeedbackCausal PASSES (no IfBit → no conditional events)
        ok_fc, _ = check_feedback_causality(prog, cfg, compiled_events=events)
        assert ok_fc

    def test_extra_delay_caught_by_frame_consist(self):
        """Compiler inserts extra delay → FrameConsist FAILS.
        Non-target: PortExcl PASSES, FeedbackCausal PASSES (no IfBit in program).
        """
        cfg = _config_1frame_with_shift()
        prog = _program_with_shift()
        events = lower_buggy_extra_delay(prog, cfg, extra_dt=50)
        state = reconstruct_state(events, cfg)
        # Target: FrameConsist FAILS
        ok_fr, errors = check_frame_consistency(state, prog, cfg)
        assert not ok_fr
        assert any("expected time" in e for e in errors)
        # Non-target: PortExcl PASSES
        ok_pe, _ = check_port_exclusivity(state)
        assert ok_pe
        # Non-target: FeedbackCausal PASSES (no IfBit → no conditional events)
        ok_fc, _ = check_feedback_causality(prog, cfg, compiled_events=events)
        assert ok_fc

    def test_reorder_caught_by_port_excl(self):
        """Compiler flattens to t=0 → PortExcl FAILS.
        Non-target: FrameConsist PASSES (per-frame time/phase correct with
        one operation each), FeedbackCausal PASSES (no IfBit in program).
        """
        cfg = _config_shared_port()
        prog = _program_shared_port()
        events = lower_buggy_reorder_ports(prog, cfg)
        state = reconstruct_state(events, cfg)
        # Target: PortExcl FAILS (overlap on shared port)
        ok_pe, errors_pe = check_port_exclusivity(state)
        assert not ok_pe
        assert any("overlap" in e for e in errors_pe)
        # Non-target: FrameConsist PASSES
        ok_fr, _ = check_frame_consistency(state, prog, cfg)
        assert ok_fr
        # Non-target: FeedbackCausal PASSES (no IfBit → no conditional events)
        ok_fc, _ = check_feedback_causality(prog, cfg, compiled_events=events)
        assert ok_fc

    def test_early_feedback_caught_by_causality(self):
        """Compiler reorders IfBit before Delay → FeedbackCausal FAILS.

        The reorder swaps (Delay, IfBit) so the conditional play executes
        at t=160 instead of t=1160, but cbit_ready=1000.
        Total time/phase per frame is unchanged (same operations, reordered).

        Non-target: PortExcl PASSES (separate ports),
                    FrameConsist PASSES (same total time/phase per frame).
        """
        cfg = _config_2frame()
        prog = _program_with_feedback()
        # Full pipeline: buggy lowering → events
        events = lower_buggy_early_feedback(prog, cfg)
        state = reconstruct_state(events, cfg)
        # Target: FeedbackCausal FAILS (compiled mode on events)
        ok_fc, errors = check_feedback_causality(
            prog, cfg, compiled_events=events,
        )
        assert not ok_fc
        assert any("not ready" in e for e in errors)
        # Non-target: PortExcl PASSES (d0 and m0 on separate ports)
        ok_pe, _ = check_port_exclusivity(state)
        assert ok_pe
        # Non-target: FrameConsist PASSES (same total time/phase per frame)
        ok_fr, _ = check_frame_consistency(state, prog, cfg)
        assert ok_fr


# ===========================================================================
# 3. Schedule structure tests
# ===========================================================================

class TestScheduleStructure:
    def test_event_count(self):
        cfg = _config_1frame_with_shift()
        prog = _program_with_shift()
        events = lower_to_schedule(prog, cfg)
        # Play + ShiftPhase + Play = 3 events
        assert len(events) == 3

    def test_events_have_explicit_times(self):
        cfg = _config_1frame_with_shift()
        prog = _program_with_shift()
        events = lower_to_schedule(prog, cfg)
        # First play: [0, 160)
        assert events[0].start == 0
        assert events[0].end == 160
        # ShiftPhase: zero duration at t=160
        assert events[1].start == 160
        assert events[1].end == 160
        # Second play: [160, 320)
        assert events[2].start == 160
        assert events[2].end == 320

    def test_phase_snapshots_in_events(self):
        cfg = _config_1frame_with_shift()
        prog = _program_with_shift()
        events = lower_to_schedule(prog, cfg)
        # ShiftPhase event should show phase jump
        shift_ev = events[1]
        assert shift_ev.kind == "shift_phase"
        assert shift_ev.phase_after == pytest.approx(
            shift_ev.phase_before + math.pi / 2
        )

    def test_conditional_events_tagged(self):
        """Events from IfBit body should have conditional_on set."""
        cfg = _config_2frame()
        prog = _program_with_feedback()
        events = lower_to_schedule(prog, cfg)
        conditional = [ev for ev in events if ev.conditional_on]
        assert len(conditional) == 1
        assert conditional[0].conditional_on == frozenset({"c0"})
        assert conditional[0].kind == "play"

    def test_nested_ifbit_accumulates_cbits(self):
        """Nested IfBit(c0, IfBit(c1, body)) tags body with {c0, c1}."""
        cfg = Config(
            frames=frozenset(["d0", "m0", "m1"]),
            ports=frozenset(["p_drive", "p_meas0", "p_meas1"]),
            port_of={"d0": "p_drive", "m0": "p_meas0", "m1": "p_meas1"},
            init_freq={"d0": 5.0e-3, "m0": 7.0e-3, "m1": 7.0e-3},
            init_phase={"d0": 0.0, "m0": 0.0, "m1": 0.0},
        )
        prog = [
            Acquire("m0", duration=1000, cbit="c0"),
            Acquire("m1", duration=1000, cbit="c1"),
            Delay(duration=1000, frame="d0"),
            IfBit("c0", IfBit("c1", Play("d0", Waveform("x", 160)))),
        ]
        events = lower_to_schedule(prog, cfg)
        conditional = [ev for ev in events if ev.conditional_on]
        assert len(conditional) == 1
        assert conditional[0].conditional_on == frozenset({"c0", "c1"})
        # Compiled-mode check should pass (d0 time=1000, both cbits ready at 1000)
        ok, _ = check_feedback_causality(prog, cfg, compiled_events=events)
        assert ok

    def test_nested_ifbit_catches_outer_cbit_not_ready(self):
        """Nested IfBit where outer cbit is not ready should FAIL compiled mode."""
        cfg = Config(
            frames=frozenset(["d0", "m0", "m1"]),
            ports=frozenset(["p_drive", "p_meas0", "p_meas1"]),
            port_of={"d0": "p_drive", "m0": "p_meas0", "m1": "p_meas1"},
            init_freq={"d0": 5.0e-3, "m0": 7.0e-3, "m1": 7.0e-3},
            init_phase={"d0": 0.0, "m0": 0.0, "m1": 0.0},
        )
        # c1 acquired at t=0 (ready at 500), c0 acquired at t=0 (ready at 2000)
        # IfBit body on d0 at t=0 — both c0 and c1 not ready
        prog = [
            Acquire("m0", duration=2000, cbit="c0"),
            Acquire("m1", duration=500, cbit="c1"),
            IfBit("c0", IfBit("c1", Play("d0", Waveform("x", 160)))),
        ]
        events = lower_to_schedule(prog, cfg)
        # Compiled mode should catch: Play at t=0, but c0 not ready until 2000
        ok, errors = check_feedback_causality(prog, cfg, compiled_events=events)
        assert not ok
        assert any("c0" in e and "not ready" in e for e in errors)
