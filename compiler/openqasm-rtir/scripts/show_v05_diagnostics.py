"""Show v0.5 checker diagnostics alongside Qiskit Dynamics fidelity.

Run with:
    conda run -n qiskit_qasm_py312 --no-capture-output \
        python scripts/show_v05_diagnostics.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pulse_ir.ir import Config, Waveform, Play, ShiftPhase, Delay, Acquire, IfBit
from pulse_lowering.lower_to_schedule import lower_to_schedule
from pulse_lowering.buggy_variants import (
    lower_buggy_drop_phase,
    lower_buggy_early_feedback,
    lower_buggy_ignore_shared_port,
)
from pulse_lowering.verify import verify_lowering
from pulse_external.qiskit_dynamics import (
    compare_schedule_lowerings,
    compare_single_frame_lowerings,
)


def _fmt_event(ev) -> str:
    cond = ""
    if ev.conditional_on:
        cond = f", cond={sorted(ev.conditional_on)}"
    port = f", port={ev.port}" if ev.port is not None else ""
    payload = f", payload={ev.payload}" if ev.payload else ""
    return (
        f"#{ev.event_id} {ev.kind}({ev.frame}{port}) "
        f"[{ev.start},{ev.end}) "
        f"phase {ev.phase_before:.3f}->{ev.phase_after:.3f}"
        f"{cond}{payload}"
    )


def _print_schedule(title: str, events: list) -> None:
    print(title)
    for ev in events:
        print(f"  {_fmt_event(ev)}")


def _show_drop_phase() -> None:
    cfg = Config(
        frames=frozenset(["d0"]),
        ports=frozenset(["p0"]),
        port_of={"d0": "p0"},
        init_freq={"d0": 5.0e-3},
        init_phase={"d0": 0.0},
    )
    prog = [
        Play("d0", Waveform("g160", 160)),
        ShiftPhase("d0", math.pi / 2),
        Play("d0", Waveform("g160", 160)),
    ]
    correct_events = lower_to_schedule(prog, cfg)
    report = verify_lowering(prog, cfg, lower=lower_buggy_drop_phase)
    witness = compare_single_frame_lowerings(
        prog,
        cfg,
        frame="d0",
        lower_candidate=lower_buggy_drop_phase,
        drive_scale=0.02,
    )

    print("=== drop_phase ===")
    _print_schedule("correct schedule:", correct_events)
    _print_schedule("buggy schedule:", report.events)
    print(f"checker verdict: FrameConsist={report.frame_consistent}")
    for mismatch in report.frame_diagnostics.mismatches:
        print(
            "  mismatch: "
            f"frame={mismatch.frame}, "
            f"expected_phase={mismatch.expected_phase:.6f}, "
            f"actual_phase={mismatch.actual_phase:.6f}, "
            f"phase_diff={mismatch.phase_diff:.6f}"
        )
    print(
        "diagnostics: "
        f"max_abs_phase_diff={report.frame_diagnostics.max_abs_phase_diff:.6f}, "
        f"max_abs_time_diff={report.frame_diagnostics.max_abs_time_diff}"
    )
    print(f"dynamics fidelity: {witness.fidelity:.6f}")
    print()


def _show_ignore_shared_port() -> None:
    cfg = Config(
        frames=frozenset(["d0", "d1"]),
        ports=frozenset(["p0"]),
        port_of={"d0": "p0", "d1": "p0"},
        init_freq={"d0": 0.0, "d1": 0.0},
        init_phase={"d0": 0.0, "d1": 0.0},
    )
    prog = [
        Play("d0", Waveform("g160", 160)),
        ShiftPhase("d1", math.pi / 2),
        Play("d1", Waveform("g200", 200)),
    ]
    correct_events = lower_to_schedule(prog, cfg)
    report = verify_lowering(prog, cfg, lower=lower_buggy_ignore_shared_port)
    witness = compare_schedule_lowerings(
        prog,
        cfg,
        scope="shared-port",
        lower_candidate=lower_buggy_ignore_shared_port,
        drive_scale=0.02,
    )

    print("=== ignore_shared_port ===")
    _print_schedule("correct schedule:", correct_events)
    _print_schedule("buggy schedule:", report.events)
    print(
        "checker verdicts: "
        f"PortExcl={report.port_exclusive}, "
        f"FrameConsist={report.frame_consistent}"
    )
    for overlap in report.port_diagnostics.overlaps:
        print(
            "  overlap: "
            f"port={overlap.port}, "
            f"first={overlap.first_interval}, "
            f"second={overlap.second_interval}, "
            f"overlap_dt={overlap.overlap_dt}"
        )
    for mismatch in report.frame_diagnostics.mismatches:
        print(
            "  mismatch: "
            f"frame={mismatch.frame}, "
            f"expected_time={mismatch.expected_time}, "
            f"actual_time={mismatch.actual_time}, "
            f"time_diff={mismatch.time_diff}"
        )
    print(
        "diagnostics: "
        f"max_overlap_dt={report.port_diagnostics.max_overlap_dt}, "
        f"total_overlap_dt={report.port_diagnostics.total_overlap_dt}, "
        f"max_abs_time_diff={report.frame_diagnostics.max_abs_time_diff}"
    )
    print(f"dynamics fidelity: {witness.fidelity:.6f}")
    print()


def _show_early_feedback() -> None:
    cfg = Config(
        frames=frozenset(["d0", "m0"]),
        ports=frozenset(["p_drive", "p_meas"]),
        port_of={"d0": "p_drive", "m0": "p_meas"},
        init_freq={"d0": 5.0e-3, "m0": 7.0e-3},
        init_phase={"d0": 0.0, "m0": 0.0},
    )
    prog = [
        Play("d0", Waveform("x_pulse", 160)),
        Acquire("m0", duration=1000, cbit="c0"),
        Delay(duration=400, frame="d0"),
        Delay(duration=600, frame="d0"),
        IfBit("c0", Play("d0", Waveform("x_pulse", 160))),
    ]
    correct_events = lower_to_schedule(prog, cfg)
    report = verify_lowering(prog, cfg, lower=lower_buggy_early_feedback)
    witness = compare_schedule_lowerings(
        prog,
        cfg,
        scope="d0-timing",
        frames=["d0"],
        lower_candidate=lower_buggy_early_feedback,
        drive_scale=0.02,
        static_drift=0.01,
    )

    print("=== early_feedback ===")
    _print_schedule("correct schedule:", correct_events)
    _print_schedule("buggy schedule:", report.events)
    print(f"checker verdict: FeedbackCausal={report.feedback_causal}")
    for violation in report.feedback_diagnostics.violations:
        print(
            "  violation: "
            f"event_id={violation.event_id}, "
            f"frame={violation.frame}, "
            f"cbit={violation.cbit}, "
            f"start={violation.start}, "
            f"ready_time={violation.ready_time}, "
            f"earliness_dt={violation.earliness_dt}"
        )
    print(
        "diagnostics: "
        f"max_earliness_dt={report.feedback_diagnostics.max_earliness_dt}, "
        f"total_earliness_dt={report.feedback_diagnostics.total_earliness_dt}"
    )
    print(f"dynamics fidelity: {witness.fidelity:.6f}")
    print()


def main() -> None:
    _show_drop_phase()
    _show_ignore_shared_port()
    _show_early_feedback()


if __name__ == "__main__":
    main()
