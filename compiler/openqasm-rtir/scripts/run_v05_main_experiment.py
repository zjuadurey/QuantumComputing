"""Run the v0.5 main experiment and export structured results.

This script extracts the three-layer evidence we use in the paper:
1. Boolean FullContract verdicts
2. Quantitative checker diagnostics
3. External Qiskit Dynamics fidelity

Outputs:
    results/v05_main_experiment.json
    results/v05_main_experiment.csv
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pulse_ir.ir import Config, Waveform, Play, ShiftPhase, Delay, Acquire, IfBit
from pulse_lowering.lower_to_schedule import lower_to_schedule
from pulse_lowering.verify import verify_lowering
from pulse_lowering.buggy_variants import (
    lower_buggy_drop_phase,
    lower_buggy_early_feedback,
    lower_buggy_ignore_shared_port,
    lower_buggy_reorder_ports,
)
from pulse_external.qiskit_dynamics import (
    compare_schedule_lowerings,
    compare_single_frame_lowerings,
)


RESULTS_DIR = ROOT / "results"
JSON_PATH = RESULTS_DIR / "v05_main_experiment.json"
CSV_PATH = RESULTS_DIR / "v05_main_experiment.csv"


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


def _bool_str(value: bool) -> str:
    return "Pass" if value else "Fail"


def _run_correct_baseline() -> dict[str, object]:
    cfg = _config_1frame_with_shift()
    prog = _program_with_shift()
    report = verify_lowering(prog, cfg, lower=lower_to_schedule)
    witness = compare_single_frame_lowerings(
        prog,
        cfg,
        frame="d0",
        lower_candidate=lower_to_schedule,
        drive_scale=0.02,
    )
    return {
        "fault_family": "correct_lowering",
        "wf": _bool_str(report.well_formed),
        "port_excl": _bool_str(report.port_exclusive),
        "feedback_causal": _bool_str(report.feedback_causal),
        "frame_consist": _bool_str(report.frame_consistent),
        "diagnostic_summary": "none",
        "max_overlap_dt": report.port_diagnostics.max_overlap_dt,
        "total_overlap_dt": report.port_diagnostics.total_overlap_dt,
        "max_earliness_dt": report.feedback_diagnostics.max_earliness_dt,
        "total_earliness_dt": report.feedback_diagnostics.total_earliness_dt,
        "max_abs_time_diff": report.frame_diagnostics.max_abs_time_diff,
        "max_abs_phase_diff": report.frame_diagnostics.max_abs_phase_diff,
        "witness_type": "phase-sensitive single-frame",
        "fidelity": witness.fidelity,
    }


def _run_drop_phase() -> dict[str, object]:
    cfg = _config_1frame_with_shift()
    prog = _program_with_shift()
    report = verify_lowering(prog, cfg, lower=lower_buggy_drop_phase)
    witness = compare_single_frame_lowerings(
        prog,
        cfg,
        frame="d0",
        lower_candidate=lower_buggy_drop_phase,
        drive_scale=0.02,
    )
    return {
        "fault_family": "drop_phase",
        "wf": _bool_str(report.well_formed),
        "port_excl": _bool_str(report.port_exclusive),
        "feedback_causal": _bool_str(report.feedback_causal),
        "frame_consist": _bool_str(report.frame_consistent),
        "diagnostic_summary": "phase drift = pi/2",
        "max_overlap_dt": report.port_diagnostics.max_overlap_dt,
        "total_overlap_dt": report.port_diagnostics.total_overlap_dt,
        "max_earliness_dt": report.feedback_diagnostics.max_earliness_dt,
        "total_earliness_dt": report.feedback_diagnostics.total_earliness_dt,
        "max_abs_time_diff": report.frame_diagnostics.max_abs_time_diff,
        "max_abs_phase_diff": report.frame_diagnostics.max_abs_phase_diff,
        "witness_type": "phase-sensitive single-frame",
        "fidelity": witness.fidelity,
    }


def _run_ignore_shared_port() -> dict[str, object]:
    cfg = _config_shared_port()
    prog = _program_shared_port_phase_diverse()
    report = verify_lowering(prog, cfg, lower=lower_buggy_ignore_shared_port)
    witness = compare_schedule_lowerings(
        prog,
        cfg,
        scope="shared-port",
        lower_candidate=lower_buggy_ignore_shared_port,
        drive_scale=0.02,
    )
    return {
        "fault_family": "ignore_shared_port",
        "wf": _bool_str(report.well_formed),
        "port_excl": _bool_str(report.port_exclusive),
        "feedback_causal": _bool_str(report.feedback_causal),
        "frame_consist": _bool_str(report.frame_consistent),
        "diagnostic_summary": "max overlap = 160 dt; time drift = 160 dt",
        "max_overlap_dt": report.port_diagnostics.max_overlap_dt,
        "total_overlap_dt": report.port_diagnostics.total_overlap_dt,
        "max_earliness_dt": report.feedback_diagnostics.max_earliness_dt,
        "total_earliness_dt": report.feedback_diagnostics.total_earliness_dt,
        "max_abs_time_diff": report.frame_diagnostics.max_abs_time_diff,
        "max_abs_phase_diff": report.frame_diagnostics.max_abs_phase_diff,
        "witness_type": "shared-port schedule witness",
        "fidelity": witness.fidelity,
    }


def _run_reorder_ports() -> dict[str, object]:
    cfg = _config_shared_port()
    prog = _program_shared_port_phase_diverse()
    report = verify_lowering(prog, cfg, lower=lower_buggy_reorder_ports)
    witness = compare_schedule_lowerings(
        prog,
        cfg,
        scope="shared-port",
        lower_candidate=lower_buggy_reorder_ports,
        drive_scale=0.02,
    )
    return {
        "fault_family": "reorder_ports",
        "wf": _bool_str(report.well_formed),
        "port_excl": _bool_str(report.port_exclusive),
        "feedback_causal": _bool_str(report.feedback_causal),
        "frame_consist": _bool_str(report.frame_consistent),
        "diagnostic_summary": "max overlap = 160 dt; time drift = 160 dt",
        "max_overlap_dt": report.port_diagnostics.max_overlap_dt,
        "total_overlap_dt": report.port_diagnostics.total_overlap_dt,
        "max_earliness_dt": report.feedback_diagnostics.max_earliness_dt,
        "total_earliness_dt": report.feedback_diagnostics.total_earliness_dt,
        "max_abs_time_diff": report.frame_diagnostics.max_abs_time_diff,
        "max_abs_phase_diff": report.frame_diagnostics.max_abs_phase_diff,
        "witness_type": "shared-port schedule witness",
        "fidelity": witness.fidelity,
    }


def _run_early_feedback() -> dict[str, object]:
    cfg = _config_2frame()
    prog = _program_with_multi_delay_feedback()
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
    return {
        "fault_family": "early_feedback",
        "wf": _bool_str(report.well_formed),
        "port_excl": _bool_str(report.port_exclusive),
        "feedback_causal": _bool_str(report.feedback_causal),
        "frame_consist": _bool_str(report.frame_consistent),
        "diagnostic_summary": "earliness = 840 dt",
        "max_overlap_dt": report.port_diagnostics.max_overlap_dt,
        "total_overlap_dt": report.port_diagnostics.total_overlap_dt,
        "max_earliness_dt": report.feedback_diagnostics.max_earliness_dt,
        "total_earliness_dt": report.feedback_diagnostics.total_earliness_dt,
        "max_abs_time_diff": report.frame_diagnostics.max_abs_time_diff,
        "max_abs_phase_diff": report.frame_diagnostics.max_abs_phase_diff,
        "witness_type": "timing-sensitive witness with drift",
        "fidelity": witness.fidelity,
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = [
        _run_correct_baseline(),
        _run_drop_phase(),
        _run_ignore_shared_port(),
        _run_reorder_ports(),
        _run_early_feedback(),
    ]

    with JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=True)

    fieldnames = [
        "fault_family",
        "wf",
        "port_excl",
        "feedback_causal",
        "frame_consist",
        "diagnostic_summary",
        "max_overlap_dt",
        "total_overlap_dt",
        "max_earliness_dt",
        "total_earliness_dt",
        "max_abs_time_diff",
        "max_abs_phase_diff",
        "witness_type",
        "fidelity",
    ]
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {JSON_PATH}")
    print(f"Wrote {CSV_PATH}")
    for row in rows:
        print(
            f"{row['fault_family']}: "
            f"WF={row['wf']}, PortExcl={row['port_excl']}, "
            f"Fb={row['feedback_causal']}, Frame={row['frame_consist']}, "
            f"diag={row['diagnostic_summary']}, fidelity={row['fidelity']:.6f}"
        )


if __name__ == "__main__":
    main()
