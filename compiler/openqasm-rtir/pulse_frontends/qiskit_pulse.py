"""Restricted Qiskit Pulse -> core IR adapter.

This module intentionally targets a small, stable subset of legacy
``qiskit.pulse`` artifacts that still exist in Qiskit 1.3:

- Play
- Delay
- ShiftPhase
- Acquire

The adapter does not try to preserve every Qiskit construct. Instead, it
extracts the linearized ``schedule.instructions`` view and converts it into
the paper's core ``PulseStmt`` list plus a matching ``Config``. Once
translated, the program enters the existing verification pipeline unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pulse_ir.ir import Acquire, Config, Delay, Play, PulseStmt, ShiftPhase, Waveform
from pulse_lowering.verify import VerificationReport, verify_lowering


@dataclass(frozen=True)
class QiskitTranslation:
    """Result of translating a Qiskit ScheduleBlock into the core IR."""

    program: list[PulseStmt]
    config: Config


def _channel_prefix(channel: Any) -> str:
    name = type(channel).__name__
    if name == "DriveChannel":
        return "d"
    if name == "AcquireChannel":
        return "m"
    if name == "MeasureChannel":
        return "r"
    if name == "ControlChannel":
        return "u"
    raise ValueError(f"unsupported Qiskit channel type: {name}")


def _frame_name(channel: Any) -> str:
    return f"{_channel_prefix(channel)}{channel.index}"


def _default_cbit_name(mem_slot: Any) -> str:
    return f"c{mem_slot.index}"


def _waveform_name(pulse_obj: Any) -> str:
    pulse_name = getattr(pulse_obj, "name", None)
    if pulse_name:
        return pulse_name
    pulse_type = getattr(pulse_obj, "pulse_type", None)
    if pulse_type:
        return f"{pulse_type}_{pulse_obj.duration}"
    return f"{type(pulse_obj).__name__}_{pulse_obj.duration}"


def translate_qiskit_scheduleblock(
    schedule: Any,
    *,
    port_of: dict[str, str] | None = None,
    init_freq: dict[str, float] | None = None,
    init_phase: dict[str, float] | None = None,
    frame_aliases: dict[str, str] | None = None,
    cbit_aliases: dict[str, str] | None = None,
) -> QiskitTranslation:
    """Translate a restricted Qiskit ``ScheduleBlock`` into core IR.

    The translation uses the explicit ``schedule.instructions`` timeline.
    If a frame-local gap exists before the next instruction on that frame,
    a core ``Delay`` is inserted to preserve the absolute start time.
    """

    if not hasattr(schedule, "instructions"):
        raise TypeError("expected a Qiskit pulse ScheduleBlock-like object with .instructions")

    frame_aliases = frame_aliases or {}
    cbit_aliases = cbit_aliases or {}

    program: list[PulseStmt] = []
    seen_frames: set[str] = set()
    frame_time: dict[str, int] = {}

    for start, inst in schedule.instructions:
        inst_type = type(inst).__name__
        channel = getattr(inst, "channel", None)
        if channel is None:
            raise ValueError(f"unsupported instruction without channel: {inst!r}")

        raw_frame = _frame_name(channel)
        frame = frame_aliases.get(raw_frame, raw_frame)
        seen_frames.add(frame)
        current_time = frame_time.get(frame, 0)

        if start < current_time:
            raise ValueError(
                f"instruction {inst!r} starts at t={start} before frame {frame} "
                f"is available at t={current_time}"
            )
        if start > current_time:
            program.append(Delay(start - current_time, frame))
            current_time = start

        if inst_type == "Play":
            pulse_obj = inst.pulse
            program.append(
                Play(
                    frame,
                    Waveform(_waveform_name(pulse_obj), pulse_obj.duration),
                )
            )
            frame_time[frame] = current_time + inst.duration
            continue

        if inst_type == "Delay":
            program.append(Delay(inst.duration, frame))
            frame_time[frame] = current_time + inst.duration
            continue

        if inst_type == "ShiftPhase":
            program.append(ShiftPhase(frame, inst.phase))
            frame_time[frame] = current_time
            continue

        if inst_type == "Acquire":
            mem_slot = inst.mem_slot
            raw_cbit = _default_cbit_name(mem_slot)
            cbit = cbit_aliases.get(raw_cbit, raw_cbit)
            program.append(Acquire(frame, inst.duration, cbit))
            frame_time[frame] = current_time + inst.duration
            continue

        raise ValueError(f"unsupported Qiskit pulse instruction: {inst_type}")

    ports = set()
    resolved_port_of: dict[str, str] = {}
    for frame in seen_frames:
        port = port_of.get(frame, frame) if port_of else frame
        resolved_port_of[frame] = port
        ports.add(port)

    resolved_init_freq = {frame: 0.0 for frame in seen_frames}
    if init_freq:
        resolved_init_freq.update(init_freq)
    resolved_init_phase = {frame: 0.0 for frame in seen_frames}
    if init_phase:
        resolved_init_phase.update(init_phase)

    config = Config(
        frames=frozenset(seen_frames),
        ports=frozenset(ports),
        port_of=resolved_port_of,
        init_freq=resolved_init_freq,
        init_phase=resolved_init_phase,
    )
    return QiskitTranslation(program=program, config=config)


def verify_qiskit_scheduleblock(
    schedule: Any,
    *,
    port_of: dict[str, str] | None = None,
    init_freq: dict[str, float] | None = None,
    init_phase: dict[str, float] | None = None,
    frame_aliases: dict[str, str] | None = None,
    cbit_aliases: dict[str, str] | None = None,
) -> tuple[QiskitTranslation, VerificationReport]:
    """Translate a Qiskit schedule and run the existing verification pipeline."""

    translation = translate_qiskit_scheduleblock(
        schedule,
        port_of=port_of,
        init_freq=init_freq,
        init_phase=init_phase,
        frame_aliases=frame_aliases,
        cbit_aliases=cbit_aliases,
    )
    report = verify_lowering(translation.program, translation.config)
    return translation, report
