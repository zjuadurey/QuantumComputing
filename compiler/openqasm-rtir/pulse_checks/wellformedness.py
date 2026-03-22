"""Well-Formedness precheck for pulse programs.

WF(P, C) checks that a source program is admissible before execution:
1. Every cbit used in IfBit must have been defined by a prior Acquire.
2. At each IfBit encounter, the REAL start time of the guarded body event
   must be >= cbit_ready for ALL dependency bits.
3. The body start is port-aware: Play/Acquire may be delayed by port_time.

This is a SOURCE-LEVEL check. It independently walks the AST and tracks
frame times using the same recurrence as the oracle but in separate code.
It does NOT import ref_semantics.
"""

from __future__ import annotations

from pulse_ir.ir import (
    Config,
    Play,
    Acquire,
    ShiftPhase,
    Delay,
    IfBit,
    PulseStmt,
)

def check_wellformedness(
    program: list[PulseStmt],
    config: Config,
) -> tuple[bool, list[str]]:
    """Check that a program is well-formed.

    Returns (ok, errors).
    """
    errors: list[str] = []
    frame_time: dict[str, int] = {f: 0 for f in config.frames}
    port_time: dict[str, int] = {p: 0 for p in config.ports}
    cbit_ready: dict[str, int] = {}
    defined_cbits: set[str] = set()

    _walk(program, frame_time, port_time, cbit_ready, defined_cbits,
          config, errors, frozenset())

    return (len(errors) == 0, errors)


def _walk(
    stmts: list[PulseStmt],
    frame_time: dict[str, int],
    port_time: dict[str, int],
    cbit_ready: dict[str, int],
    defined_cbits: set[str],
    config: Config,
    errors: list[str],
    active_cbits: frozenset[str],
) -> None:
    """Walk program AST, tracking time independently (port-aware)."""
    for stmt in stmts:
        match stmt:
            case Play(frame, waveform):
                d = waveform.duration
                p = config.port_of[frame]
                start = max(frame_time[frame], port_time[p])
                end = start + d
                frame_time[frame] = end
                port_time[p] = end

            case Acquire(frame, duration, cbit):
                p = config.port_of[frame]
                start = max(frame_time[frame], port_time[p])
                end = start + duration
                frame_time[frame] = end
                port_time[p] = end
                cbit_ready[cbit] = end
                defined_cbits.add(cbit)

            case ShiftPhase():
                pass  # zero duration

            case Delay(duration, frame):
                frame_time[frame] += duration

            case IfBit(cbit, body):
                # Check 1: cbit must be defined by prior Acquire
                if cbit not in defined_cbits:
                    errors.append(
                        f"IfBit({cbit}): cbit was never acquired"
                    )
                else:
                    # Check 2: real body start at encounter >= cbit_ready
                    body_frame = _body_frame(body)
                    t_use = _body_start(body, frame_time, port_time, config)
                    if body_frame is not None and t_use is not None:
                        t_ready = cbit_ready[cbit]
                        if t_use < t_ready:
                            errors.append(
                                f"IfBit({cbit}): body on frame {body_frame} "
                                f"at t={t_use} but cbit not ready until "
                                f"t={t_ready}"
                            )

                    # Check 3: nested — all ancestor cbits must also be ready
                    for dep_cbit in active_cbits:
                        if dep_cbit in cbit_ready and body_frame is not None and t_use is not None:
                            t_ready = cbit_ready[dep_cbit]
                            if t_use < t_ready:
                                errors.append(
                                    f"IfBit({cbit}): ancestor dep "
                                    f"{dep_cbit} not ready at t={t_use} "
                                    f"(ready at t={t_ready})"
                                )

                # Walk body (always-taken for timing)
                _walk([body], frame_time, port_time, cbit_ready,
                      defined_cbits, config, errors,
                      active_cbits | {cbit})


def _body_frame(stmt: PulseStmt) -> str | None:
    """Extract the frame that a statement operates on."""
    match stmt:
        case Play(frame, _):
            return frame
        case Acquire(frame, _, _):
            return frame
        case ShiftPhase(frame, _):
            return frame
        case Delay(_, frame):
            return frame
        case IfBit(_, body):
            return _body_frame(body)
    return None


def _body_start(
    stmt: PulseStmt,
    frame_time: dict[str, int],
    port_time: dict[str, int],
    config: Config,
) -> int | None:
    """Compute the real start time of the guarded body event at encounter.

    This mirrors the oracle/lowering recurrence without executing the body:
    Play/Acquire may be delayed by port availability, while Delay/ShiftPhase
    start at the current frame time. Nested IfBit delegates to its body.
    """
    match stmt:
        case Play(frame, _):
            p = config.port_of[frame]
            return max(frame_time[frame], port_time[p])
        case Acquire(frame, _, _):
            p = config.port_of[frame]
            return max(frame_time[frame], port_time[p])
        case ShiftPhase(frame, _):
            return frame_time[frame]
        case Delay(_, frame):
            return frame_time[frame]
        case IfBit(_, body):
            return _body_start(body, frame_time, port_time, config)
    return None
