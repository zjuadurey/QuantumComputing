"""Schedule-level feedback causality checker.

v0.4 splits feedback reasoning into two layers:
- Source-side legality is checked by pulse_checks.wellformedness
- Schedule-side causality is checked here on lowered PulseEvent records

Property:
    FeedbackCausal_sched(S) ≡ ∀ event e ∈ S with conditional_on ⊇ {c₁..cₖ},
                                 e.start ≥ cbit_ready(cᵢ) for all cᵢ

where cbit_ready is derived from acquire events in the same schedule.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pulse_lowering.schedule import PulseEvent


@dataclass(frozen=True)
class FeedbackViolation:
    event_id: int
    kind: str
    frame: str
    cbit: str
    start: int
    ready_time: int | None
    earliness_dt: int | None


@dataclass(frozen=True)
class FeedbackCausalityDiagnostics:
    violations: list[FeedbackViolation] = field(default_factory=list)
    num_violations: int = 0
    num_missing_cbits: int = 0
    max_earliness_dt: int = 0
    total_earliness_dt: int = 0


def check_schedule_causality(
    events: list[PulseEvent],
) -> tuple[bool, list[str]]:
    """Check event-level feedback causality on a lowered schedule."""
    diagnostics = diagnose_schedule_causality(events)
    errors = format_schedule_causality_errors(diagnostics)
    return (len(errors) == 0, errors)


def diagnose_schedule_causality(
    events: list[PulseEvent],
) -> FeedbackCausalityDiagnostics:
    """Return structured schedule-level causality diagnostics."""
    cbit_ready: dict[str, int] = {}
    for ev in events:
        if ev.kind == "acquire" and ev.cbit is not None:
            cbit_ready[ev.cbit] = ev.end

    violations: list[FeedbackViolation] = []
    num_missing_cbits = 0
    max_earliness_dt = 0
    total_earliness_dt = 0

    for ev in events:
        if ev.conditional_on:
            for cbit in ev.conditional_on:
                t_ready = cbit_ready.get(cbit)
                if t_ready is None:
                    violations.append(FeedbackViolation(
                        event_id=ev.event_id,
                        kind=ev.kind,
                        frame=ev.frame,
                        cbit=cbit,
                        start=ev.start,
                        ready_time=None,
                        earliness_dt=None,
                    ))
                    num_missing_cbits += 1
                elif ev.start < t_ready:
                    earliness_dt = t_ready - ev.start
                    violations.append(FeedbackViolation(
                        event_id=ev.event_id,
                        kind=ev.kind,
                        frame=ev.frame,
                        cbit=cbit,
                        start=ev.start,
                        ready_time=t_ready,
                        earliness_dt=earliness_dt,
                    ))
                    max_earliness_dt = max(max_earliness_dt, earliness_dt)
                    total_earliness_dt += earliness_dt

    return FeedbackCausalityDiagnostics(
        violations=violations,
        num_violations=len(violations),
        num_missing_cbits=num_missing_cbits,
        max_earliness_dt=max_earliness_dt,
        total_earliness_dt=total_earliness_dt,
    )


def format_schedule_causality_errors(
    diagnostics: FeedbackCausalityDiagnostics,
) -> list[str]:
    """Render human-readable feedback-causality errors."""
    errors: list[str] = []
    for violation in diagnostics.violations:
        if violation.ready_time is None:
            errors.append(
                f"Event {violation.event_id} ({violation.kind} on {violation.frame}): "
                f"conditional on {violation.cbit} but cbit was never acquired"
            )
        else:
            errors.append(
                f"Event {violation.event_id} ({violation.kind} on {violation.frame}): "
                f"starts at t={violation.start} but {violation.cbit} not ready "
                f"until t={violation.ready_time} (earliness_dt={violation.earliness_dt})"
            )
    return errors
