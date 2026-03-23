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

from pulse_lowering.schedule import PulseEvent


def check_schedule_causality(
    events: list[PulseEvent],
) -> tuple[bool, list[str]]:
    """Check event-level feedback causality on a lowered schedule."""
    errors: list[str] = []

    # Build cbit_ready from acquire events.
    cbit_ready: dict[str, int] = {}
    for ev in events:
        if ev.kind == "acquire" and ev.cbit is not None:
            cbit_ready[ev.cbit] = ev.end

    # Check that every conditional event waits for all dependency bits.
    for ev in events:
        if ev.conditional_on:
            for cbit in ev.conditional_on:
                t_ready = cbit_ready.get(cbit)
                if t_ready is None:
                    errors.append(
                        f"Event {ev.event_id} ({ev.kind} on {ev.frame}): "
                        f"conditional on {cbit} but cbit was never acquired"
                    )
                elif ev.start < t_ready:
                    errors.append(
                        f"Event {ev.event_id} ({ev.kind} on {ev.frame}): "
                        f"starts at t={ev.start} but {cbit} not ready "
                        f"until t={t_ready}"
                    )

    return (len(errors) == 0, errors)
