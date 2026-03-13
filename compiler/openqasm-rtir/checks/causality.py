"""Check that every depends_on dependency finishes before the dependent starts."""

from __future__ import annotations
from rt_ir.ir import RTEvent


def check_causality(events: list[RTEvent]) -> tuple[bool, list[str]]:
    """Return (ok, errors).  ok=True means all causal deps are satisfied."""
    by_id: dict[int, RTEvent] = {e.event_id: e for e in events}
    errors: list[str] = []

    for e in events:
        for dep_id in e.depends_on:
            dep = by_id.get(dep_id)
            if dep is None:
                errors.append(
                    f"E{e.event_id} depends on E{dep_id}, which does not exist"
                )
                continue
            if dep.end > e.start:
                errors.append(
                    f"causality violation: E{e.event_id} (start={e.start}) "
                    f"depends on E{dep_id} (end={dep.end})"
                )

    return (len(errors) == 0, errors)
