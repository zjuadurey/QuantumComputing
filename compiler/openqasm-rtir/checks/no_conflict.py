"""Check that no two events overlap on the same resource."""

from __future__ import annotations
from collections import defaultdict
from rt_ir.ir import RTEvent


def check_no_conflict(events: list[RTEvent]) -> tuple[bool, list[str]]:
    """Return (ok, errors).  ok=True means no overlapping intervals on any resource."""
    by_resource: dict[str, list[RTEvent]] = defaultdict(list)
    for e in events:
        by_resource[e.resource].append(e)

    errors: list[str] = []
    for res, evts in by_resource.items():
        evts.sort(key=lambda e: e.start)
        for i in range(len(evts) - 1):
            a, b = evts[i], evts[i + 1]
            if a.end > b.start:
                errors.append(
                    f"conflict on {res}: E{a.event_id} [{a.start}..{a.end}) "
                    f"overlaps E{b.event_id} [{b.start}..{b.end})"
                )

    return (len(errors) == 0, errors)
