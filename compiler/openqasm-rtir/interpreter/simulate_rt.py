"""Print a human-readable timeline table from a list of RTEvents."""

from __future__ import annotations
from rt_ir.ir import RTEvent


def print_timeline(events: list[RTEvent]) -> None:
    """Print a formatted timeline table to stdout."""
    if not events:
        print("  (no events)")
        return

    hdr = (
        f"{'ID':>4s}  {'Kind':7s}  {'Start':>5s}  {'End':>5s}  "
        f"{'Dur':>4s}  {'Resource':<14s}  {'Q':>2s}  "
        f"{'Creg':>4s}  {'Payload':<14s}  {'Cond':<12s}  {'Deps'}"
    )
    print(hdr)
    print("-" * len(hdr))
    for e in events:
        creg = str(e.creg) if e.creg is not None else "-"
        cond = e.condition or "-"
        deps = ",".join(str(d) for d in e.depends_on) if e.depends_on else "-"
        print(
            f"{e.event_id:4d}  {e.kind:7s}  {e.start:5d}  {e.end:5d}  "
            f"{e.duration:4d}  {e.resource:<14s}  {e.qubit:2d}  "
            f"{creg:>4s}  {e.payload:<14s}  {cond:<12s}  {deps}"
        )
