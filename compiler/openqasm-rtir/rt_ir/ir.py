"""Minimal real-time IR event definition."""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class RTEvent:
    event_id: int
    kind: str                        # gate | delay | measure | branch
    start: int                       # start time in dt
    duration: int                    # duration in dt
    resource: str                    # e.g. drive_q0, measure_q0
    qubit: int
    creg: int | None = None          # classical bit index (measure / branch)
    condition: str | None = None     # e.g. "c[0]==1"
    payload: str = ""                # gate name, delay amount, etc.
    depends_on: list[int] = field(default_factory=list)

    @property
    def end(self) -> int:
        return self.start + self.duration

    def __repr__(self) -> str:
        dep = f" dep={self.depends_on}" if self.depends_on else ""
        cond = f" cond={self.condition}" if self.condition else ""
        return (
            f"E{self.event_id:02d} [{self.kind:7s}] "
            f"t={self.start:>4d}..{self.end:<4d} "
            f"res={self.resource:<12s} q={self.qubit} "
            f"payload={self.payload}{cond}{dep}"
        )
