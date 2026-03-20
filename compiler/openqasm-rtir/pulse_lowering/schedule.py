"""PulseEvent — the output of lowering.

A PulseEvent is a fully resolved, explicit-time record.
Each event has concrete start, end, port, frame, and phase snapshot.
This is what a "compiled schedule" looks like.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PulseEvent:
    event_id: int
    kind: str               # play | acquire | shift_phase | delay
    frame: str
    port: str | None        # None for shift_phase and delay (no port activity)
    start: int              # in dt
    end: int                # in dt (= start for zero-duration events)
    phase_before: float     # frame phase at entry
    phase_after: float      # frame phase at exit
    cbit: str | None = None                      # only for acquire
    payload: str = ""                            # waveform name, angle value, etc.
    conditional_on: frozenset[str] = frozenset() # cbits this event depends on (from IfBit nesting)

    @property
    def duration(self) -> int:
        return self.end - self.start
