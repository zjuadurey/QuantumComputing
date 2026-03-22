"""Pulse-level IR data types.

Follows v04_fullcontract_spec.md:
- PulseStmt ::= Play | Acquire | ShiftPhase | Delay | IfBit
- Config    = static hardware description (frames, ports, mappings)
- FrameState = mutable execution state (time, phase, port_time, cbit, occupancy)
- Waveform  = named envelope with duration
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Waveform — named envelope with a duration (content is opaque)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Waveform:
    name: str
    duration: int  # in dt


# ---------------------------------------------------------------------------
# PulseStmt — abstract syntax (union of 5 variants)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Play:
    frame: str
    waveform: Waveform


@dataclass(frozen=True)
class Acquire:
    frame: str
    duration: int
    cbit: str


@dataclass(frozen=True)
class ShiftPhase:
    frame: str
    angle: float  # radians


@dataclass(frozen=True)
class Delay:
    duration: int
    frame: str


@dataclass(frozen=True)
class IfBit:
    cbit: str
    body: PulseStmt


# Union type for type hints
PulseStmt = Play | Acquire | ShiftPhase | Delay | IfBit


# ---------------------------------------------------------------------------
# Config — static hardware description (immutable after init)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    frames: frozenset[str]
    ports: frozenset[str]
    port_of: dict[str, str]       # frame → port (not injective)
    init_freq: dict[str, float]   # frame → frequency (Hz-like, unitless in model)
    init_phase: dict[str, float]  # frame → initial phase (radians)

    def __post_init__(self) -> None:
        for f in self.frames:
            assert f in self.port_of, f"frame {f} has no port mapping"
            assert f in self.init_freq, f"frame {f} has no init_freq"
            assert f in self.init_phase, f"frame {f} has no init_phase"
            assert self.port_of[f] in self.ports, (
                f"frame {f} maps to unknown port {self.port_of[f]}"
            )


# ---------------------------------------------------------------------------
# FrameState — mutable execution state
# ---------------------------------------------------------------------------

@dataclass
class FrameState:
    time: dict[str, int]                        # frame → real elapsed time including port waits (dt)
    phase: dict[str, float]                     # frame → accumulated phase (rad)
    port_time: dict[str, int]                   # port → latest time port becomes free (dt)
    cbit: dict[str, int | None]                 # cbit → value (None = not yet available)
    cbit_ready: dict[str, int]                  # cbit → earliest usable time
    occupancy: dict[str, list[tuple[int, int]]] # port → list of (start, end) intervals

    @staticmethod
    def initial(config: Config) -> FrameState:
        """Create σ₀ from a Config."""
        return FrameState(
            time={f: 0 for f in config.frames},
            phase={f: config.init_phase[f] for f in config.frames},
            port_time={p: 0 for p in config.ports},
            cbit={},
            cbit_ready={},
            occupancy={p: [] for p in config.ports},
        )

    def copy(self) -> FrameState:
        """Deep copy for snapshot / comparison."""
        return deepcopy(self)
