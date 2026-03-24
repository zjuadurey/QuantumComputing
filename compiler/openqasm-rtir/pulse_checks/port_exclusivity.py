"""Port Exclusivity checker.

Property (formal_definitions_v0.md §2.1):
    PortExcl(σ) ≡ ∀p ∈ ports,
                   ∀(s₁,e₁),(s₂,e₂) ∈ σ.occupancy(p):
                     (s₁,e₁) ≠ (s₂,e₂) ⟹ e₁ ≤ s₂ ∨ e₂ ≤ s₁

This checker operates on the FrameState produced by the oracle.
It does NOT call ref_semantics — it only inspects the occupancy map.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pulse_ir.ir import FrameState


@dataclass(frozen=True)
class PortOverlap:
    port: str
    first_interval: tuple[int, int]
    second_interval: tuple[int, int]
    overlap_dt: int


@dataclass(frozen=True)
class PortExclusivityDiagnostics:
    overlaps: list[PortOverlap] = field(default_factory=list)
    num_conflicts: int = 0
    max_overlap_dt: int = 0
    total_overlap_dt: int = 0


def check_port_exclusivity(state: FrameState) -> tuple[bool, list[str]]:
    """Check that no two intervals on the same port overlap.

    Returns (ok, errors) where errors is a list of human-readable messages.
    """
    diagnostics = diagnose_port_exclusivity(state)
    errors = format_port_exclusivity_errors(diagnostics)
    return (len(errors) == 0, errors)


def diagnose_port_exclusivity(state: FrameState) -> PortExclusivityDiagnostics:
    """Return structured overlap diagnostics for each port."""
    overlaps: list[PortOverlap] = []
    max_overlap_dt = 0
    total_overlap_dt = 0

    for port, intervals in state.occupancy.items():
        sorted_ivs = sorted(intervals)
        for i in range(len(sorted_ivs) - 1):
            s1, e1 = sorted_ivs[i]
            s2, e2 = sorted_ivs[i + 1]
            if e1 > s2:
                overlap_dt = min(e1, e2) - s2
                overlaps.append(PortOverlap(
                    port=port,
                    first_interval=(s1, e1),
                    second_interval=(s2, e2),
                    overlap_dt=overlap_dt,
                ))
                max_overlap_dt = max(max_overlap_dt, overlap_dt)
                total_overlap_dt += overlap_dt

    return PortExclusivityDiagnostics(
        overlaps=overlaps,
        num_conflicts=len(overlaps),
        max_overlap_dt=max_overlap_dt,
        total_overlap_dt=total_overlap_dt,
    )


def format_port_exclusivity_errors(
    diagnostics: PortExclusivityDiagnostics,
) -> list[str]:
    """Render human-readable overlap errors."""
    errors: list[str] = []
    for overlap in diagnostics.overlaps:
        (s1, e1) = overlap.first_interval
        (s2, e2) = overlap.second_interval
        errors.append(
            f"port {overlap.port}: interval [{s1},{e1}) overlaps "
            f"[{s2},{e2}) (overlap_dt={overlap.overlap_dt})"
        )
    return errors
