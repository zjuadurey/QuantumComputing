"""Port Exclusivity checker.

Property (formal_definitions_v0.md §2.1):
    PortExcl(σ) ≡ ∀p ∈ ports,
                   ∀(s₁,e₁),(s₂,e₂) ∈ σ.occupancy(p):
                     (s₁,e₁) ≠ (s₂,e₂) ⟹ e₁ ≤ s₂ ∨ e₂ ≤ s₁

This checker operates on the FrameState produced by the oracle.
It does NOT call ref_semantics — it only inspects the occupancy map.
"""

from __future__ import annotations

from pulse_ir.ir import FrameState


def check_port_exclusivity(state: FrameState) -> tuple[bool, list[str]]:
    """Check that no two intervals on the same port overlap.

    Returns (ok, errors) where errors is a list of human-readable messages.
    """
    errors: list[str] = []

    for port, intervals in state.occupancy.items():
        # Sort by start time for efficient pairwise check
        sorted_ivs = sorted(intervals)
        for i in range(len(sorted_ivs) - 1):
            s1, e1 = sorted_ivs[i]
            s2, e2 = sorted_ivs[i + 1]
            if e1 > s2:
                errors.append(
                    f"port {port}: interval [{s1},{e1}) overlaps [{s2},{e2})"
                )

    return (len(errors) == 0, errors)
