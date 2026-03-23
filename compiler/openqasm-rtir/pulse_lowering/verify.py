"""Unified lowering contract verification.

verify_lowering() is the single entry point that runs the full pipeline:
1. WF precheck on source program
2. Oracle (reference semantics) for ground truth
3. Lowering to explicit schedule
4. Reconstruct compiled state from schedule
5. Three property checks on compiled output

Returns a VerificationReport with all results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from pulse_ir.ir import Config, FrameState, PulseStmt
from pulse_ir.ref_semantics import run as oracle_run
from pulse_lowering.schedule import PulseEvent
from pulse_lowering.lower_to_schedule import lower_to_schedule
from pulse_lowering.reconstruct import reconstruct_state
from pulse_checks.wellformedness import check_wellformedness
from pulse_checks.port_exclusivity import check_port_exclusivity
from pulse_checks.feedback_causality import check_schedule_causality
from pulse_checks.frame_consistency import check_frame_consistency


@dataclass
class VerificationReport:
    well_formed: bool
    wf_errors: list[str] = field(default_factory=list)
    port_exclusive: bool = False
    feedback_causal: bool = False
    frame_consistent: bool = False
    overall_ok: bool = False
    oracle_state: FrameState | None = None
    compiled_state: FrameState | None = None
    events: list[PulseEvent] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def verify_lowering(
    program: list[PulseStmt],
    config: Config,
    lower: Callable[[list[PulseStmt], Config], list[PulseEvent]] = lower_to_schedule,
) -> VerificationReport:
    """Run the full verification pipeline.

    1. Check source well-formedness
    2. Run oracle for ground truth
    3. Lower to schedule
    4. Reconstruct compiled state
    5. Check all three properties on compiled output
    """
    # Step 1: WF precheck
    wf_ok, wf_errors = check_wellformedness(program, config)
    if not wf_ok:
        return VerificationReport(
            well_formed=False,
            wf_errors=wf_errors,
            errors=wf_errors,
        )

    # Step 2: Oracle
    oracle_state = oracle_run(program, config)

    # Step 3: Lower
    events = lower(program, config)

    # Step 4: Reconstruct
    compiled_state = reconstruct_state(events, config)

    # Step 5: Check properties on compiled output
    port_ok, port_errors = check_port_exclusivity(compiled_state)
    causal_ok, causal_errors = check_schedule_causality(events)
    fc_ok, fc_errors = check_frame_consistency(
        compiled_state, program, config,
    )

    all_errors = port_errors + causal_errors + fc_errors
    overall = port_ok and causal_ok and fc_ok

    return VerificationReport(
        well_formed=True,
        port_exclusive=port_ok,
        feedback_causal=causal_ok,
        frame_consistent=fc_ok,
        overall_ok=overall,
        oracle_state=oracle_state,
        compiled_state=compiled_state,
        events=events,
        errors=all_errors,
    )
