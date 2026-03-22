# Review Spec — v0.4 FullContract: Port-Aware Semantics + WF Precheck

Attach this file to the current thread before running `/review`.

## 1. Basic Info

- Title: v0.4 Port-aware semantics, well-formedness precheck, unified verification entry
- Date: 2026-03-23
- Branch / PR: main (on top of v0.3 tag)
- Owner: adurey
- Design spec: `openqasm-rtir/docs/v04_fullcontract_spec.md`

## 2. Review Goal

Redesign the formal model from per-frame-only clocks to port-aware resource-constrained semantics. Add source-level well-formedness precheck and a unified `verify_lowering()` entry point. This establishes the "FullContract" pipeline: WF precheck → oracle → lowering → reconstruct → 3 property checks.

Key semantic changes:
1. `Play/Acquire` start at `max(time[f], port_time[p])` — shared port auto-serializes
2. Phase evolves during implicit port stall: `phase += 2π × freq × total_advance`
3. `IfBit` does NOT auto-wait for cbit_ready — ill-formed programs rejected by WF precheck
4. New 5th buggy variant: `lower_buggy_ignore_shared_port` (ignores port_time)

## 3. In Scope

- `pulse_ir/ir.py` — FrameState adds `port_time: dict[str, int]`
- `pulse_ir/ref_semantics.py` — port-aware step rules with stall phase evolution
- `pulse_checks/wellformedness.py` — **NEW**: WF(P,C) source precheck
- `pulse_checks/feedback_causality.py` — source mode now uses port-aware body start; compiled mode unchanged
- `pulse_checks/frame_consistency.py` — port-aware source-vs-compiled correspondence
- `pulse_lowering/lower_to_schedule.py` — port-aware lowering
- `pulse_lowering/reconstruct.py` — reconstructs port_time from events
- `pulse_lowering/verify.py` — **NEW**: `verify_lowering()` + `VerificationReport`
- `pulse_lowering/buggy_variants.py` — +`lower_buggy_ignore_shared_port`
- `pulse_examples/correct_shared_port.py` — **NEW**: shared-port correct example
- `pulse_examples/violation_port_conflict.py` — repurposed as lowering bug demo
- `pulse_examples/violation_causality.py` — repurposed as ill-formed (WF rejects)
- `tests/test_pulse.py` — 24 tests (new: WF, shared-port feedback stall, verify_lowering)
- `tests/test_lowering_pulse.py` — 14 tests (new: ignore_shared_port)
- `docs/v04_guide.md` — **NEW**: usage documentation

## 4. Out of Scope

- Gate-level code — unchanged (10 tests still passing)
- `pulse_checks/port_exclusivity.py` — logic unchanged
- `pulse_lowering/schedule.py` — PulseEvent dataclass unchanged
- Paper sections — not updated in this PR
- Partial validators / ablations — future work

## 5. Files / Areas To Review Carefully

- `pulse_ir/ref_semantics.py` — verify port-aware step: `start = max(time[f], port_time[p])`, phase includes stall
- `pulse_checks/wellformedness.py` — verify: (1) cbit defined before use, (2) t_use >= cbit_ready, (3) nested deps all checked, (4) port-aware time tracking, (5) does NOT import ref_semantics
- `pulse_checks/feedback_causality.py` — verify source mode uses the same port-aware body-start notion as WF/compiled semantics
- `pulse_checks/frame_consistency.py` — verify `_compute_expected` is port-aware (matches oracle logic but independent code)
- `pulse_lowering/lower_to_schedule.py` — verify port-aware lowering matches oracle semantics
- `pulse_lowering/verify.py` — verify pipeline order: WF → oracle → lower → reconstruct → 3 checks
- `pulse_lowering/buggy_variants.py` — verify `lower_buggy_ignore_shared_port` deliberately uses only frame-local time

## 6. Expected Behavior After This Change

- `verify_lowering(prog, cfg)` returns `VerificationReport` with `overall_ok=True` for all WF correct programs
- `verify_lowering(ill_formed_prog, cfg)` returns `well_formed=False`, stops early
- `verify_lowering(prog, cfg, lower=lower_buggy_ignore_shared_port)` returns `port_exclusive=False`
- Shared-port program: oracle serializes → `PortExcl` passes; buggy lowering overlaps → `PortExcl` fails
- All 48 tests pass (10 gate + 24 pulse + 14 lowering)

## 7. Non-Negotiable Acceptance Criteria

- `ref_semantics.py` step rules: `start = max(time[f], port_time[p])` for Play/Acquire
- Phase advance = `2π × freq × (end - old_time[f])` (covers stall + operation)
- `wellformedness.py` does NOT import `ref_semantics`
- `frame_consistency._compute_expected` does NOT import `ref_semantics`
- `lower_to_schedule` does NOT import `ref_semantics`
- WF checks: (1) cbit defined by prior Acquire, (2) t_use >= cbit_ready at encounter, (3) nested IfBit all ancestors checked
- WF uses port-aware time tracking (same recurrence as oracle, separate code)
- `verify_lowering` stops early if WF fails (does not run oracle/lowering)
- For WF programs, oracle satisfies PortExcl (port_time serializes)
- `VerificationReport.overall_ok = well_formed & port_exclusive & feedback_causal & frame_consistent`

## 8. Semantic Invariants

All v0.3 invariants still hold, plus:
- `FrameState.port_time[p]` = latest time port p becomes free
- For Play/Acquire: `start = max(time[f], port_time[p])`, `end = start + duration`
- `time[f] = end` after Play/Acquire (includes stall)
- `phase[f]` advances by `2π × freq × (end - old_time[f])` (includes stall)
- Delay does NOT affect port_time (silence, no port activity)
- ShiftPhase does NOT affect time or port_time (zero duration)
- WF precheck uses same time/port_time recurrence as oracle (independent code path)

## 9. Known Risks

- `lower_buggy_reorder_ports` now fails BOTH PortExcl and FrameConsist (not specific to one checker). This is correct: flattening to t=0 violates both port serialization and time correspondence. The test documents this explicitly.
- `lower_buggy_ignore_shared_port` also fails both PortExcl and FrameConsist. Same reason: missing port stall means both wrong timing and overlapping intervals.
- WF precheck tracks port_time independently — if the recurrence drifts from oracle, WF could accept programs the oracle would handle differently. Mitigated by test_lowering_matches_oracle.

## 10. Tests Expected

- Step unit tests: 5 (existing 4 + shared_port_serialization)
- Correct examples: 4 (existing 3 + shared_port)
- WF precheck: 5 (correct passes, causality rejected, undefined cbit rejected, shared port passes, port stall can make IfBit legal)
- Violation examples: 3 (causality WF, phase corruption, time drift)
- Edge cases: 3 (unchanged)
- verify_lowering: 4 (correct pipeline, illformed rejected, shared port, shared-port feedback)
- Lowering correct: 3 (unchanged)
- Lowering buggy: 5 (existing 4 + ignore_shared_port)
- Schedule structure: 6 (unchanged)
- Total: 38 pulse + 10 gate = 48

## 11. Reviewer Focus

- **Port-aware correctness**: Do oracle, lowering, WF checker, and FrameConsist checker all use the same `max(time[f], port_time[p])` recurrence? Are they independent code paths?
- **WF completeness**: Does WF catch all three cases (undefined cbit, early use, nested ancestor not ready)?
- **verify_lowering pipeline**: Does it stop early on WF failure? Does it correctly wire all checkers?
- **IgnoreSharedPort bug**: Is the buggy variant a faithful simulation of "compiler ignores port serialization"?
- **No regressions**: Do all 10 gate-level tests still pass?

## 12. Done Means

- [ ] Port-aware semantics in oracle, lowering, WF, and FrameConsist (4 independent paths)
- [ ] WF rejects ill-formed programs, passes well-formed ones
- [ ] verify_lowering() returns correct VerificationReport
- [ ] 5 buggy variants each caught by their targeted checker(s)
- [ ] 48 tests all passing
- [ ] No ref_semantics imports in checkers or lowering
- [ ] docs/v04_guide.md present and accurate
