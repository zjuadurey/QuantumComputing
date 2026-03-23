# v0.4 FullContract Specification

This document is the current design source of truth for v0.4.
When code, paper text, older notes, or chat history disagree, this file
takes precedence for the semantics, API, terminology, and acceptance
criteria of FullContract.

Minimal context-recovery order across machines:
1. Read this file first.
2. Then read `docs/v04_guide.md` for the runnable implementation surface.
3. Read `docs/research_log.md` only if you need historical decisions.
4. Read `PROJECT_INTENT.md` only if you need the broader research motivation.

---

## 1. Frozen Terminology

| Term | Definition | Do NOT call it |
|------|-----------|----------------|
| Source-level core calculus | The 5-construct language: Play, Acquire, ShiftPhase, Delay, IfBit | DSL, QASM program (core calculus or core IR are both acceptable) |
| Source program | A `list[PulseStmt]` written in the core calculus | input program, QASM program |
| Configuration | Static record: frames, ports, port_of, freq_0, phase_0 | config, hardware spec |
| FrameState | Mutable state: time, phase, port_time, cbit, cbit_ready, occupancy | execution state, sigma |
| Reference semantics (oracle) | Sequential fold of step() over source program → FrameState | interpreter, simulator |
| Well-formedness (WF) | Source-level precheck: rejects ill-formed programs | validation, type check |
| Lowered schedule | Explicit list[PulseEvent] with start/end/port/phase/conditional_on | compiled output, IR |
| Lowering | Function: source program × config → lowered schedule | compiler, transpiler |
| Reconstruct | Function: lowered schedule × config → FrameState | bridge, adapter |
| FullContract | Unified lowering contract verification: source well-formedness, hardware-constrained source semantics, schedule-level contract checking | framework, tool, checker suite |
| Partial validator | Simplified checker covering a subset of properties | baseline, competing method |
| Ablated variant | FullContract with one component removed | degraded version |
| Coverage-driven benchmark suite | Pool B: curated programs organized by semantic dimensions | test suite, examples |
| Supplementary externally sourced kernels | Pool A: programs from public benchmarks, sliced to core calculus | external dataset |

---

## 2. FullContract API

### Input

```python
program: list[PulseStmt]   # source program in core calculus
config: Config              # static hardware configuration
lower: Callable             # lowering function (default: lower_to_schedule)
```

### Output

```python
@dataclass
class VerificationReport:
    well_formed: bool                # WF(P, C) precheck result
    wf_errors: list[str]             # WF violation details (if any)
    port_exclusive: bool             # PortExcl on lowered schedule
    feedback_causal: bool            # FeedbackCausal on lowered schedule (event-level)
    frame_consistent: bool           # FrameConsist: compiled state vs source semantics
    overall_ok: bool                 # well_formed & port_exclusive & feedback_causal & frame_consistent
    oracle_state: FrameState         # reference semantics output
    compiled_state: FrameState       # reconstructed from lowered schedule
    events: list[PulseEvent]         # the lowered schedule itself
    errors: list[str]                # all violation details
```

### Pipeline

```
source program + config
    │
    ├─→ WF precheck: check_wellformedness(program, config)
    │     └─ FAIL → report(well_formed=False), STOP
    │
    ├─→ Oracle: ref_semantics.run(program, config) → oracle_state
    │
    ├─→ Lowering: lower(program, config) → events
    │     └─→ Reconstruct: reconstruct_state(events, config) → compiled_state
    │
    └─→ Checkers (on compiled output):
          ├─ check_port_exclusivity(compiled_state) → port_exclusive
          ├─ check_schedule_causality(events) → feedback_causal
          └─ check_frame_consistency(compiled_state, program, config) → frame_consistent
```

---

## 3. Semantic Definitions (frozen)

### FrameState (v0.4)

```
FrameState = {
    time      : FrameId → ℕ          # real elapsed time (includes port wait)
    phase     : FrameId → ℝ          # accumulated phase (includes wait evolution)
    port_time : PortId → ℕ           # latest time port becomes free
    cbit      : Cbit → {⊥, 0, 1}
    cbit_ready: Cbit → ℕ
    occupancy : PortId → list[(ℕ, ℕ)]
}
```

### Step rules (v0.4)

**Play(f, w):**
```
start = max(σ.time[f], σ.port_time[p])
end = start + dur(w)
stall = start - σ.time[f]

σ'.time[f]      = end
σ'.phase[f]     = σ.phase[f] + 2π × freq_0(f) × (stall + dur(w))
σ'.port_time[p] = end
σ'.occupancy[p] = σ.occupancy[p] ∪ {(start, end)}
```

**Acquire(f, d, c):** Same as Play for time/phase/port_time/occupancy, plus:
```
σ'.cbit_ready[c] = end
```

**ShiftPhase(f, θ):**
```
σ'.phase[f] = σ.phase[f] + θ
# zero duration, no time/port change
```

**Delay(d, f):**
```
σ'.time[f]  = σ.time[f] + d
σ'.phase[f] = σ.phase[f] + 2π × freq_0(f) × d
# no port change (delay is silence)
```

**IfBit(c, body):**
```
# always-taken, NO auto-wait for cbit_ready
σ' = step(σ, body)
```

### Well-Formedness

```
WF(P, C) ≡ for each IfBit(c, body) encountered during sequential walk:
              1. c must have been defined by a prior Acquire (∃ Acquire(_, _, c) before this point)
              2. t_use = REAL start time of the guarded body event at point of encounter
                 (Play/Acquire use max(time[f_body], port_time[p_body]); Delay/ShiftPhase use time[f_body])
              3. t_use ≥ cbit_ready[c]
              4. for nested IfBit: ALL ancestor dependencies must also be satisfied
                 (if IfBit(c0, IfBit(c1, body)), then t_use ≥ cbit_ready[c0] AND t_use ≥ cbit_ready[c1])
```

WF is checked by independently walking the AST and tracking frame times
(same recurrence as oracle but separate code path).

**Oracle guarantees (for WF programs):**

Source-level guarantees (on FrameState):
- PortExcl: port_time serializes shared-port access → no overlap in occupancy
- FrameConsist: oracle time/phase obey source semantics by construction (this is the trivial direction; the non-trivial check is compiled-vs-source) (phase = init + shifts + 2π×freq×time)

Source-level well-formedness guarantee:
- WF ensures the real body start time satisfies t_use ≥ cbit_ready at every IfBit encounter point

Schedule-level guarantee (separate verification on lowered schedule):
- FeedbackCausal_sched: must be independently verified on the lowered schedule's events
  (oracle produces FrameState, not a schedule with conditional_on tags)

### Property Definitions

**PortExcl(σ):** ∀p, all intervals in σ.occupancy[p] are non-overlapping.

**FeedbackCausal_sched(S):** ∀ event e ∈ S with conditional_on ⊇ {c₁..cₖ},
  e.start ≥ cbit_ready(cᵢ) for all cᵢ.
  (cbit_ready derived from Acquire events in S)

**FrameConsist(σ_compiled, P, C):** ∀f:
  let (expected_time, expected_phase) = independently compute from source program P using C
  σ_compiled.time[f] = expected_time(f)  AND
  σ_compiled.phase[f] = expected_phase(f)

  This is a **source-vs-compiled correspondence**: the checker independently computes
  expected time and phase from the source program, then compares against the compiled
  state. It does NOT use compiled.time to validate compiled.phase (that would be
  self-referential).

---

## 4. Fault Class → Method Capability Truth Table

| Fault class | Description | FullContract | B1-PortOnly | B2-EndState | B3-FrameLocal | A1-noWF | A2-noEventCausal |
|-------------|-------------|:---:|:---:|:---:|:---:|:---:|:---:|
| DropShiftPhase | Lowering drops ShiftPhase | ✓ FC | ✗ | ✓ | ✓ | ✓ FC | ✓ FC |
| ExtraDelay | Lowering inserts extra delay | ✓ FC | ✗ | ✓ | ✓ | ✓ FC | ✓ FC |
| EarlyFeedback | Conditional event before cbit_ready | ✓ Causal | ✗ | ✗ | ✗ | ✓ Causal | ✗ |
| IgnoreSharedPort | Shared port not serialized | ✓ Port | ✓ | ✗ | ✗ | ✓ Port | ✓ Port |
| NestedDepLoss | Nested IfBit loses outer cbit | ✓ Causal | ✗ | ✗ | ✗ | ✓ Causal | ✗ |

Key:
- ✓ = detected, with which checker
- ✗ = missed
- FC = FrameConsist, Causal = FeedbackCausal, Port = PortExcl

**B2-EndState definition:** Compares only final per-frame (time, phase) against
source-derived expectations. Ignores event-level schedule structure, conditional
dependencies, and port occupancy.

**B3-FrameLocal definition:** Only does frame-local time/phase correspondence.
Does NOT do event-level causality. Does NOT model shared-port wait.
Therefore misses: EarlyFeedback, NestedDepLoss (no causality), IgnoreSharedPort (no port modeling).

**Key observations from the table:**
- B1-PortOnly only catches IgnoreSharedPort (1/5)
- B2-EndState only catches phase/time drift (2/5), misses event-level bugs
- B3-FrameLocal catches phase/time drift but misses causality and port bugs (2/5)
- A2-noEventCausal misses all feedback bugs (3/5)
- Only FullContract catches all 5/5

---

## 5. Files to Create / Modify

### New files

```
pulse_checks/wellformedness.py     ← WF(P, C) precheck
pulse_lowering/verify.py           ← verify_lowering() unified entry + VerificationReport
```

### Modified files

```
pulse_ir/ir.py                     ← FrameState adds port_time
pulse_ir/ref_semantics.py          ← step rules: start = max(time[f], port_time[p])
pulse_lowering/lower_to_schedule.py ← same port_time logic
pulse_checks/wellformedness.py      ← source-side `check_wellformedness(P, C)`
pulse_checks/feedback_causality.py  ← schedule-side `check_schedule_causality(events)`
pulse_checks/port_exclusivity.py    ← unchanged (logic same, but oracle now guarantees it for WF programs)
pulse_checks/frame_consistency.py   ← unchanged (formula same, time[f] semantics updated)
pulse_examples/                     ← update violation_port_conflict → lowering bug, add shared-port correct example
tests/test_pulse.py                 ← update expected values for port-aware semantics
tests/test_lowering_pulse.py        ← update expected values, add WF tests
```

### Unchanged files

```
pulse_lowering/schedule.py          ← PulseEvent dataclass (no change)
pulse_lowering/reconstruct.py       ← reads from events (no change)
pulse_lowering/buggy_variants.py    ← add IgnoreSharedPort variant
```

---

## 6. Acceptance Criteria

- [ ] WF precheck rejects ill-formed programs (IfBit before cbit_ready, undefined cbit, nested dep missing)
- [ ] WF precheck passes all well-formed programs
- [ ] Oracle produces PortExcl-satisfying state for all WF programs (port_time serializes shared-port access)
- [ ] WF implies source-side feedback legality at each IfBit encounter point
- [ ] Oracle matches source semantics by construction for time/phase (FrameConsist is trivial on oracle output)
- [ ] Schedule-level FeedbackCausal_sched is independently checked on lowered events
- [ ] Correct lowering matches oracle for shared-port programs (port serialization)
- [ ] verify_lowering() returns clean report for correct lowering of WF programs
- [ ] Each buggy variant caught by its targeted checker (5 fault classes)
- [ ] All partial validators and ablations can be derived from FullContract code
- [ ] No regression in gate-level tests (10 tests)
- [ ] lower_to_schedule does NOT import ref_semantics
- [ ] wellformedness does NOT import ref_semantics
