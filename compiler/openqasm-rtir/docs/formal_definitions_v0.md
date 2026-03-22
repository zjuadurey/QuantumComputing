# Formal Definitions (Draft v0)

---

## Part 1: Abstract Syntax, State, and Execution Semantics

### 1.1 Gate-level subset (warmup only)

```
GateStmt ::= Gate(g, q)           where g ∈ {H, X}, q ∈ Qubit
           | Delay(d, q)          where d ∈ ℕ (duration in dt)
           | Measure(q, c)        where c ∈ Cbit
           | If(c, GateStmt)
```

A gate-level program is `P_gate = list[GateStmt]`.

Gate-level IR event (already implemented in v0.1 as RTEvent):
```
GateEvent = (id: ℕ, kind: GateKind, start: ℕ, duration: ℕ, resource: Resource, deps: set[ℕ])
```

Gate-level lowering: `lower_gate : P_gate → list[GateEvent]`

This level only demonstrates no_conflict and causality. Not a paper contribution.

---

### 1.2 Pulse-level core subset

#### Syntax

```
PulseStmt ::= Play(f, w)          -- play waveform w on frame f
            | Acquire(f, d, c)    -- capture for duration d on frame f, write result to c
            | ShiftPhase(f, θ)    -- shift phase of frame f by θ (instantaneous)
            | Delay(d, f)         -- advance frame f's clock by d (no port activity)
```

Where:
- `f ∈ FrameId`
- `w ∈ Waveform` with `dur(w) ∈ ℕ`
- `d ∈ ℕ` (duration in dt)
- `c ∈ Cbit`
- `θ ∈ ℝ` (phase in radians)

A pulse-level program is `P_pulse = list[PulseStmt]`.

#### Configuration (static, fixed at initialization)

```
Config = {
    frames    : set[FrameId],
    ports     : set[PortId],
    port_of   : FrameId → PortId,          -- each frame binds to exactly one port
    init_freq : FrameId → ℝ,               -- initial frequency (constant; no SetFreq)
    init_phase: FrameId → ℝ,               -- initial phase
}
```

Key constraint: multiple frames may share a port (`port_of` is not injective).

#### State

```
State = {
    time     : FrameId → ℕ,                -- each frame's local clock
    phase    : FrameId → ℝ,                -- each frame's accumulated phase
    cbit     : Cbit → {⊥} ∪ {0, 1},       -- classical bit value (⊥ = not yet available)
    cbit_ready: Cbit → ℕ,                  -- earliest time cbit value is usable
    occupancy: PortId → list[(ℕ, ℕ)],      -- intervals during which port is active
}
```

Initial state `σ₀`:
```
    time(f)      = 0                        ∀f ∈ frames
    phase(f)     = init_phase(f)            ∀f ∈ frames
    cbit(c)      = ⊥                        ∀c
    cbit_ready(c)= 0                        ∀c
    occupancy(p) = []                       ∀p ∈ ports
```

#### Single-step execution: σ' = step(σ, s)

**Play(f, w)**:
```
    let t   = σ.time(f)
    let d   = dur(w)
    let p   = port_of(f)
    σ'.time(f)      = t + d
    σ'.phase(f)     = σ.phase(f) + 2π · init_freq(f) · d
    σ'.occupancy(p) = σ.occupancy(p) ∪ {(t, t + d)}
    -- all other state components unchanged
```

**Acquire(f, d, c)**:
```
    let t   = σ.time(f)
    let p   = port_of(f)
    σ'.time(f)       = t + d
    σ'.phase(f)      = σ.phase(f) + 2π · init_freq(f) · d
    σ'.occupancy(p)  = σ.occupancy(p) ∪ {(t, t + d)}
    σ'.cbit(c)       = ⊥  (resolved at hardware level; abstractly non-deterministic)
    σ'.cbit_ready(c) = t + d
```

**ShiftPhase(f, θ)**:
```
    σ'.phase(f) = σ.phase(f) + θ
    σ'.time(f)  = σ.time(f)         -- zero duration, no time advance
    -- no port occupancy change
```

**Delay(d, f)**:
```
    σ'.time(f)  = σ.time(f) + d
    σ'.phase(f) = σ.phase(f) + 2π · init_freq(f) · d
    -- no port occupancy change (delay is silence, no physical signal)
```

#### Multi-step execution

Given program P = [s₁, s₂, ..., sₙ]:
```
    σₙ = step(... step(step(σ₀, s₁), s₂) ..., sₙ)
```

This is the **reference semantics**: a straightforward sequential interpreter.
It serves as the independent oracle for correctness evaluation.

---

## Part 2: Three Pulse-level Correctness Properties

All three are predicates on the final state σₙ produced by the reference semantics.

### Property 1: Port Exclusivity

**Informal**: No two operations occupy the same port at the same time.

**Formal**:
```
PortExcl(σ) ≡ ∀p ∈ ports,
               ∀(s₁, e₁), (s₂, e₂) ∈ σ.occupancy(p):
                 (s₁, e₁) ≠ (s₂, e₂) ⟹ e₁ ≤ s₂ ∨ e₂ ≤ s₁
```

**What it catches**: Two frames sharing a port that attempt overlapping play/acquire.

**Important**: PortExcl is NOT an invariant of the reference semantics.
Because each frame maintains an independent local clock, a program operating on
two frames sharing a port (e.g., `Play(f1,w); Play(f2,w)` where `port_of(f1) = port_of(f2)`)
produces overlapping intervals on that port even under sequential execution.
PortExcl is a safety property to CHECK, not one guaranteed by construction.
It detects conflicts in both source programs and compiled schedules.

---

### Property 2: Feedback Causality

**Informal**: A conditional action depending on classical bit c must not begin
before the acquire that produces c has finished.

**Formal**:

For a program extended with conditional statements:
```
PulseStmt ::= ... | IfBit(c, PulseStmt)
```

```
FeedbackCausal(σ, P) ≡ ∀ IfBit(c, body) in P:
                          let t_use = σ.time(f_body)    -- time when body would execute
                          t_use ≥ σ.cbit_ready(c)
```

where `f_body` is the frame that `body` operates on.

**Important**: t_use is the frame time AT THE POINT where IfBit is encountered
(before the body executes), not the final state time. The source-level checker
independently tracks frame times by walking the AST.

At the schedule level, this is checked per-event: each event tagged with
`conditional_on ⊇ {c1, ..., ck}` must satisfy `start ≥ cbit_ready(ci)` for all ci.

Like PortExcl, FeedbackCausal is NOT guaranteed by the reference semantics.
A program that places IfBit before its corresponding Acquire will violate causality.

**What it catches**: A conditional gate that fires before its triggering measurement
has completed — i.e., using a result that doesn't exist yet.

**Refinement note**: In real hardware, there is additional classical processing latency
between acquire end and result availability. For v0, we model this as zero latency
(cbit_ready = acquire end time). Future versions may add a latency parameter.

---

### Property 3: Frame Consistency

**Informal**: The accumulated phase of each frame equals the sum of explicit shifts
plus the frequency-driven phase evolution over elapsed time.

**Formal**:
```
FrameConsist(σ, P) ≡ ∀f ∈ frames:
    σ.phase(f) = init_phase(f)
                 + Σ { θ | ShiftPhase(f, θ) ∈ P }
                 + 2π · init_freq(f) · σ.time(f)
```

**What it catches**: An implementation that:
- forgets to accumulate phase during a Delay
- applies ShiftPhase at the wrong point in time
- reorders operations in a way that changes the net phase
- miscalculates the elapsed time for a frame

**Why this matters physically**: Frame phase determines the carrier phase of every
subsequent pulse. A wrong phase = a wrong rotation axis = a wrong gate.
This is the formal basis for virtual Z gate correctness.

**Key subtlety**: If reference semantics is the oracle, then FrameConsist is trivially
satisfied by the reference semantics itself (it's true by construction of the step function).
The property becomes non-trivial when checking a **compiled/lowered output** against
the reference: does the lowering preserve frame consistency?

This means FrameConsist is actually a **correspondence property**:
```
FrameConsist_compiled(P, P') ≡ ∀f ∈ frames:
    phase_ref(f, P) = phase_compiled(f, P')
```

where `phase_ref` comes from reference semantics and `phase_compiled` comes from
the checker running on the lowered output.

---

## Summary: what each property tests

| Property | Type | Oracle guarantees? | Non-trivial when |
|----------|------|--------------------|------------------|
| Port exclusivity | Safety check | No | Multiple frames share a port, or reordering occurs |
| Feedback causality | Causality check | No | Conditional depends on measurement result |
| Frame consistency | Correspondence | Yes (by construction) | Lowering reorders or transforms operations |

**Property classification**:
- PortExcl and FeedbackCausal are safety properties that can be violated by source programs.
- FrameConsist is a correspondence property guaranteed by the oracle, only non-trivial for compiled output.

---

## Next: what to implement

1. `pulse_ir/ir.py` — PulseEvent dataclass + FrameState + Config
2. `pulse_ir/ref_semantics.py` — Sequential interpreter (reference semantics / oracle)
3. `pulse_checks/port_exclusivity.py` — Check PortExcl
4. `pulse_checks/feedback_causality.py` — Check FeedbackCausal
5. `pulse_checks/frame_consistency.py` — Check FrameConsist (correspondence version)
6. `pulse_examples/` — Small pulse programs with known violations
