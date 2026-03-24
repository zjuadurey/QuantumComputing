# v0.5 Supporting-Evidence Specification

This document records the v0.5 extension on top of v0.4.
It is not a replacement for `docs/v04_fullcontract_spec.md`. Instead, it
defines the role of an external, community-recognized corroboration layer.

## 1. Purpose

v0.4 established the main claim:

- FullContract is the **primary verifier** of source-to-schedule lowering correctness.

v0.5 adds a secondary evidence layer:

- A trusted external component is used as **supporting evidence**, not as the
  definition of the semantics and not as the main verifier.

The goal is to avoid a closed self-justifying story of:

```
we define the semantics
→ we implement the checker
→ we seed the faults
→ we declare success
```

v0.5 instead asks a narrower question:

> If FullContract reports a lowering violation, can a trusted external
> component independently observe a meaningful deviation?

## 2. Frozen Positioning

### Primary evidence (unchanged)

- source semantics / oracle
- lowering contract
- independent checkers
- seeded lowering faults

### Secondary evidence (new in v0.5)

- Qiskit Dynamics as a trusted pulse-level corroboration layer
- optional backend-facing checks in future work

### Important non-goals

v0.5 does **not** claim:

- that Qiskit Dynamics defines the source semantics
- that Qiskit Dynamics replaces FullContract
- that simulator agreement proves physics-level correctness

### Role separation (frozen)

v0.5 uses three distinct roles and they must not be conflated:

1. **Core idea**
   - our pulse-level lowering story
   - our contract decomposition
   - our checker pipeline

2. **Primary verifier**
   - `FullContract`
   - defines what counts as lowering correctness in this project

3. **Primary external experimental witness**
   - `Qiskit Dynamics`
   - supplies community-recognized, independent evidence in the main experiments

This means Qiskit Dynamics may appear as the main empirical platform in the
evaluation section while still not being the semantic oracle or the main verifier.

The role of Qiskit Dynamics is strictly:

- externally recognized
- independent from our checker implementation
- able to show externally visible deviation for selected fault families
- positioned as the main external witness in the empirical story

## 3. Fault-family matrix

| Fault family | Main semantic obligation | Primary checker | Best external anchor | v0.5 status |
|---|---|---|---|---|
| `drop_phase` | frame / phase semantics | `FrameConsist` | `Qiskit Dynamics` | Implemented |
| `early_feedback` | feedback causality | `FeedbackCausal_sched` | timing-sensitive `Qiskit Dynamics` witness with static drift | Implemented |
| `ignore_shared_port` | shared-port serialization | `PortExcl` + `FrameConsist` | schedule-level `Qiskit Dynamics` witness on shared-port programs | Implemented |
| `reorder_ports` | sequencing + shared resource | `PortExcl` + `FrameConsist` | schedule-level `Qiskit Dynamics` witness on shared-port programs | Implemented |
| `extra_delay` | time preservation | `FrameConsist` | weak external support only | Deliberately de-prioritized |

## 4. Implemented v0.5 surface

### New module

`pulse_external/qiskit_dynamics.py`

Provides:

- `simulate_single_frame_schedule(...)`
- `compare_single_frame_lowerings(...)`
- `DynamicsCorroborationResult`

Current scope:

- single-frame and multi-frame play schedules projected to one effective qubit
- phase-sensitive witnesses (`drop_phase`)
- shared-port serialization / reordering witnesses (`ignore_shared_port`, `reorder_ports`)
- timing-sensitive witness with static drift (`early_feedback`)

Design intent:

- convert `PulseEvent` schedules into orthogonal `ux` / `uy` drive channels
- use event `phase_before` snapshots to build a pulse-level witness
- compare correct vs faulty lowering with final-state fidelity

### New tests

`tests/test_v05_external_corroboration.py`

Coverage:

- regression test for order-insensitive reconstruction
- stronger `early_feedback` seeded fault across delay blocks
- Qiskit Dynamics identity sanity check
- Qiskit Dynamics witness for `drop_phase`

## 5. Interpreting the evidence

The external witness is meant to support the following sentence in the paper:

> Our pulse-level lowering contract is the core idea and FullContract is the
> primary verifier. Qiskit Dynamics is used as the primary external experimental
> witness, showing that selected contract-detected schedule violations also
> induce externally meaningful deviation.

This sentence is intentionally weaker than:

> The simulator proves the contract is correct.

The first is the intended v0.5 claim. The second is explicitly out of scope.

## 6. Next-step experiment plan

Minimal credible evaluation plan after v0.5:

1. `drop_phase`
   - show fidelity deviation between correct vs faulty schedules

2. `ignore_shared_port`
   - show schedule-level external deviation on a shared-port, phase-diverse program

3. `reorder_ports`
   - show external deviation under aggressive flattening on the same witness program

4. `early_feedback`
   - show timing-sensitive deviation by simulating the affected frame under static drift

5. keep `extra_delay` as a contract-level example unless a compelling external
   witness is found
