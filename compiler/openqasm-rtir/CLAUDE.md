# openqasm-rtir — Project Context

> **If you are a new Claude Code session**: read this file first. It contains everything
> you need to understand the project and start coding. For detailed formal specs,
> read `docs/formal_definitions_v0.md`. For OpenPulse reference, read
> `docs/openpulse_semantics_summary.md`.

## Project identity

**One-sentence essence**: We are building a correctness model for pulse-level quantum
programs — formalizing "what does correct mean" and "how to check it automatically"
for OpenPulse-style control sequences (play, acquire, shift_phase, delay on ports and frames).

**What this is NOT**: not a full quantum compiler, not a waveform-to-unitary verifier,
not a vendor API tool, not a physics simulator.

See `PROJECT_INTENT.md` for full research motivation.

## Environment & preferences

- Python 3.12 via conda env `qiskit_qasm_py312`
- Run ALL commands with: `conda run -n qiskit_qasm_py312 --no-capture-output <cmd>`
- Dependencies: qiskit 2.3.1, qiskit_qasm3_import, pytest
- Communication: 中文交流，代码和注释用英文
- Style: MVP first, no over-engineering, iterate incrementally

## Research positioning

- Core question: how to formally define, reliably lower, and verify correctness of quantum programs with timing, pulse, and measurement-feedback semantics
- Structural gap: existing verified quantum compilation (VOQC/SQIR/CertiQ) covers gate-level equivalence but NOT timing/resource/feedback semantics
- Target venues: POPL / PLDI / ICSE / ASPLOS (CCF-A)
- Three contributions: (1) formal semantics, (2) verifiable compilation chain, (3) regression + safety property verification

## Paper strategy (decided 2026-03-19)

- **Gate-level = methodology warmup** (≤3 pages): no_conflict + causality on toy gate IR
- **Pulse-level = core contribution**: formal definitions + reference semantics + checker
- Anchored to OpenQASM/OpenPulse **spec**, not vendor APIs (IBM deprecated qiskit.pulse)
- Pulse core subset: `{Play, Acquire, ShiftPhase, Delay}` × `{port, frame}`
- Three pulse-level properties: **port exclusivity**, **feedback causality**, **frame consistency**
- NOT doing: waveform-to-unitary, defcal binding, strong timed bisimulation, SetFreq

## Key decisions (consolidated)

1. **Oracle must be independent from checker** — if checker uses Z3, oracle must use a different path (e.g., exhaustive enumeration, hand proof, reference interpreter)
2. **Giallar/VOQC go in related work capability table, NOT in evaluation section**
3. **Scope is frozen** — don't expand beyond {Play, Acquire, ShiftPhase, Delay} for v0.2
4. **Definitions before code** — "定义是第一产物，checker 只是定义的一个实现"
5. **Frame consistency is a correspondence property** — checks source semantics vs compiled output, not just a standalone predicate

## Current status

| Item | Status |
|------|--------|
| v0.1 gate-level MVP | ✅ Done — parse → regex lowering → IR → timeline → checks, 10 tests passing |
| Formal definitions (docs/formal_definitions_v0.md) | ✅ Done — abstract syntax, Config/State, 4 step rules, 3 properties |
| OpenPulse spec summary (docs/openpulse_semantics_summary.md) | ✅ Done |
| Research log (docs/research_log.md) | ✅ Done — records all decisions from 2026-03-13 to 2026-03-19 |
| v0.2 pulse-level prototype | ✅ Done — ir.py + ref_semantics.py + 3 checkers + 6 examples + 14 tests |
| v0.3 pulse lowering + correspondence | ✅ Done — lower_to_schedule + reconstruct + 4 buggy variants + 13 tests (3 rounds Codex review) |
| v0.4 FullContract alignment | ✅ Done — port-aware timing + wellformedness + verify_lowering + schedule-level feedback checker, 48 tests passing |

## Historical v0.2 implementation plan (completed; kept for context)

Based on the formal definitions in `docs/formal_definitions_v0.md`, implement the following:

### Files to create

```
pulse_ir/
├── __init__.py
├── ir.py                    ← Config, FrameState, PulseStmt (union type), PulseEvent
└── ref_semantics.py         ← step() function + run() sequential interpreter (= oracle)

pulse_checks/
├── __init__.py
├── port_exclusivity.py      ← PortExcl: no overlapping intervals on same port
├── feedback_causality.py    ← v0.4: FeedbackCausal_sched on lowered events only
├── wellformedness.py        ← v0.4: source-side legality precheck
└── frame_consistency.py     ← FrameConsist: phase = init + Σshifts + 2π·freq·elapsed

pulse_examples/
├── __init__.py
├── correct_single_play.py   ← one frame, one play — should PASS all checks
├── correct_measure_feedback.py ← acquire then conditional play — should PASS
├── correct_multi_frame.py   ← two frames on different ports — should PASS
├── violation_port_conflict.py  ← two frames share port, overlapping play — should FAIL port_excl
├── violation_causality.py   ← use cbit before acquire finishes — should FAIL causality
└── violation_phase.py       ← wrong phase accumulation — should FAIL frame_consist

tests/
└── test_pulse.py            ← pytest: correct examples PASS, violations FAIL expected checks
```

### Key specs (from formal_definitions_v0.md)

**PulseStmt** (4 variants):
- `Play(frame, waveform)` — advances frame.time by dur(waveform), occupies port
- `Acquire(frame, duration, cbit)` — advances frame.time, occupies port, sets cbit_ready
- `ShiftPhase(frame, angle)` — adds angle to frame.phase, zero duration, no port activity
- `Delay(duration, frame)` — advances frame.time, no port activity

**State** tracks: `time[frame]`, `phase[frame]`, `cbit[c]`, `cbit_ready[c]`, `occupancy[port]`

**Three properties**:
1. `PortExcl(σ)`: ∀ same-port interval pairs, no overlap
2. `FeedbackCausal(σ, P)`: conditional start ≥ cbit_ready
3. `FrameConsist(σ, P)`: phase = init_phase + Σ(shifts) + 2π × freq × elapsed_time

### How to run

```bash
conda run -n qiskit_qasm_py312 --no-capture-output python -m pytest tests/ -v
conda run -n qiskit_qasm_py312 --no-capture-output python run_demo.py examples/simple_delay.qasm
```

## Existing v0.1 project structure (gate-level, keep as-is)

```
openqasm-rtir/
├── run_demo.py                 # CLI entry: validate → lower → timeline → checks
├── examples/*.qasm             # toy OpenQASM 3 test inputs
├── parser_bridge/import_qasm3.py  # qiskit bridge for syntax validation
├── rt_ir/ir.py                 # RTEvent dataclass
├── rt_ir/lowering.py           # regex-based toy lowering
├── interpreter/simulate_rt.py  # timeline table printer
├── checks/no_conflict.py       # resource overlap check
├── checks/causality.py         # depends_on causality check
└── tests/test_lowering.py      # 10 tests, all passing
```

## Baseline & evaluation

- **Evaluation**: benchmark × independent oracle × checker × Qiskit/pytket engineering comparison
- **Related work** (NOT evaluation): VOQC / Giallar / CertiQ capability table
- **Oracle independence**: oracle = reference semantics interpreter; checker = separate implementation
- Three evaluation categories: resource violations, feedback violations, timing/phase violations

## File map

| File | Contents |
|------|----------|
| `CLAUDE.md` | This file — complete project context for any Claude session |
| `PROJECT_INTENT.md` | Research motivation, structural gap, paper positioning |
| `docs/formal_definitions_v0.md` | **Formal spec**: abstract syntax, state, step rules, 3 properties |
| `docs/openpulse_semantics_summary.md` | OpenPulse spec reference: port/frame/waveform semantics |
| `docs/research_log.md` | Chronological decision log (v0.1 → baseline → pulse pivot) |

## Known gotchas

- `qiskit_qasm3_import.parse()` requires `if (c[0])` not `if (c[0] == 1)` for bit conditions
- IBM deprecated `qiskit.pulse` — anchor to OpenQASM/OpenPulse spec, not vendor API
- Delay does NOT occupy port (silence, no physical signal) — different from gate-level where delay occupies drive resource
