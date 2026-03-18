# openqasm-rtir — Project Context

## What this is

Research prototype: OpenQASM 3 toy subset → real-time IR → timeline + correctness checks.
NOT a full quantum compiler. Focus is on timing/feedback/resource semantics formalization.

See `PROJECT_INTENT.md` for full research motivation.

## Research positioning

- Core question: how to formally define, reliably lower, and verify correctness of quantum programs with timing, pulse, and measurement-feedback semantics
- Structural gap: existing verified quantum compilation (VOQC/SQIR/CertiQ) covers gate-level equivalence but NOT timing/resource/feedback semantics
- Target venues: POPL / PLDI / ICSE / ASPLOS (CCF-A)
- Three contributions: (1) formal semantics, (2) verifiable compilation chain, (3) regression + safety property verification

## Paper strategy (decided 2026-03-19)

- **Gate-level = methodology warmup** (≤3 pages): no_conflict + causality on toy gate IR
- **Pulse-level = core contribution**: formal definitions + reference semantics + checker for OpenPulse core
- Anchored to OpenQASM/OpenPulse **spec**, not vendor APIs (IBM deprecated qiskit.pulse)
- Pulse core subset: `{Play, Acquire, ShiftPhase, Delay}` × `{port, frame}`
- Three pulse-level properties: port exclusivity, feedback causality, frame consistency
- NOT doing: waveform-to-unitary, defcal binding, strong timed bisimulation
- Execution order: formal definitions → reference semantics (independent oracle) → checker → benchmark → engineering comparison
- Oracle must be independent from checker (no shared Z3 encoding)
- Giallar/VOQC go in related work capability table, NOT in evaluation
- See `docs/openpulse_semantics_summary.md` for OpenPulse spec reference

## Current status: v0.1 MVP (2026-03-13)

Minimal closed loop working: parse → regex lowering → IR → timeline → checks (no_conflict + causality).

## Environment

- Python 3.12 via conda env `qiskit_qasm_py312`
- Dependencies: qiskit 2.3.1, qiskit_qasm3_import, pytest
- Run commands with: `conda run -n qiskit_qasm_py312 --no-capture-output <cmd>`

## Project structure

```
openqasm-rtir/
├── run_demo.py                 # CLI entry: validate → lower → timeline → checks
├── examples/*.qasm             # toy OpenQASM 3 test inputs
├── parser_bridge/import_qasm3.py  # qiskit bridge for syntax validation
├── rt_ir/ir.py                 # RTEvent dataclass (event_id, kind, start, duration, resource, ...)
├── rt_ir/lowering.py           # regex-based toy lowering (受控子集 only)
├── interpreter/simulate_rt.py  # timeline table printer
├── checks/no_conflict.py       # resource overlap check
├── checks/causality.py         # depends_on causality check
└── tests/test_lowering.py      # 10 tests, all passing
```

## Supported subset (v0.1)

- `h q[i];` / `x q[i];` — single-qubit gates (duration=10dt)
- `delay[Ndt] q[i];` — delay
- `c[i] = measure q[j];` — measure-assign (duration=30dt)
- `if (c[i]) { gate q[j]; }` — conditional branch (depends_on measure)

## Key design decisions

1. **Regex lowering first** — intentionally toy; AST-driven lowering is planned for v0.2
2. **Resources**: `drive_q{i}` for gates/delays, `measure_q{i}` for measurements
3. **Scheduling**: greedy sequential — start = max(qubit_ready, resource_ready, classical_ready)
4. **qiskit quirk**: bit conditions must be `if (c[0])` not `if (c[0] == 1)` — regex supports both

## Baseline & evaluation

- **Evaluation**: benchmark × independent oracle × checker × Qiskit/pytket engineering comparison
- **Related work** (NOT evaluation): VOQC / Giallar / CertiQ capability table
- **Oracle independence**: oracle must use a different path from checker (e.g., exhaustive enumeration for tiny programs)
- Three evaluation categories: resource violations, feedback violations, timing violations

## Next steps (v0.2)

1. **Formal definitions**: abstract syntax for gate subset + pulse subset, three pulse-level properties
2. **Pulse-IR**: PulseEvent dataclass (play/acquire/shift_phase/delay on port+frame)
3. **Reference semantics**: independent oracle for small programs
4. **Pulse-level checker**: port exclusivity, feedback causality, frame consistency
5. Gate-level Z3 upgrade deferred — pulse core is higher priority

## Research log

Detailed session notes in `docs/research_log.md`.
