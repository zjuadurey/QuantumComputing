# openqasm-rtir — Project Context

## What this is

Research prototype: OpenQASM 3 toy subset → real-time IR → timeline + correctness checks.
NOT a full quantum compiler. Focus is on timing/feedback/resource semantics formalization.

See `PROJECT_INTENT.md` for full research motivation.

## Research positioning

- Core question: how to formally define, reliably lower, and verify correctness of quantum programs with timing, pulse, and measurement-feedback semantics
- Structural gap: existing verified quantum compilation (VOQC/SQIR/CertiQ) covers gate-level equivalence but NOT timing/resource/feedback semantics
- Target venues: POPL / PLDI / ICSE (CCF-A)
- Three contributions: (1) formal semantics, (2) verifiable compilation chain, (3) regression + safety property verification
- Phased roadmap: Phase 1 quick prototype → Phase 2 Z3 automated verification → Phase 3 Coq or SMT/contract formalization

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

## Known v0.2 directions (not yet implemented)

1. AST-driven lowering via qiskit QuantumCircuit instructions
2. Z3 constraint-based formal checking
3. Multi-qubit gates (cx) + barrier support
4. More explicit feedback latency modeling

## Research log

Detailed session notes in `docs/research_log.md`.
