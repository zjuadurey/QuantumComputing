# openqasm-rtir

Minimal prototype: OpenQASM 3 toy subset → real-time IR → timeline + correctness checks.

## Requirements

- Python 3.12 (uses `match` statements and `X | Y` type union syntax)
- Conda environment: `qiskit_qasm_py312`

## Quick start

```bash
# Gate-level demo
conda run -n qiskit_qasm_py312 --no-capture-output python run_demo.py examples/simple_delay.qasm
conda run -n qiskit_qasm_py312 --no-capture-output python run_demo.py examples/measure_if.qasm

# Run all tests (gate-level + pulse-level)
conda run -n qiskit_qasm_py312 --no-capture-output pytest -q
```

## Supported subset

### Gate-level (v0.1)

| Construct | Example |
|-----------|---------|
| Single-qubit gate | `h q[0];` / `x q[0];` |
| Delay | `delay[20dt] q[0];` |
| Measure-assign | `c[0] = measure q[0];` |
| If-branch (single gate) | `if (c[0] == 1) { x q[0]; }` |

### Pulse-level (v0.2)

| Statement | Semantics |
|-----------|-----------|
| `Play(frame, waveform)` | Advance frame time, occupy port |
| `Acquire(frame, duration, cbit)` | Advance frame time, occupy port, set cbit_ready |
| `ShiftPhase(frame, angle)` | Shift phase, zero duration |
| `Delay(duration, frame)` | Advance frame time, no port activity |
| `IfBit(cbit, body)` | Conditional on classical bit |

## Correctness checks

### Gate-level

- **no_conflict** — no two events overlap on the same hardware resource
- **causality** — every `depends_on` edge is satisfied

### Pulse-level

- **port exclusivity** — no two operations occupy the same port simultaneously
- **feedback causality** — conditional actions wait for measurement completion
- **frame consistency** — phase matches source program semantics (correspondence check)
