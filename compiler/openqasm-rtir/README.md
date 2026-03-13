# openqasm-rtir

Minimal prototype: OpenQASM 3 toy subset → real-time IR → timeline + correctness checks.

## Quick start

```bash
python run_demo.py examples/simple_delay.qasm
python run_demo.py examples/measure_if.qasm
pytest -q
```

## Supported subset (v0.1)

| Construct | Example |
|-----------|---------|
| Single-qubit gate | `h q[0];` / `x q[0];` |
| Delay | `delay[20dt] q[0];` |
| Measure-assign | `c[0] = measure q[0];` |
| If-branch (single gate) | `if (c[0] == 1) { x q[0]; }` |

## Checks

- **no_conflict** – no two events overlap on the same hardware resource
- **causality** – every `depends_on` edge is satisfied (dependency finishes before dependent starts)
