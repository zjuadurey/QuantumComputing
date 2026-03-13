"""Bridge to qiskit_qasm3_import for syntax validation."""

from pathlib import Path


def read_text(path: str) -> str:
    """Read an OpenQASM 3 source file."""
    return Path(path).read_text()


def validate_source(src: str) -> tuple[bool, str]:
    """Validate OpenQASM 3 source via qiskit_qasm3_import.parse().

    Returns (ok, message).
    """
    try:
        from qiskit_qasm3_import import parse
        circuit = parse(src)
        return True, f"OK – {circuit.num_qubits} qubit(s), depth {circuit.depth()}"
    except Exception as exc:
        return False, f"parse error: {exc}"
