#!/usr/bin/env python3
"""Run the openqasm-rtir demo pipeline on a .qasm file.

Usage:  python run_demo.py examples/simple_delay.qasm
"""

import sys
from pathlib import Path

# ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from parser_bridge.import_qasm3 import read_text, validate_source
from rt_ir.lowering import lower_qasm3
from interpreter.simulate_rt import print_timeline
from checks.no_conflict import check_no_conflict
from checks.causality import check_causality


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python run_demo.py <file.qasm>")
        sys.exit(1)

    path = sys.argv[1]
    src = read_text(path)
    print(f"=== source: {path} ===\n{src}")

    # 1. syntax validation via qiskit
    ok, msg = validate_source(src)
    print(f"[validate] {msg}")
    if not ok:
        sys.exit(1)

    # 2. lowering
    events = lower_qasm3(src)
    print(f"\n=== timeline ({len(events)} events) ===")
    print_timeline(events)

    # 3. checks
    ok_c, errs_c = check_no_conflict(events)
    ok_d, errs_d = check_causality(events)

    print(f"\n=== checks ===")
    print(f"  no_conflict : {'PASS' if ok_c else 'FAIL'}")
    for e in errs_c:
        print(f"    - {e}")
    print(f"  causality   : {'PASS' if ok_d else 'FAIL'}")
    for e in errs_d:
        print(f"    - {e}")

    if ok_c and ok_d:
        print("\nAll checks passed.")
    else:
        print("\nSome checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
