"""Toy regex-based lowering from OpenQASM 3 subset to RTEvent list.

Supported subset:
  h q[i];  /  x q[i];  /  delay[Ndt] q[i];
  c[i] = measure q[j];
  if (c[i] == 1) { gate q[j]; }
"""

from __future__ import annotations

import re
from collections import defaultdict
from .ir import RTEvent

# ── hardcoded timing ──────────────────────────────────────────────
GATE_DUR = 10       # dt for h / x
MEASURE_DUR = 30    # dt for measure

# ── regex patterns ────────────────────────────────────────────────
RE_GATE = re.compile(r"^(h|x)\s+q\[(\d+)\]\s*;")
RE_DELAY = re.compile(r"^delay\[(\d+)dt\]\s+q\[(\d+)\]\s*;")
RE_MEASURE = re.compile(r"^c\[(\d+)\]\s*=\s*measure\s+q\[(\d+)\]\s*;")
# matches both: if (c[0]) { ... } and if (c[0] == 1) { ... }
RE_IF = re.compile(
    r"^if\s*\(\s*c\[(\d+)\]\s*(?:==\s*(?:1|true)\s*)?\)\s*\{\s*(h|x)\s+q\[(\d+)\]\s*;\s*\}"
)


def lower_qasm3(source: str) -> list[RTEvent]:
    """Lower an OpenQASM 3 toy-subset source string to a list of RTEvents."""
    events: list[RTEvent] = []
    eid = 0

    # tracking maps
    qubit_ready: dict[int, int] = defaultdict(int)      # qubit -> earliest free time
    resource_ready: dict[str, int] = defaultdict(int)    # resource -> earliest free time
    classical_ready: dict[int, int] = defaultdict(int)   # cbit -> readable time
    last_writer: dict[int, int | None] = defaultdict(lambda: None)  # cbit -> event_id

    for raw_line in source.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # ── single-qubit gate ──
        m = RE_GATE.match(line)
        if m:
            gate, qi = m.group(1), int(m.group(2))
            res = f"drive_q{qi}"
            start = max(qubit_ready[qi], resource_ready[res])
            ev = RTEvent(
                event_id=eid, kind="gate", start=start,
                duration=GATE_DUR, resource=res, qubit=qi,
                payload=gate,
            )
            events.append(ev)
            qubit_ready[qi] = ev.end
            resource_ready[res] = ev.end
            eid += 1
            continue

        # ── delay ──
        m = RE_DELAY.match(line)
        if m:
            dur, qi = int(m.group(1)), int(m.group(2))
            res = f"drive_q{qi}"
            start = max(qubit_ready[qi], resource_ready[res])
            ev = RTEvent(
                event_id=eid, kind="delay", start=start,
                duration=dur, resource=res, qubit=qi,
                payload=f"delay[{dur}dt]",
            )
            events.append(ev)
            qubit_ready[qi] = ev.end
            resource_ready[res] = ev.end
            eid += 1
            continue

        # ── measure ──
        m = RE_MEASURE.match(line)
        if m:
            ci, qi = int(m.group(1)), int(m.group(2))
            res = f"measure_q{qi}"
            start = max(qubit_ready[qi], resource_ready[res])
            ev = RTEvent(
                event_id=eid, kind="measure", start=start,
                duration=MEASURE_DUR, resource=res, qubit=qi,
                creg=ci, payload="measure",
            )
            events.append(ev)
            qubit_ready[qi] = ev.end
            resource_ready[res] = ev.end
            classical_ready[ci] = ev.end    # cbit readable after measure
            last_writer[ci] = eid
            eid += 1
            continue

        # ── if-branch ──
        m = RE_IF.match(line)
        if m:
            ci, gate, qi = int(m.group(1)), m.group(2), int(m.group(3))
            res = f"drive_q{qi}"
            start = max(
                qubit_ready[qi],
                resource_ready[res],
                classical_ready[ci],    # must wait for measurement result
            )
            deps: list[int] = []
            writer = last_writer[ci]
            if writer is not None:
                deps.append(writer)
            ev = RTEvent(
                event_id=eid, kind="branch", start=start,
                duration=GATE_DUR, resource=res, qubit=qi,
                creg=ci, condition=f"c[{ci}]",
                payload=gate, depends_on=deps,
            )
            events.append(ev)
            qubit_ready[qi] = ev.end
            resource_ready[res] = ev.end
            eid += 1
            continue

        # lines we skip silently: OPENQASM, include, qubit/bit decl, etc.

    return events
