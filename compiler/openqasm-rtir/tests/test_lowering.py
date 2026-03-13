"""Tests for the toy lowering pipeline."""

import sys, os
# ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from rt_ir.lowering import lower_qasm3
from checks.no_conflict import check_no_conflict
from checks.causality import check_causality

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


# ── simple_delay.qasm ────────────────────────────────────────────

def test_simple_delay_event_count():
    src = (EXAMPLES / "simple_delay.qasm").read_text()
    events = lower_qasm3(src)
    # h, delay, measure, if-branch -> 4 events
    assert len(events) == 4


def test_simple_delay_kinds():
    src = (EXAMPLES / "simple_delay.qasm").read_text()
    events = lower_qasm3(src)
    kinds = [e.kind for e in events]
    assert kinds == ["gate", "delay", "measure", "branch"]


def test_simple_delay_no_conflict():
    src = (EXAMPLES / "simple_delay.qasm").read_text()
    events = lower_qasm3(src)
    ok, errs = check_no_conflict(events)
    assert ok, errs


def test_simple_delay_causality():
    src = (EXAMPLES / "simple_delay.qasm").read_text()
    events = lower_qasm3(src)
    ok, errs = check_causality(events)
    assert ok, errs


def test_simple_delay_branch_depends_on_measure():
    src = (EXAMPLES / "simple_delay.qasm").read_text()
    events = lower_qasm3(src)
    branch = [e for e in events if e.kind == "branch"][0]
    measure = [e for e in events if e.kind == "measure"][0]
    assert measure.event_id in branch.depends_on


# ── measure_if.qasm ──────────────────────────────────────────────

def test_measure_if_event_count():
    src = (EXAMPLES / "measure_if.qasm").read_text()
    events = lower_qasm3(src)
    # x, measure, if-branch -> 3 events
    assert len(events) == 3


def test_measure_if_kinds():
    src = (EXAMPLES / "measure_if.qasm").read_text()
    events = lower_qasm3(src)
    kinds = [e.kind for e in events]
    assert kinds == ["gate", "measure", "branch"]


def test_measure_if_no_conflict():
    src = (EXAMPLES / "measure_if.qasm").read_text()
    events = lower_qasm3(src)
    ok, errs = check_no_conflict(events)
    assert ok, errs


def test_measure_if_causality():
    src = (EXAMPLES / "measure_if.qasm").read_text()
    events = lower_qasm3(src)
    ok, errs = check_causality(events)
    assert ok, errs


def test_measure_if_timing():
    """Branch must start no earlier than measure end."""
    src = (EXAMPLES / "measure_if.qasm").read_text()
    events = lower_qasm3(src)
    measure = [e for e in events if e.kind == "measure"][0]
    branch = [e for e in events if e.kind == "branch"][0]
    assert branch.start >= measure.end
