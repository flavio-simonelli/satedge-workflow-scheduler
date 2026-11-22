"""Tests of the computing window and task overlap blocks: rows must be (6)-(10), term by term."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from sample import sample_instance

from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.solvers.milp.backends.fake_recording import (
    FakeRecordingBackend,
)
from satedge_workflow_scheduler.solvers.milp.blocks.base import BuildContext
from satedge_workflow_scheduler.solvers.milp.blocks.computing_window import (
    ComputingWindowBlock,
)
from satedge_workflow_scheduler.solvers.milp.blocks.task_overlap import TaskOverlapBlock
from satedge_workflow_scheduler.solvers.milp.expr import Row, Sense
from satedge_workflow_scheduler.solvers.milp.model_data import ModelData, TaskKey
from satedge_workflow_scheduler.solvers.milp.variables import create_decision_variables


def _context() -> tuple[BuildContext, FakeRecordingBackend]:
    data = ModelData(sample_instance(), DeadlineMode.HARD)
    backend = FakeRecordingBackend()
    variables = create_decision_variables(backend, data)
    return BuildContext(data, variables), backend


def _row(rows: Sequence[Row], name: str) -> Row:
    (row,) = [r for r in rows if r.name == name]
    return row


# --- computing windows (6)-(7) -------------------------------------------------------


def test_computing_window_writes_two_rows_per_task() -> None:
    ctx, backend = _context()
    assert ComputingWindowBlock().build(ctx, backend) == 2 * 3


def test_start_is_not_before_the_start_of_the_chosen_window() -> None:
    ctx, backend = _context()
    ComputingWindowBlock().build(ctx, backend)
    row = _row(backend.rows, "window_start[r1.0]")
    # the window starts at 0 s: the only term is t ≥ 0
    assert row.sense is Sense.GE
    assert dict(row.terms) == {ctx.variables.t(TaskKey("r1", 0)): 1.0}
    assert row.rhs == pytest.approx(0.0)


def test_end_of_the_task_is_before_the_end_of_its_window() -> None:
    ctx, backend = _context()
    ComputingWindowBlock().build(ctx, backend)
    row = _row(backend.rows, "window_end[r1.0]")
    v = ctx.variables
    # window [0, 100]: 100 u − t − 15 (x) ≥ 0, and x is the sum of the u, so u carries 85
    assert row.terms[v.t(TaskKey("r1", 0))] == pytest.approx(-1.0)
    assert row.terms[v.u(TaskKey("r1", 0), "S1", 0)] == pytest.approx(85.0)
    assert row.terms[v.u(TaskKey("r1", 0), "S2", 0)] == pytest.approx(85.0)
    assert row.rhs == pytest.approx(0.0)


# --- task overlap (8)-(10) -----------------------------------------------------------


def test_overlap_writes_the_shared_window_rows_and_the_two_orders() -> None:
    ctx, backend = _context()
    # two pairs of tasks of different requests, one window on each of two satellites
    # (8): 2 pairs × 2 satellites × 1 window = 4, (9) and (10): 2 pairs × 2 = 4
    assert TaskOverlapBlock().build(ctx, backend) == 8


def test_same_window_rows_force_same_when_both_tasks_use_the_window() -> None:
    ctx, backend = _context()
    TaskOverlapBlock().build(ctx, backend)
    row = _row(backend.rows, "same[r1.0,r2.0,S1,0]")
    assert row.sense is Sense.GE
    assert row.rhs == pytest.approx(-1.0)
    assert row.terms[ctx.variables.u(TaskKey("r1", 0), "S1", 0)] == pytest.approx(-1.0)
    assert row.terms[ctx.variables.u(TaskKey("r2", 0), "S1", 0)] == pytest.approx(-1.0)


def test_order_rows_use_the_larger_limit_of_the_two_requests_as_big_m() -> None:
    ctx, backend = _context()
    TaskOverlapBlock().build(ctx, backend)
    before = _row(backend.rows, "before[r1.0,r2.0]")
    # the limits are 500 (r1) and 550 (r2): M = 550, and both (9) and (10) carry M(1−·)
    assert before.sense is Sense.GE
    assert before.rhs == pytest.approx(-2 * 550.0)
    aux = [v for v in backend.variables if v.name.startswith("order[")]
    assert aux and all(v.kind.value == "binary" for v in aux)
