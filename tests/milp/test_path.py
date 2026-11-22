"""Tests of the path block: the rows must be (11)-(13), term by term."""

from __future__ import annotations

import pytest
from sample import sample_instance

from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.solvers.milp.backends.fake_recording import (
    FakeRecordingBackend,
)
from satedge_workflow_scheduler.solvers.milp.blocks.base import BuildContext
from satedge_workflow_scheduler.solvers.milp.blocks.path import PathBlock
from satedge_workflow_scheduler.solvers.milp.expr import Row, Sense
from satedge_workflow_scheduler.solvers.milp.model_data import (
    DataKey,
    ModelData,
    TaskKey,
)
from satedge_workflow_scheduler.solvers.milp.variables import create_decision_variables


def _context(mode: DeadlineMode) -> tuple[BuildContext, FakeRecordingBackend]:
    data = ModelData(sample_instance(), mode)
    backend = FakeRecordingBackend()
    variables = create_decision_variables(
        backend, data, frozenset({"rejection", "u", "chi"})
    )
    return BuildContext(data, variables), backend


def _row(rows: list[Row], name: str) -> Row:
    (row,) = [r for r in rows if r.name == name]
    return row


def test_hard_deadlines_write_flow_ends_and_rejection_rows() -> None:
    ctx, backend = _context(DeadlineMode.HARD)
    # 5 data, 3 nodes (S1, S2, G), 3 arcs: 15 flow rows, 30 ends rows, 15 rejection rows
    assert PathBlock().build(ctx, backend) == 60


def test_soft_deadlines_have_no_rejection_rows() -> None:
    ctx, backend = _context(DeadlineMode.SOFT)
    assert PathBlock().build(ctx, backend) == 45


def test_flow_at_the_origin_of_the_raw_data() -> None:
    ctx, backend = _context(DeadlineMode.HARD)
    PathBlock().build(ctx, backend)
    raw = DataKey("r1", 0)
    row = _row(backend.rows, f"flow[{raw},S1]")
    v = ctx.variables
    # χ(S1→S2) − χ(S2→S1) = (1 − ρ) − x_{r1.0,S1}, with x = u (one window)
    assert row.sense is Sense.EQ
    assert row.rhs == pytest.approx(1.0)
    assert row.terms[v.arc_chi[(raw, 0)]] == pytest.approx(1.0)
    assert row.terms[v.arc_chi[(raw, 1)]] == pytest.approx(-1.0)
    assert row.terms[v.rejected["r1"]] == pytest.approx(1.0)
    assert row.terms[v.u(TaskKey("r1", 0), "S1", 0)] == pytest.approx(1.0)


def test_the_raw_data_cannot_enter_its_origin_when_accepted() -> None:
    ctx, backend = _context(DeadlineMode.HARD)
    PathBlock().build(ctx, backend)
    raw = DataKey("r1", 0)
    row = _row(backend.rows, f"ends-in[{raw},S1]")
    # χ(S2→S1) ≤ 1 − (1 − ρ) = ρ
    assert row.sense is Sense.LE
    assert row.rhs == pytest.approx(0.0)
    assert row.terms[ctx.variables.arc_chi[(raw, 1)]] == pytest.approx(1.0)
