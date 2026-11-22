"""Tests of the first blocks: the rows must be the equations of the document, term by term."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import pytest
from sample import sample_instance

from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.solvers.milp.backends.fake_recording import (
    FakeRecordingBackend,
)
from satedge_workflow_scheduler.solvers.milp.blocks.assignment import AssignmentBlock
from satedge_workflow_scheduler.solvers.milp.blocks.base import (
    BuildContext,
    ConstraintBlock,
)
from satedge_workflow_scheduler.solvers.milp.blocks.task_timing import TaskTimingBlock
from satedge_workflow_scheduler.solvers.milp.expr import Row, Sense
from satedge_workflow_scheduler.solvers.milp.model_data import ModelData, TaskKey
from satedge_workflow_scheduler.solvers.milp.variables import create_decision_variables


def _context(mode: DeadlineMode) -> tuple[BuildContext, FakeRecordingBackend]:
    data = ModelData(sample_instance(), mode)
    backend = FakeRecordingBackend()
    variables = create_decision_variables(backend, data)
    return BuildContext(data, variables), backend


def _row(rows: Sequence[Row], name: str) -> Row:
    (row,) = [r for r in rows if r.name == name]
    return row


# --- the base ------------------------------------------------------------------------


def test_a_block_must_declare_its_name_equations_and_families() -> None:
    with pytest.raises(TypeError, match="equations"):

        class Incomplete(ConstraintBlock):
            name = "incomplete"
            requires = frozenset()

            def rows(
                self, ctx: BuildContext, aux: None
            ) -> Iterator[Row]:  # pragma: no cover - never built
                return iter(())


def test_build_writes_the_rows_and_counts_them() -> None:
    ctx, backend = _context(DeadlineMode.HARD)
    written = AssignmentBlock().build(ctx, backend)
    assert written == 3  # one row per task
    assert len(backend.rows) == 3


# --- assignment (2) ------------------------------------------------------------------


def test_assignment_row_sums_the_windows_of_every_satellite() -> None:
    ctx, backend = _context(DeadlineMode.HARD)
    AssignmentBlock().build(ctx, backend)
    row = _row(backend.rows, "assignment[r1.0]")
    # two satellites, one window each: u(S1) + u(S2) + ρ = 1
    assert row.sense is Sense.EQ
    assert row.rhs == pytest.approx(1.0)
    u_s1 = ctx.variables.u(TaskKey("r1", 0), "S1", 0)
    u_s2 = ctx.variables.u(TaskKey("r1", 0), "S2", 0)
    rho = ctx.variables.rejection("r1").terms
    assert set(row.terms) == {u_s1, u_s2, *rho}
    assert all(c == pytest.approx(1.0) for c in row.terms.values())


def test_assignment_under_soft_deadlines_has_no_rejection_term() -> None:
    ctx, backend = _context(DeadlineMode.SOFT)
    AssignmentBlock().build(ctx, backend)
    row = _row(backend.rows, "assignment[r2.0]")
    assert set(row.terms) == {
        ctx.variables.u(TaskKey("r2", 0), s, 0) for s in ("S1", "S2")
    }
    assert row.rhs == pytest.approx(1.0)


# --- timing (3), (4) ------------------------------------------------------------------


def test_precedence_orders_the_start_after_the_predecessor_busy_time() -> None:
    ctx, backend = _context(DeadlineMode.HARD)
    TaskTimingBlock().build(ctx, backend)
    row = _row(backend.rows, "precedence[r1.0,r1.1]")
    # t_{r1.1} − t_{r1.0} − 15 (x S1 + x S2) ≥ 0, with 15 s the duration of r1.0 on any satellite
    v = ctx.variables
    assert row.sense is Sense.GE
    assert row.rhs == pytest.approx(0.0)
    assert row.terms[v.t(TaskKey("r1", 1))] == pytest.approx(1.0)
    assert row.terms[v.t(TaskKey("r1", 0))] == pytest.approx(-1.0)
    for s in ("S1", "S2"):
        assert row.terms[v.u(TaskKey("r1", 0), s, 0)] == pytest.approx(-15.0)


def test_there_is_one_precedence_per_consecutive_pair_only() -> None:
    ctx, backend = _context(DeadlineMode.HARD)
    TaskTimingBlock().build(ctx, backend)
    names = [r.name for r in backend.rows if r.name.startswith("precedence")]
    assert names == ["precedence[r1.0,r1.1]"]  # r2 has a single task


def test_release_is_the_start_of_the_first_task_after_the_raw_data() -> None:
    ctx, backend = _context(DeadlineMode.HARD)
    TaskTimingBlock().build(ctx, backend)
    row = _row(backend.rows, "release[r1]")
    # t_{r1.0} ≥ b · (1 − ρ) with b = 0 in the sample
    assert row.sense is Sense.GE
    assert set(row.terms) == {ctx.variables.t(TaskKey("r1", 0))}
    assert row.rhs == pytest.approx(0.0)


def test_under_soft_deadlines_the_release_has_no_rejection_term() -> None:
    ctx, backend = _context(DeadlineMode.SOFT)
    TaskTimingBlock().build(ctx, backend)
    row = _row(backend.rows, "release[r2]")
    assert set(row.terms) == {ctx.variables.t(TaskKey("r2", 0))}
    assert row.rhs == pytest.approx(0.0)
