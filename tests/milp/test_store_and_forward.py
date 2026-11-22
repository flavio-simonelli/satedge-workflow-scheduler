"""Tests of the store-and-forward block: the rows must be (14a)-(16b), term by term."""

from __future__ import annotations

import pytest
from sample import sample_instance

from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.solvers.milp.backends.fake_recording import (
    FakeRecordingBackend,
)
from satedge_workflow_scheduler.solvers.milp.blocks.base import BuildContext
from satedge_workflow_scheduler.solvers.milp.blocks.store_and_forward import (
    StoreAndForwardBlock,
)
from satedge_workflow_scheduler.solvers.milp.expr import Row, Sense
from satedge_workflow_scheduler.solvers.milp.model_data import DataKey, ModelData
from satedge_workflow_scheduler.solvers.milp.variables import create_decision_variables

FAMILIES = frozenset({"rejection", "t", "u", "chi", "nu", "xi"})


def _context() -> tuple[BuildContext, FakeRecordingBackend]:
    data = ModelData(sample_instance(), DeadlineMode.HARD)
    backend = FakeRecordingBackend()
    variables = create_decision_variables(backend, data, FAMILIES)
    return BuildContext(data, variables), backend


def _row(rows: list[Row], name: str) -> Row:
    (row,) = [r for r in rows if r.name == name]
    return row


def test_rows_per_equation() -> None:
    ctx, backend = _context()
    # (14a)+(14b): 5 data × 3 arcs = 15; (15): 5 data × (1 + 2) pairs at S1, S2 = 15;
    # (16a): 3 data with a consumer × 2 arcs into the satellites = 6;
    # (16b): 2 final data × 1 arc into the ground station = 2
    assert StoreAndForwardBlock().build(ctx, backend) == 15 + 15 + 6 + 2


def test_raw_data_starts_at_its_release_on_every_arc() -> None:
    ctx, backend = _context()
    StoreAndForwardBlock().build(ctx, backend)
    raw = DataKey("r1", 0)
    row = _row(backend.rows, f"raw[{raw},0]")
    # ξ_{a,0} − b_r χ_{a,0} ≥ 0, with b_r = 0 in the sample
    assert row.sense is Sense.GE
    assert row.rhs == pytest.approx(0.0)
    assert row.terms[ctx.variables.xi(raw, 0)] == pytest.approx(1.0)
    # with b_r = 0 the term b_r · χ is zero, and the row keeps only ξ
    assert ctx.variables.arc_chi[(raw, 0)] not in row.terms


def test_final_output_reaches_the_ground_station_before_its_limit() -> None:
    ctx, backend = _context()
    StoreAndForwardBlock().build(ctx, backend)
    final = DataKey("r1", 2)
    row = _row(backend.rows, f"deliver[{final},2]")
    # τ̂ + c(1 − χ) − ξ − c ≥ 0 with τ̂ = 500: ξ_{a,2} + c χ_{a,2} ≤ 500
    assert row.sense is Sense.GE
    assert row.rhs == pytest.approx(-500.0)
    assert row.terms[ctx.variables.xi(final, 2)] == pytest.approx(-1.0)
