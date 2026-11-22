"""Tests of the path block: the rows must be (11)-(13), term by term."""

from __future__ import annotations

import pytest
from sample import build

from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.solvers.milp.blocks.path import (
    PathBlock,
    RoutingVariables,
)
from satedge_workflow_scheduler.solvers.milp.expr import Sense
from satedge_workflow_scheduler.solvers.milp.model_data import DataKey, TaskKey


def test_hard_deadlines_write_flow_ends_and_rejection_rows() -> None:
    built = build([PathBlock()])
    # 5 data, 3 nodes (S1, S2, G), 3 arcs: 15 flow rows, 30 ends rows, 15 rejection rows
    assert built.model.rows["path"] == 60


def test_soft_deadlines_have_no_rejection_rows() -> None:
    built = build([PathBlock()], DeadlineMode.SOFT)
    assert built.model.rows["path"] == 45


def test_routing_variables_are_chi_on_the_arcs_and_nu_on_the_windows() -> None:
    built = build([PathBlock()])
    routing = built.model.families.get(RoutingVariables)
    raw = DataKey("r1", 0)
    # arc 0 is S1→S2, always available: χ is a variable; arc 2 is S2→G, with windows
    assert (raw, 0) in routing.crossed
    assert (raw, 2) not in routing.crossed
    assert routing.chi(raw, 2).terms == {routing.nu(raw, 2, 0): 1.0}


def test_flow_at_the_origin_of_the_raw_data() -> None:
    built = build([PathBlock()])
    v, routing = built.ctx.variables, built.model.families.get(RoutingVariables)
    raw = DataKey("r1", 0)
    row = built.row(f"flow[{raw},S1]")
    # χ(S1→S2) − χ(S2→S1) = (1 − ρ) − x_{r1.0,S1}, with x = u (one window)
    assert row.sense is Sense.EQ
    assert row.rhs == pytest.approx(1.0)
    assert row.terms[routing.crossed[(raw, 0)]] == pytest.approx(1.0)
    assert row.terms[routing.crossed[(raw, 1)]] == pytest.approx(-1.0)
    assert row.terms[v.rejected["r1"]] == pytest.approx(1.0)
    assert row.terms[v.u(TaskKey("r1", 0), "S1", 0)] == pytest.approx(1.0)


def test_the_raw_data_cannot_enter_its_origin_when_accepted() -> None:
    built = build([PathBlock()])
    routing = built.model.families.get(RoutingVariables)
    raw = DataKey("r1", 0)
    row = built.row(f"ends-in[{raw},S1]")
    # χ(S2→S1) ≤ 1 − (1 − ρ) = ρ
    assert row.sense is Sense.LE
    assert row.rhs == pytest.approx(0.0)
    assert row.terms[routing.crossed[(raw, 1)]] == pytest.approx(1.0)
