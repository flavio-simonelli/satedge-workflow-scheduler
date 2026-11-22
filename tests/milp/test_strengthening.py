"""Tests of the strengthenings (rules 7 and 8): the rows, and the same optimum as without them."""

from __future__ import annotations

import pytest
from sample import Built, build

from satedge_workflow_scheduler.solvers.milp.blocks.link_window import LinkWindowBlock
from satedge_workflow_scheduler.solvers.milp.blocks.path import (
    PathBlock,
)
from satedge_workflow_scheduler.solvers.milp.blocks.store_and_forward import (
    StoreAndForwardBlock,
    TransmissionStarts,
)
from satedge_workflow_scheduler.solvers.milp.blocks.strengthening import (
    ArrivalWithoutBigMBlock,
    ContactCapacityBlock,
)
from satedge_workflow_scheduler.solvers.milp.expr import Sense
from satedge_workflow_scheduler.solvers.milp.model_data import DataKey


def _built() -> Built:
    return build(
        [
            PathBlock(),
            StoreAndForwardBlock(),
            ArrivalWithoutBigMBlock(),
            LinkWindowBlock(),
            ContactCapacityBlock(),
        ]
    )


def test_rule_seven_rows() -> None:
    # unused: 5 data × 3 arcs = 15; arrival without big-M: 3 data with a task consumer × 3 arcs
    # = 9; delivery without big-M: 2 final data × 3 arcs = 6
    assert _built().model.rows["arrival without big-m"] == 15 + 9 + 6


def test_a_transmission_is_zero_on_an_arc_the_data_does_not_cross() -> None:
    built = _built()
    starts = built.model.families.get(TransmissionStarts)
    routing = built.model.families.get(
        __import__(
            "satedge_workflow_scheduler.solvers.milp.blocks.path",
            fromlist=["RoutingVariables"],
        ).RoutingVariables
    )
    a = DataKey("r1", 0)
    row = built.row(f"unused[{a},0]")
    # ξ_{a,0} − τ̂ χ_{a,0} ≤ 0, with τ̂ = 500 for r1
    assert row.sense is Sense.LE
    assert row.rhs == pytest.approx(0.0)
    assert row.terms[starts.xi(a, 0)] == pytest.approx(1.0)
    assert row.terms[routing.crossed[(a, 0)]] == pytest.approx(-500.0)


def test_rule_eight_has_one_row_per_window_of_a_windowed_arc() -> None:
    built = _built()
    # the windowed arc S2→G has one window [0, 1000]: one row
    assert built.model.rows["contact capacity"] == 1
    row = built.row("contact[2,0]")
    assert row.sense is Sense.LE
    assert row.rhs == pytest.approx(1000.0)


def test_the_strengthened_model_has_the_same_optimum_as_the_plain_one() -> None:
    pytest.importorskip("gurobipy")
    from sample import sample_instance

    from satedge_workflow_scheduler.solvers.milp.backends.gurobi import GurobiBackend
    from satedge_workflow_scheduler.solvers.milp.config import (
        Level,
        MilpConfig,
        Strengthening,
    )
    from satedge_workflow_scheduler.solvers.milp.solver import MilpSolver

    plain = MilpSolver(MilpConfig(level=Level.ROUTING), GurobiBackend).solve(
        sample_instance()
    )
    strengthened = MilpSolver(
        MilpConfig(
            level=Level.ROUTING,
            strengthening=Strengthening(
                arrival_without_big_m=True, contact_capacity=True
            ),
        ),
        GurobiBackend,
    ).solve(sample_instance())
    assert plain.objective is not None and strengthened.objective is not None
    assert strengthened.objective.primary == pytest.approx(plain.objective.primary)
    assert strengthened.objective.response == pytest.approx(plain.objective.response)
