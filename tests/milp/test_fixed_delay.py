"""Tests of level 3: the fixed delays, the route linearization and the delivery without routing.

The fixed delays are computed by hand on the sample instance: S1 → S2 is 1 Gb/s and 1 s of
latency, S2 → G is 2 Gb/s and 5 s of latency, and S1 → S2 → G is the only way to the ground.
"""

from __future__ import annotations

import pytest
from sample import Built, build, sample_instance

from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.solvers.milp.blocks.delivery import DeliveryTimes
from satedge_workflow_scheduler.solvers.milp.blocks.fixed_delay import (
    FixedDelayBlock,
    FixedRoute,
    FixedRouteBlock,
)
from satedge_workflow_scheduler.solvers.milp.config import Level, MilpConfig
from satedge_workflow_scheduler.solvers.milp.expr import Sense
from satedge_workflow_scheduler.solvers.milp.model_data import ModelData, TaskKey


def _built() -> Built:
    return build([FixedRouteBlock(), FixedDelayBlock()])


def test_the_fixed_delay_is_the_shortest_path_of_transmission_and_latency() -> None:
    data = ModelData(sample_instance(), DeadlineMode.HARD)
    # S1 → S2 for 2 Gb: 2 / 1 + 1 = 3 s
    assert data.fixed_delay("S1", "S2", 2.0) == pytest.approx(3.0)
    # S1 → S2 → G for 1 Gb: (1 / 1 + 1) + (1 / 2 + 5) = 7.5 s
    assert data.fixed_delay("S1", "G", 1.0) == pytest.approx(7.5)
    assert data.fixed_delay("S2", "S2", 1.0) == 0.0


def test_a_node_with_no_path_gets_a_delay_no_time_limit_allows() -> None:
    data = ModelData(sample_instance(), DeadlineMode.HARD)
    # the ground station has no outgoing arc: nothing leaves it
    assert data.fixed_delay("G", "S1", 1.0) == pytest.approx(data.horizon + 1.0)


def test_one_route_per_precedence_and_pair_of_satellites() -> None:
    built = _built()
    routes = built.model.families.get(FixedRoute)
    # 1 precedence (r1.0 → r1.1) × 2 × 2 satellites
    assert len(routes.z) == 1 * 2 * 2


def test_rows_of_the_route_and_of_the_delays() -> None:
    built = _built()
    # route: 1 precedence × (2 columns + 2 rows) = 4
    assert built.model.rows["fixed route"] == 4
    # delays: 1 precedence + 2 raw + 2 × (delivery and deadline) = 7
    assert built.model.rows["fixed delay"] == 1 + 2 + 2 * 2


def test_the_precedence_pays_the_delay_of_the_pair_of_satellites() -> None:
    built = _built()
    routes = built.model.families.get(FixedRoute)
    i, j = TaskKey("r1", 0), TaskKey("r1", 1)
    row = built.row(f"precedence[{i},{j}]")
    # the output of r1.0 is 3 Gb: S1 → S2 costs 3 + 1 = 4 s; the same satellite costs nothing
    assert row.sense is Sense.GE
    assert row.terms[routes.at(i, j, "S1", "S2")] == pytest.approx(-4.0)
    assert routes.at(i, j, "S1", "S1") not in row.terms


def test_the_delivery_comes_after_the_last_task_and_the_way_to_the_ground() -> None:
    built = _built()
    deliveries = built.model.families.get(DeliveryTimes)
    row = built.row("delivery[r2]")
    # r2 runs on S2 (its origin) and sends 2 Gb to G: 2 / 2 + 5 = 6 s
    assert row.sense is Sense.GE
    assert row.terms[deliveries.deliv("r2")] == pytest.approx(1.0)


def test_a_level_three_solution_is_solved_and_no_slower_than_the_routed_level() -> None:
    pytest.importorskip("gurobipy")
    from satedge_workflow_scheduler.solvers.milp.backends.gurobi import GurobiBackend
    from satedge_workflow_scheduler.solvers.milp.solver import MilpSolver

    instance = sample_instance()
    fixed = MilpSolver(
        MilpConfig(level=Level.FIXED_DELAY_WINDOWS), GurobiBackend
    ).solve(instance)
    routed = MilpSolver(MilpConfig(level=Level.ROUTING), GurobiBackend).solve(instance)
    assert fixed.schedule.transfers == ()
    assert all(o.accepted for o in fixed.outcomes)
    assert fixed.objective is not None and routed.objective is not None
    # the fixed delays are lower bounds of the transfers: the level three is a relaxation
    assert fixed.objective.response <= routed.objective.response + 1e-6
