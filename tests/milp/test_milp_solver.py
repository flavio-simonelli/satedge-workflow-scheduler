"""Integration test of the MILP solver with Gurobi, on the sample instance.

The solution is checked by the scheduling module, which does not use the model: a mistake in
the constraints or in the decoding shows up as a violation there. Skipped without gurobipy or
without a licence for its version.
"""

from __future__ import annotations

import pytest
from sample import sample_instance

pytest.importorskip("gurobipy")

from satedge_workflow_scheduler.domain.solution import Solution, SolveStatus
from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.scheduling.feasibility import check
from satedge_workflow_scheduler.solvers.milp.backends.gurobi import (
    GurobiBackend,
)
from satedge_workflow_scheduler.solvers.milp.config import Level, MilpConfig
from satedge_workflow_scheduler.solvers.milp.solver import MilpSolver


@pytest.fixture(scope="module")
def solution() -> Solution:
    config = MilpConfig(level=Level.ROUTING, deadline=DeadlineMode.HARD)
    return MilpSolver(config, GurobiBackend).solve(sample_instance())


def test_the_routing_level_is_solved_to_optimality(solution: Solution) -> None:
    assert solution.status is SolveStatus.OPTIMAL


def test_both_requests_are_accepted_with_the_deliveries_of_the_hand_check(
    solution: Solution,
) -> None:
    by_id = {o.request_id: o for o in solution.outcomes}
    assert all(o.accepted for o in by_id.values())
    # r1 is delivered at 46.5 s and r2 at 14 s, as the hand computation of the calendar gives
    assert by_id["r1"].delivery_time == pytest.approx(46.5)
    assert by_id["r2"].delivery_time == pytest.approx(14.0)


def test_the_solution_passes_the_independent_checker(solution: Solution) -> None:
    violations = check(sample_instance(), solution, DeadlineMode.HARD)
    assert violations == (), [str(v) for v in violations]


def test_the_contention_level_is_solved_and_passes_the_checker() -> None:
    config = MilpConfig(level=Level.LINK_CONTENTION, deadline=DeadlineMode.HARD)
    solution = MilpSolver(config, GurobiBackend).solve(sample_instance())
    assert solution.status is SolveStatus.OPTIMAL
    violations = check(sample_instance(), solution, DeadlineMode.HARD)
    assert violations == (), [str(v) for v in violations]
