"""Tests of the warm start: the heuristic calendar is handed to the solver, when it applies.

The recording backend shows what the solver receives, without a licence. The Gurobi test checks
that starting from the heuristic does not change the optimum.
"""

from __future__ import annotations

import pytest
from sample import online_instance, sample_instance

from satedge_workflow_scheduler.domain.solution import SolveStatus
from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.scheduling.feasibility import check
from satedge_workflow_scheduler.solvers.errors import SolverError
from satedge_workflow_scheduler.solvers.milp.backends.fake_recording import (
    FakeRecordingBackend,
)
from satedge_workflow_scheduler.solvers.milp.config import Level, MilpConfig
from satedge_workflow_scheduler.solvers.milp.solver import MilpSolver
from satedge_workflow_scheduler.solvers.milp.strategy import warm_start_of


def test_without_the_option_there_is_no_warm_start() -> None:
    config = MilpConfig(level=Level.ROUTING)
    assert warm_start_of(sample_instance(), config) is None


def test_the_heuristic_calendar_is_the_warm_start_of_an_offline_instance() -> None:
    config = MilpConfig(level=Level.ROUTING, warm_start=True)
    instance = sample_instance()
    start = warm_start_of(instance, config)
    assert start is not None
    assert len(start.executions) == 3 and len(start.transfers) == 3


def test_an_online_instance_starts_from_a_calendar_that_keeps_the_committed_requests() -> (
    None
):
    config = MilpConfig(level=Level.ROUTING, warm_start=True)
    instance = online_instance({"S1": 0.0, "S2": 0.0})
    start = warm_start_of(instance, config)
    assert start is not None
    assert start.restricted_to(["r2"]) == instance.committed


def test_the_solver_receives_the_starting_values_before_it_solves() -> None:
    backends: list[FakeRecordingBackend] = []

    def factory() -> FakeRecordingBackend:
        backend = FakeRecordingBackend()
        backends.append(backend)
        return backend

    config = MilpConfig(level=Level.ROUTING, warm_start=True)
    # the fake backend has no scripted result: the solve reports no solution, and the solver
    # raises, but only after it has received the starting values
    with pytest.raises(SolverError):
        MilpSolver(config, factory).solve(sample_instance())
    (backend,) = backends
    assert backend.starts
    assert set(backend.starts.values()) >= {0.0, 1.0}


def test_the_optimum_is_the_same_with_and_without_the_warm_start() -> None:
    pytest.importorskip("gurobipy")
    from satedge_workflow_scheduler.solvers.milp.backends.gurobi import GurobiBackend

    cold = MilpSolver(
        MilpConfig(level=Level.ROUTING, warm_start=False), GurobiBackend
    ).solve(sample_instance())
    warm_config = MilpConfig(level=Level.ROUTING, warm_start=True)
    warm = MilpSolver(warm_config, GurobiBackend).solve(sample_instance())
    assert warm.status is SolveStatus.OPTIMAL
    assert check(sample_instance(), warm, DeadlineMode.HARD) == ()
    assert cold.objective is not None and warm.objective is not None
    assert warm.objective.primary == pytest.approx(cold.objective.primary)
    assert warm.objective.response == pytest.approx(cold.objective.response)
