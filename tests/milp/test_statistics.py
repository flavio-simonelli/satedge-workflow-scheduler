"""Statistics of a solve: time, bound, gap, nodes and size, and the reasons of rejections."""

from __future__ import annotations

import dataclasses

import pytest
from sample import sample_instance

from satedge_workflow_scheduler.domain.requests import Task
from satedge_workflow_scheduler.domain.solution import SolveStatus
from satedge_workflow_scheduler.solvers.milp.config import Level, MilpConfig
from satedge_workflow_scheduler.solvers.milp.solver import MilpSolver
from satedge_workflow_scheduler.solvers.milp.strategy import BackendFactory
from satedge_workflow_scheduler.solvers.sequential.config import SequentialConfig
from satedge_workflow_scheduler.solvers.sequential.solver import SequentialSolver


def _gurobi() -> BackendFactory:
    pytest.importorskip("gurobipy")
    from satedge_workflow_scheduler.solvers.milp.backends.gurobi import GurobiBackend

    return GurobiBackend


def test_the_sequential_run_reports_its_time_and_the_same_solution() -> None:
    instance = sample_instance()
    solver = SequentialSolver(SequentialConfig())
    run = solver.run(instance)

    assert run.statistics.method == "sequential"
    assert run.statistics.wall_time_s is not None and run.statistics.wall_time_s >= 0.0
    assert run.statistics.solver_time_s is None
    assert run.statistics.bound is None and run.statistics.nodes is None
    assert run.solution.status is SolveStatus.FEASIBLE
    assert run.solution.outcomes == solver.solve(instance).outcomes


def test_the_milp_run_reports_bound_gap_nodes_and_model_size() -> None:
    run = MilpSolver(MilpConfig(level=Level.PLMM_CAPACITY), _gurobi()).run(
        sample_instance()
    )
    stats = run.statistics

    assert run.solution.status is SolveStatus.OPTIMAL
    assert stats.method == "milp"
    assert stats.rounds == 1
    assert stats.bound is not None
    assert stats.gap == pytest.approx(0.0, abs=1e-6)
    assert stats.nodes is not None and stats.nodes >= 0.0
    assert stats.solver_time_s is not None and stats.solver_time_s >= 0.0
    assert stats.wall_time_s is not None and stats.wall_time_s >= 0.0
    assert stats.binary_variables is not None and stats.binary_variables > 0
    assert stats.constraints is not None and stats.constraints > 0


def test_the_milp_gives_the_reason_of_a_request_it_rejects() -> None:
    instance = sample_instance()
    endless = tuple(
        dataclasses.replace(r, tasks=(Task(execution_time=1e6, output_size=0.001),))
        if r.id == "r1"
        else r
        for r in instance.requests
    )
    impossible = dataclasses.replace(instance, requests=endless)
    run = MilpSolver(MilpConfig(level=Level.PLMM_CAPACITY), _gurobi()).run(impossible)

    outcome = next(o for o in run.solution.outcomes if o.request_id == "r1")
    assert not outcome.accepted
    assert outcome.rejection_reason is not None
