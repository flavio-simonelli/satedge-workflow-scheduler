from __future__ import annotations

from dataclasses import replace

import pytest
from sample import online_instance, sample_instance

from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.solution import Schedule, TaskExecution
from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.scheduling.feasibility import check
from satedge_workflow_scheduler.solvers.errors import InfeasibleModelError
from satedge_workflow_scheduler.solvers.sequential.config import SequentialConfig
from satedge_workflow_scheduler.solvers.sequential.errors import CommittedCalendarError
from satedge_workflow_scheduler.solvers.sequential.solver import SequentialSolver


def _with_deadline(
    instance: ProblemInstance, request_id: str, deadline: float
) -> ProblemInstance:
    requests = tuple(
        replace(r, deadline=deadline) if r.id == request_id else r
        for r in instance.requests
    )
    return replace(instance, requests=requests)


def test_hard_mode_serves_every_request_that_fits_and_the_calendar_is_feasible() -> (
    None
):
    instance = sample_instance()
    solution = SequentialSolver(SequentialConfig(deadline=DeadlineMode.HARD)).solve(
        instance
    )
    assert all(o.accepted for o in solution.outcomes)
    assert not check(instance, solution, DeadlineMode.HARD)


def test_hard_mode_rejects_the_request_that_cannot_meet_its_deadline_and_serves_the_others() -> (
    None
):
    instance = _with_deadline(sample_instance(), "r1", deadline=5.0)
    solution = SequentialSolver(SequentialConfig(deadline=DeadlineMode.HARD)).solve(
        instance
    )
    outcomes = {o.request_id: o for o in solution.outcomes}
    assert not outcomes["r1"].accepted
    assert outcomes["r1"].rejection_reason is not None
    assert outcomes["r2"].accepted
    assert not check(instance, solution, DeadlineMode.HARD)


def test_soft_mode_serves_every_request_and_the_calendar_is_feasible() -> None:
    instance = sample_instance()
    solution = SequentialSolver(SequentialConfig(deadline=DeadlineMode.SOFT)).solve(
        instance
    )
    assert all(o.accepted for o in solution.outcomes)
    assert not check(instance, solution, DeadlineMode.SOFT)


def test_soft_mode_is_infeasible_when_a_request_cannot_be_delivered_by_the_horizon() -> (
    None
):
    instance = replace(sample_instance(), horizon=5.0)
    solver = SequentialSolver(SequentialConfig(deadline=DeadlineMode.SOFT))
    with pytest.raises(InfeasibleModelError) as error:
        solver.solve(instance)
    assert set(error.value.reasons) == {"r1", "r2"} or set(error.value.reasons) <= {
        "r1",
        "r2",
    }
    assert error.value.reasons


def test_a_committed_request_is_accepted_with_the_delivery_of_its_calendar() -> None:
    instance = online_instance({"S1": 0.0, "S2": 10.0})
    solution = SequentialSolver(SequentialConfig(deadline=DeadlineMode.HARD)).solve(
        instance
    )
    outcomes = {o.request_id: o for o in solution.outcomes}
    # r2 is committed: its output leaves S2 -> G at 8, arrives at 9, and the latency is 5 s
    assert outcomes["r2"].accepted
    assert outcomes["r2"].delivery_time == pytest.approx(14.0)
    assert outcomes["r1"].accepted
    assert not check(instance, solution, DeadlineMode.HARD)


def test_the_new_requests_wait_for_the_plan_release_of_their_satellites() -> None:
    instance = online_instance({"S1": 10.0, "S2": 10.0})
    solution = SequentialSolver(SequentialConfig(deadline=DeadlineMode.SOFT)).solve(
        instance
    )
    new = [e for e in solution.schedule.executions if e.request_id == "r1"]
    assert new
    assert all(e.start >= 10.0 - 1e-9 for e in new if e.satellite == "S1")
    assert not check(instance, solution, DeadlineMode.SOFT)


def test_a_committed_request_without_delivery_is_a_fault_of_the_calendar() -> None:
    base = online_instance({"S1": 0.0, "S2": 0.0})
    broken = replace(
        base,
        committed=Schedule(executions=(TaskExecution("r2", 0, "S2", 0.0, 8.0),)),
    )
    solver = SequentialSolver(SequentialConfig())
    with pytest.raises(CommittedCalendarError):
        solver.solve(broken)
