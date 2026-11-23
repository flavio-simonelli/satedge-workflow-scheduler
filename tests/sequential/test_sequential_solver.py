from __future__ import annotations

from dataclasses import replace

import pytest
from sample import sample_instance

from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.scheduling.feasibility import check
from satedge_workflow_scheduler.solvers.errors import InfeasibleModelError
from satedge_workflow_scheduler.solvers.sequential.config import SequentialConfig
from satedge_workflow_scheduler.solvers.sequential.errors import SequentialConfigError
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
    with pytest.raises(InfeasibleModelError) as error:
        SequentialSolver(SequentialConfig(deadline=DeadlineMode.SOFT)).solve(instance)
    assert set(error.value.reasons) == {"r1", "r2"} or set(error.value.reasons) <= {
        "r1",
        "r2",
    }
    assert error.value.reasons


def test_an_instance_with_a_plan_release_is_refused_for_now() -> None:
    instance = replace(sample_instance(), plan_release={"S1": 10.0, "S2": 0.0})
    with pytest.raises(SequentialConfigError):
        SequentialSolver(SequentialConfig()).solve(instance)
