from __future__ import annotations

import pytest

from satedge_workflow_scheduler.common.errors import InstanceError
from satedge_workflow_scheduler.domain.solution import (
    RequestOutcome,
    Schedule,
    Solution,
    SolveStatus,
    TaskExecution,
    Transfer,
)

A_TASK = TaskExecution("a", 0, "S1", 0.0, 15.0)
A_HOP = Transfer("a", 1, "S1", "G", 15.0, 18.0)
B_TASK = TaskExecution("b", 0, "S2", 0.0, 15.0)


def test_a_task_cannot_be_executed_twice() -> None:
    with pytest.raises(InstanceError, match="executed more than once"):
        Schedule(executions=(A_TASK, TaskExecution("a", 0, "S2", 0.0, 15.0)))


def test_a_data_item_cannot_use_the_same_arc_twice() -> None:
    with pytest.raises(InstanceError, match="same arc"):
        Schedule(transfers=(A_HOP, Transfer("a", 1, "S1", "G", 30.0, 33.0)))


def test_restricted_to_keeps_only_the_given_requests() -> None:
    schedule = Schedule(executions=(A_TASK, B_TASK), transfers=(A_HOP,))
    assert schedule.restricted_to(["b"]) == Schedule(executions=(B_TASK,))
    assert schedule.restricted_to(["a"]).request_ids == frozenset({"a"})


def _solution(schedule: Schedule, *outcomes: RequestOutcome) -> Solution:
    return Solution(schedule=schedule, outcomes=outcomes, status=SolveStatus.FEASIBLE)


def test_solution_calendar_holds_exactly_the_accepted_requests() -> None:
    solution = _solution(
        Schedule(executions=(A_TASK,), transfers=(A_HOP,)),
        RequestOutcome("a", accepted=True, delivery_time=19.0),
        RequestOutcome("b", accepted=False),
    )
    assert solution.accepted_ids == ("a",)


def test_a_rejected_request_cannot_have_a_calendar() -> None:
    with pytest.raises(InstanceError, match="exactly the accepted"):
        _solution(Schedule(executions=(B_TASK,)), RequestOutcome("b", accepted=False))


def test_an_accepted_request_needs_a_calendar() -> None:
    with pytest.raises(InstanceError, match="exactly the accepted"):
        _solution(Schedule(), RequestOutcome("a", accepted=True, delivery_time=19.0))


def test_a_request_cannot_have_two_outcomes() -> None:
    with pytest.raises(InstanceError, match="more than one outcome"):
        _solution(
            Schedule(),
            RequestOutcome("b", accepted=False),
            RequestOutcome("b", accepted=False),
        )
