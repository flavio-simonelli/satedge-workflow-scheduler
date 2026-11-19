from __future__ import annotations

from satedge_workflow_scheduler.domain.requests import Priority, Request, Task
from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode, delivery_limit


def _request(deadline: float) -> Request:
    return Request(
        id="r",
        priority=Priority.NON_URGENT,
        origin="S1",
        release=0.0,
        deadline=deadline,
        raw_size=1.0,
        tasks=(Task(duration=10.0, output_size=1.0),),
    )


def test_hard_limit_is_the_deadline() -> None:
    assert delivery_limit(_request(40.0), DeadlineMode.HARD, horizon=100.0) == 40.0


def test_soft_limit_is_the_horizon() -> None:
    assert delivery_limit(_request(40.0), DeadlineMode.SOFT, horizon=100.0) == 100.0


def test_mode_values() -> None:
    assert DeadlineMode("hard") is DeadlineMode.HARD
    assert DeadlineMode("soft") is DeadlineMode.SOFT
