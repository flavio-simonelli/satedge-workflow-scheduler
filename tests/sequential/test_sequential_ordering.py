from __future__ import annotations

from satedge_workflow_scheduler.domain.requests import Priority, Request, Task
from satedge_workflow_scheduler.solvers.sequential.config import Ordering
from satedge_workflow_scheduler.solvers.sequential.factory import ordering_for
from satedge_workflow_scheduler.solvers.sequential.ordering.priority_edf import (
    PriorityEdfOrdering,
)

U, N = Priority.URGENT, Priority.NON_URGENT


def _req(
    rid: str, priority: Priority, deadline: float, release: float = 0.0
) -> Request:
    return Request(
        id=rid,
        priority=priority,
        origin="S1",
        release=release,
        deadline=deadline,
        raw_size=1.0,
        tasks=(Task(execution_time=10.0, output_size=1.0),),
    )


def _ids(requests: list[Request]) -> list[str]:
    return [r.id for r in requests]


def test_urgent_comes_first_even_with_a_later_deadline() -> None:
    requests = [_req("n", N, deadline=100.0), _req("u", U, deadline=500.0)]
    assert _ids(PriorityEdfOrdering().order(requests)) == ["u", "n"]


def test_within_a_class_earlier_deadline_comes_first() -> None:
    requests = [_req("late", N, deadline=300.0), _req("early", N, deadline=100.0)]
    assert _ids(PriorityEdfOrdering().order(requests)) == ["early", "late"]


def test_equal_class_and_deadline_is_broken_by_release() -> None:
    requests = [
        _req("later", U, deadline=50.0, release=20.0),
        _req("earlier", U, deadline=50.0, release=5.0),
    ]
    assert _ids(PriorityEdfOrdering().order(requests)) == ["earlier", "later"]


def test_full_ties_keep_the_instance_order() -> None:
    requests = [
        _req("first", N, deadline=50.0, release=5.0),
        _req("second", N, deadline=50.0, release=5.0),
        _req("third", N, deadline=50.0, release=5.0),
    ]
    assert _ids(PriorityEdfOrdering().order(requests)) == ["first", "second", "third"]


def test_order_does_not_modify_the_input() -> None:
    requests = [_req("n", N, deadline=100.0), _req("u", U, deadline=500.0)]
    PriorityEdfOrdering().order(requests)
    assert _ids(requests) == ["n", "u"]


def test_the_factory_gives_priority_edf() -> None:
    assert isinstance(ordering_for(Ordering.PRIORITY_EDF), PriorityEdfOrdering)
