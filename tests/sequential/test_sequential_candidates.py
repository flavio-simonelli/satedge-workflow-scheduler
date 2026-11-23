from __future__ import annotations

from sample import sample_instance

from satedge_workflow_scheduler.domain.solution import TaskExecution
from satedge_workflow_scheduler.scheduling.bounds import request_bounds
from satedge_workflow_scheduler.solvers.sequential.placement.candidates import (
    TaskCandidate,
    task_candidates,
)
from satedge_workflow_scheduler.solvers.sequential.reservations import Reservations
from satedge_workflow_scheduler.solvers.sequential.routing.ports import RouteQuery
from satedge_workflow_scheduler.solvers.sequential.routing.safe_interval import (
    SafeIntervalRouter,
)

LIMIT = 600.0


def _candidates(reservations: Reservations, task_index: int = 0) -> list[TaskCandidate]:
    instance = sample_instance()
    request = instance.request("r1")
    bounds = request_bounds(instance, request, LIMIT)
    query = RouteQuery(
        source=request.origin,
        held_since=request.release,
        ready=request.release,
        size=request.data_sizes[task_index],
        limit=LIMIT,
    )
    routes = SafeIntervalRouter().routes_from(instance, reservations, query)
    return task_candidates(instance, reservations, request, bounds, task_index, routes)


def test_the_candidates_come_in_earliest_finish_order() -> None:
    found = _candidates(Reservations())
    assert found, "a feasible task must have candidates"
    ends = [c.end for c in found]
    assert ends == sorted(ends)


def test_the_satellite_of_the_data_wins_when_it_is_free() -> None:
    # r1 is on S1, and no transmission is needed to run its first task there
    best = _candidates(Reservations())[0]
    assert best.satellite.id == "S1"
    assert best.input_route.hops == ()


def test_a_busy_hpc_pushes_the_task_later() -> None:
    free = _candidates(Reservations())[0]
    busy = Reservations().with_execution(TaskExecution("other", 0, "S1", 0.0, 40.0))
    later = next(c for c in _candidates(busy) if c.satellite.id == "S1")
    assert later.start >= 40.0
    assert later.end > free.end


def test_the_criteria_of_the_eft_order_are_applied_in_turn() -> None:
    route = _candidates(Reservations())[0].input_route
    satellite = _candidates(Reservations())[0].satellite

    def make(
        end: float, hops: int, position: int, window_start: float
    ) -> TaskCandidate:
        return TaskCandidate(
            end=end,
            hops=hops,
            position=position,
            window_start=window_start,
            satellite=satellite,
            start=end - 1.0,
            input_route=route,
        )

    later_end = make(end=9.0, hops=0, position=0, window_start=0.0)
    more_hops = make(end=5.0, hops=2, position=0, window_start=0.0)
    later_network = make(end=5.0, hops=1, position=3, window_start=0.0)
    first_network = make(end=5.0, hops=1, position=1, window_start=0.0)
    later_window = make(end=5.0, hops=1, position=1, window_start=7.0)
    first_window = make(end=5.0, hops=1, position=1, window_start=2.0)
    ordered = sorted(
        [later_window, first_network, later_network, more_hops, later_end, first_window]
    )
    # by end first; then fewer hops; then the earlier satellite; then the earlier window
    assert ordered == [
        first_network,
        first_window,
        later_window,
        later_network,
        more_hops,
        later_end,
    ]
