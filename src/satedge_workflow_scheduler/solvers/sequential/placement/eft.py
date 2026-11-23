"""Placement of one request, task by task, in earliest-finish (EFT) order.

A data item occupies the PLMM of a node from when it starts entering it to when it has left.
When the data arrives the start is known, but the end depends on a later choice: the raw data
stays on its origin until it leaves or the first task uses it, and the output of a task stays
on its satellite until it leaves or the next task uses it. This occupancy stays *open*, and the
choice that sends or consumes the data closes it, finding room for it too.

For each task, in pipeline order:

1. the input is routed from where it is to every satellite it can reach, by the router (every
   node it leaves must have room for its stay);
2. the candidates, satellite, route and compatible window, are in EFT order
   (:func:`task_candidates`);
3. the first candidate whose PLMM has room for what it fixes is chosen: the input stays on the
   satellite until the end of the task, and the output is there from the start of its writing;
4. the choice is reserved tentatively, and the next task starts from the satellite it ran on.

For the last task the delivery is part of the choice: the output is routed to the ground station,
and the candidate that delivers first is chosen.

The search goes task by task, trying the candidates of each task in EFT order. If the tasks
that follow cannot be placed after a candidate, the search goes back and tries the next candidate
of the earlier task. Each such return costs one unit of the budget ``max_backtracks``: when the
budget is spent, the request fails with the reason of the deepest failure. With no budget the
search is the greedy placement: the first candidate that fits is the only one tried.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass

from satedge_workflow_scheduler.common.intervals import TIME_TOLERANCE
from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.nodes import NodeId
from satedge_workflow_scheduler.domain.requests import Request
from satedge_workflow_scheduler.domain.solution import Schedule, TaskExecution, Transfer
from satedge_workflow_scheduler.scheduling.bounds import RequestBounds
from satedge_workflow_scheduler.scheduling.plmm_occupancy import PlmmOccupancyInterval
from satedge_workflow_scheduler.solvers.sequential.placement.candidates import (
    TaskCandidate,
    task_candidates,
)
from satedge_workflow_scheduler.solvers.sequential.placement.ports import (
    PlanFailure,
    RequestPlan,
)
from satedge_workflow_scheduler.solvers.sequential.reservations import Reservations
from satedge_workflow_scheduler.solvers.sequential.routing.ports import (
    Route,
    RouteQuery,
    Router,
)


@dataclass
class _Budget:
    """How many returns to an earlier task the search may still make for one request."""

    remaining: int

    def spend(self) -> bool:
        """Use one return if there is one left; ``False`` when the budget is spent."""
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        return True


@dataclass(frozen=True)
class _Progress:
    """Where the placement of a request has got to.

    Attributes:
        reservations: All the reservations, with the tentative ones of the request.
        location: The node the current data item is on.
        held_since: Since when it occupies the PLMM there: its open occupancy.
        ready: From when it can leave or be used.
        executions: The tasks placed so far.
        transfers: The transmissions placed so far.
    """

    reservations: Reservations
    location: NodeId
    held_since: float
    ready: float
    executions: tuple[TaskExecution, ...] = ()
    transfers: tuple[Transfer, ...] = ()


class EftScheduler:
    """Places a request task by task, each task at the candidate that finishes first.

    Args:
        router: How the data items are routed.
        max_backtracks: How many returns to an earlier task the search may make for one request.
            ``0`` is the greedy placement.
        eps: Tolerance for comparisons between instants.
    """

    def __init__(
        self, router: Router, max_backtracks: int = 0, eps: float = TIME_TOLERANCE
    ) -> None:
        self._router = router
        self._max_backtracks = max_backtracks
        self._eps = eps

    def schedule(
        self,
        instance: ProblemInstance,
        request: Request,
        bounds: RequestBounds,
        reservations: Reservations,
    ) -> RequestPlan | PlanFailure:
        """Plan ``request`` by ``bounds.limit`` around ``reservations``."""
        progress = _Progress(
            reservations, request.origin, request.release, request.release
        )
        budget = _Budget(self._max_backtracks)
        return self._search(instance, request, bounds, 0, progress, budget)

    def _search(
        self,
        instance: ProblemInstance,
        request: Request,
        bounds: RequestBounds,
        task_index: int,
        progress: _Progress,
        budget: _Budget,
    ) -> RequestPlan | PlanFailure:
        """Place the tasks from ``task_index`` on, trying the candidates in EFT order.

        A candidate that cannot be placed, because its PLMM has no room, is skipped at no cost:
        the greedy placement skips it too. A candidate whose later tasks fail costs one unit of the
        budget before the next candidate is tried.
        """
        if task_index == len(request.tasks) - 1:
            return self._place_last_task(instance, request, bounds, progress)
        candidates = self._candidates(instance, request, bounds, task_index, progress)
        if not candidates:
            return self._no_candidate(instance, request, bounds, task_index, progress)
        failure: PlanFailure = PlanFailure(
            f"task {task_index}: no candidate has room in the PLMM"
        )
        tried = False
        for candidate in candidates:
            after = self._fix(request, task_index, candidate, progress)
            if after is None:
                continue
            if tried and not budget.spend():
                return failure
            tried = True
            result = self._search(
                instance, request, bounds, task_index + 1, after, budget
            )
            if isinstance(result, RequestPlan):
                return result
            failure = result
        return failure

    def _place_last_task(
        self,
        instance: ProblemInstance,
        request: Request,
        bounds: RequestBounds,
        progress: _Progress,
    ) -> RequestPlan | PlanFailure:
        """Place the last task and deliver its output: the candidate that delivers first."""
        task_index = len(request.tasks) - 1
        candidates = self._candidates(instance, request, bounds, task_index, progress)
        if not candidates:
            return self._no_candidate(instance, request, bounds, task_index, progress)
        ground = instance.network.ground_station.id
        best: tuple[float, TaskCandidate, _Progress, Route] | None = None
        for candidate in candidates:
            after = self._fix(request, task_index, candidate, progress)
            if after is None:
                continue
            routes = self._routes(
                instance, after, request.output_size, bounds.limit, to_ground=True
            )
            deliveries = routes.get(ground, ())
            # the ground station has no PLMM: its first route is the earliest delivery
            if deliveries and (best is None or deliveries[0].arrival < best[0]):
                best = (deliveries[0].arrival, candidate, after, deliveries[0])
        if best is None:
            return PlanFailure(
                f"task {task_index}: no candidate delivers the output by the limit"
            )
        delivery, _, chosen, route = best
        final = _transfers(request.id, len(request.tasks), route)
        schedule = Schedule(chosen.executions, (*chosen.transfers, *final))
        return RequestPlan(request.id, schedule, delivery)

    def _no_candidate(
        self,
        instance: ProblemInstance,
        request: Request,
        bounds: RequestBounds,
        task_index: int,
        progress: _Progress,
    ) -> PlanFailure:
        """Why a task has no candidate: the PLMM on the way of its input, or time.

        The candidates are searched again without the PLMM, only to tell the two apart.
        """
        relaxed = dataclasses.replace(
            progress, reservations=progress.reservations.without_plmm()
        )
        if self._candidates(instance, request, bounds, task_index, relaxed):
            return PlanFailure(
                f"task {task_index}: no room in the PLMM for its input on the way"
            )
        return PlanFailure(f"task {task_index}: no satellite can run it in time")

    def _candidates(
        self,
        instance: ProblemInstance,
        request: Request,
        bounds: RequestBounds,
        task_index: int,
        progress: _Progress,
    ) -> list[TaskCandidate]:
        size = request.data_sizes[task_index]
        routes = self._routes(instance, progress, size, bounds.limit, to_ground=False)
        return task_candidates(
            instance,
            progress.reservations,
            request,
            bounds,
            task_index,
            routes,
            self._eps,
        )

    def _routes(
        self,
        instance: ProblemInstance,
        progress: _Progress,
        size: float,
        limit: float,
        *,
        to_ground: bool,
    ) -> Mapping[NodeId, tuple[Route, ...]]:
        query = RouteQuery(
            source=progress.location,
            held_since=progress.held_since,
            ready=progress.ready,
            size=size,
            limit=limit,
            to_ground=to_ground,
        )
        return self._router.routes_from(instance, progress.reservations, query)

    def _fix(
        self,
        request: Request,
        task_index: int,
        candidate: TaskCandidate,
        progress: _Progress,
    ) -> _Progress | None:
        """Reserve ``candidate`` tentatively, or ``None`` if its PLMM has no room.

        The stays of the input route were checked by the router. On the satellite of the task the
        input stays until the end of the task, and the output is there from the start of its
        writing: both must fit together. The output then stays open.
        """
        satellite = candidate.satellite
        route = candidate.input_route
        input_size = request.data_sizes[task_index]
        output_size = request.data_sizes[task_index + 1]

        def occupancy(start: float, end: float) -> PlmmOccupancyInterval:
            return PlmmOccupancyInterval(request.id, task_index, input_size, start, end)

        reservations = progress.reservations.with_occupancies(
            (stay.node, occupancy(stay.start, stay.end)) for stay in route.stays
        )
        task_input = occupancy(route.held_since, candidate.end)
        if not reservations.plmm_has_room(
            satellite, input_size, task_input.start, task_input.end, self._eps
        ):
            return None
        reservations = reservations.with_occupancies([(satellite.id, task_input)])
        write_start = candidate.end - satellite.hpc_io_time(output_size)
        if not reservations.plmm_has_room(
            satellite, output_size, write_start, candidate.end, self._eps
        ):
            return None

        execution = TaskExecution(
            request.id, task_index, satellite.id, candidate.start, candidate.end
        )
        transfers = _transfers(request.id, task_index, route)
        reservations = reservations.with_execution(execution)
        for transfer in transfers:
            reservations = reservations.with_transfer(transfer)
        return _Progress(
            reservations=reservations,
            location=satellite.id,
            held_since=write_start,
            ready=candidate.end,
            executions=(*progress.executions, execution),
            transfers=(*progress.transfers, *transfers),
        )


def _transfers(request_id: str, data_index: int, route: Route) -> tuple[Transfer, ...]:
    return tuple(
        Transfer(request_id, data_index, hop.source, hop.target, hop.start, hop.end)
        for hop in route.hops
    )
