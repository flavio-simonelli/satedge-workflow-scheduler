"""Planning of one whole request: placement of its tasks, delivery and PLMM check."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from satedge_workflow_scheduler.common.intervals import TimeWindow
from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.nodes import NodeId
from satedge_workflow_scheduler.domain.requests import Request
from satedge_workflow_scheduler.domain.solution import TaskExecution, Transfer
from satedge_workflow_scheduler.scheduling.plmm_occupancy import (
    PlmmOccupancyInterval,
    plmm_intervals,
    within_capacity,
)
from satedge_workflow_scheduler.solvers.priority_edf_eft.ledger import (
    ArcKey,
    HpcKey,
    ResourceLedger,
    Transaction,
)
from satedge_workflow_scheduler.solvers.priority_edf_eft.placement import EftPlacement
from satedge_workflow_scheduler.solvers.priority_edf_eft.routing import (
    EarliestArrivalRouter,
    Route,
)


@dataclass(frozen=True)
class RequestPlan:
    """A planned request: its executions, its transfers and its delivery instant.

    Attributes:
        request_id: Identifier of the request.
        executions: Executions of its tasks, in pipeline order.
        transfers: Transmissions of its data.
        delivery_time: Instant at which the final output reaches the ground station.
    """

    request_id: str
    executions: tuple[TaskExecution, ...]
    transfers: tuple[Transfer, ...]
    delivery_time: float


@dataclass(frozen=True)
class PlanFailure:
    """A request that could not be planned.

    Attributes:
        reason: Human readable reason.
        plmm: Whether the failure comes only from the PLMM capacity. In that case a later
            retry, at another instant, may succeed.
    """

    reason: str
    plmm: bool


class RequestPlanner:
    """Plans one request inside a transaction opened by the solver."""

    def __init__(
        self,
        instance: ProblemInstance,
        ledger: ResourceLedger,
        placement: EftPlacement,
        router: EarliestArrivalRouter,
        eps: float,
    ) -> None:
        self._instance = instance
        self._ledger = ledger
        self._placement = placement
        self._router = router
        self._eps = eps

    def plan(
        self,
        request: Request,
        limit: float,
        not_before: float | None,
        tx: Transaction,
        extra_plmm: Mapping[NodeId, Sequence[PlmmOccupancyInterval]],
    ) -> RequestPlan | PlanFailure:
        """Plan ``request`` before ``limit``, reserving everything in ``tx``.

        The raw data leaves its origin no earlier than ``not_before``, if given.
        ``extra_plmm`` holds the PLMM occupancies that other pending requests will need.
        Nothing is kept in ``tx`` unless the request is planned and the caller commits it.
        """
        network = self._instance.network
        ground = network.ground_station.id
        executions: list[TaskExecution] = []
        transfers: list[Transfer] = []

        location = request.origin
        ready = (
            request.release if not_before is None else max(request.release, not_before)
        )

        for task_index in range(len(request.tasks)):
            placed = self._placement.place(request, task_index, location, ready, limit)
            if placed is None:
                return PlanFailure("no feasible placement before the limit", plmm=False)
            # the input of task k is data item k
            self._reserve_route(
                tx, request.id, task_index, placed.input_route, transfers
            )
            tx.reserve(HpcKey(placed.satellite), TimeWindow(placed.start, placed.end))
            executions.append(
                TaskExecution(
                    request.id, task_index, placed.satellite, placed.start, placed.end
                )
            )
            location = placed.satellite
            ready = placed.end

        route = self._router.routes_from(
            location, ready, request.output_size, limit, to_ground=True
        ).get(ground)
        if route is None:
            return PlanFailure(
                "no delivery to the ground station before the limit", plmm=False
            )
        self._reserve_route(tx, request.id, len(request.tasks), route, transfers)
        delivery = route.arrival

        occupancies = plmm_intervals(self._instance, executions, transfers)
        for node, items in occupancies.items():
            capacity = network.satellite(node).plmm_capacity
            present = (
                *self._ledger.plmm(node),
                *extra_plmm.get(node, ()),
                *items,
            )
            if not within_capacity(present, capacity, self._eps):
                return PlanFailure("PLMM capacity exceeded", plmm=True)
        for node, items in occupancies.items():
            tx.add_occupancies(node, items)

        return RequestPlan(request.id, tuple(executions), tuple(transfers), delivery)

    def _reserve_route(
        self,
        tx: Transaction,
        request_id: str,
        data_index: int,
        route: Route,
        transfers: list[Transfer],
    ) -> None:
        """Reserve the arcs of ``route`` and record its transfers of data ``data_index``."""
        for hop in route.hops:
            tx.reserve(ArcKey(hop.source, hop.target), TimeWindow(hop.start, hop.end))
            transfers.append(
                Transfer(
                    request_id, data_index, hop.source, hop.target, hop.start, hop.end
                )
            )
