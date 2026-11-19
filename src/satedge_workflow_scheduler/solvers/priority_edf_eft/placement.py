"""Placement of a task: the satellite and the instant that let it finish first (EFT)."""

from __future__ import annotations

from dataclasses import dataclass

from satedge_workflow_scheduler.domain.network import Network
from satedge_workflow_scheduler.domain.nodes import NodeId, Satellite
from satedge_workflow_scheduler.solvers.priority_edf_eft.ledger import (
    HpcKey,
    ResourceLedger,
)
from satedge_workflow_scheduler.solvers.priority_edf_eft.routing import (
    EarliestArrivalRouter,
    Route,
)


@dataclass(frozen=True, slots=True)
class TaskPlacement:
    """Where and when a task runs, and the route that brings its input to the satellite.

    Attributes:
        satellite: Satellite that runs the task.
        start: Start of the task in seconds.
        end: End of the task in seconds.
        input_route: Route of the input data from its location to ``satellite``.
    """

    satellite: NodeId
    start: float
    end: float
    input_route: Route


class EftPlacement:
    """Chooses the satellite on which a task finishes first."""

    def __init__(
        self,
        network: Network,
        ledger: ResourceLedger,
        router: EarliestArrivalRouter,
        eps: float,
    ) -> None:
        self._network = network
        self._ledger = ledger
        self._router = router
        self._eps = eps

    def earliest_slot(
        self, satellite: Satellite, ready: float, duration: float
    ) -> float | None:
        """Earliest start of a task of ``duration`` on ``satellite``, or ``None``.

        The task must lie entirely inside one HPC window and must not overlap the HPC
        reservations already in the ledger.
        """
        ready = max(ready, self._ledger.not_before(satellite.id))
        busy = self._ledger.timeline(HpcKey(satellite.id))
        for window in satellite.hpc_windows:
            if window.end < ready + duration - self._eps:
                continue
            start = busy.first_gap(max(ready, window.start), duration, self._eps)
            if start + duration <= window.end + self._eps:
                return start
        return None

    def place(
        self,
        location: NodeId,
        ready: float,
        size: float,
        duration: float,
        limit: float,
    ) -> TaskPlacement | None:
        """Best placement of a task whose input of ``size`` Gb is at ``location``.

        Among the satellites that can run the task before ``limit``, the one where it ends
        first is chosen. Ties go to the satellite reached with fewer hops, then to the one
        that comes first in the network.
        """
        routes = self._router.routes_from(location, ready, size, limit, to_ground=False)
        best: tuple[tuple[float, int, int], TaskPlacement] | None = None
        for index, satellite in enumerate(self._network.satellites):
            route = routes.get(satellite.id)
            if route is None:
                continue
            start = self.earliest_slot(satellite, route.arrival, duration)
            if start is None:
                continue
            end = start + duration
            if end > limit + self._eps:
                continue
            key = (end, len(route.hops), index)
            if best is None or key < best[0]:
                best = (
                    key,
                    TaskPlacement(satellite.id, start, end, route),
                )
        return None if best is None else best[1]
