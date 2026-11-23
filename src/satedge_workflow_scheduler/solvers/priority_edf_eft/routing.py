"""Time-dependent earliest-arrival routing of a data item over the contact plan.

The router does not reserve anything: it only finds, for each reachable node, the route that
delivers a data item first, given the reservations already in the ledger.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from satedge_workflow_scheduler.domain.network import Arc, Network, WindowedAvailability
from satedge_workflow_scheduler.domain.nodes import NodeId
from satedge_workflow_scheduler.scheduling.earliest import Reached, earliest_arrivals
from satedge_workflow_scheduler.solvers.priority_edf_eft.ledger import (
    ArcKey,
    ResourceLedger,
)


@dataclass(frozen=True, slots=True)
class Hop:
    """One transmission of a route.

    Attributes:
        source: Tail node of the arc.
        target: Head node of the arc.
        start: Start of the transmission in seconds.
        end: End of the transmission (``start + transmission time``).
        arrival: Arrival of the last bit (``end + latency``).
    """

    source: NodeId
    target: NodeId
    start: float
    end: float
    arrival: float


@dataclass(frozen=True, slots=True)
class Route:
    """A sequence of hops from the source, with the arrival of the data at its end.

    Attributes:
        hops: The transmissions, in order. Empty for the source itself.
        arrival: Arrival of the data at the last node. Equals the ready instant if the
            route is empty.
    """

    hops: tuple[Hop, ...]
    arrival: float


class EarliestArrivalRouter:
    """Finds the earliest-arrival routes from a node, avoiding the busy arcs of the ledger."""

    def __init__(self, network: Network, ledger: ResourceLedger, eps: float) -> None:
        self._network = network
        self._ledger = ledger
        self._eps = eps

    def departure(self, arc: Arc, ready: float, size: float) -> float | None:
        """Earliest instant at which ``size`` Gb can start on ``arc``, or ``None``.

        The data cannot leave before ``ready``, nor before the node is available. On a
        windowed arc the whole transmission plus the latency must fit in one window.
        """
        ready = max(ready, self._ledger.not_before(arc.source))
        tx = arc.transmission_time(size)
        busy = self._ledger.timeline(ArcKey(arc.source, arc.target))

        if not isinstance(arc.availability, WindowedAvailability):
            return busy.first_gap(ready, tx, self._eps)

        for window in arc.availability.windows:
            if window.end < ready + tx + arc.latency - self._eps:
                continue
            start = busy.first_gap(max(ready, window.start), tx, self._eps)
            if start + tx + arc.latency <= window.end + self._eps:
                return start
        return None

    def routes_from(
        self,
        source: NodeId,
        ready: float,
        size: float,
        limit: float,
        *,
        to_ground: bool,
    ) -> dict[NodeId, Route]:
        """Earliest-arrival routes from ``source`` to every node reachable before ``limit``.

        A Dijkstra search on arrival instants. It is correct because the network is FIFO:
        leaving later never makes the data arrive earlier. The ground station is never
        expanded, and it is entered only if ``to_ground`` is true. Routes that arrive after
        ``limit`` are discarded. The source itself is included with an empty route.
        """
        ground = self._network.ground_station.id
        network = self._network

        def crossing(index: int, at_tail: float) -> tuple[float, float] | None:
            arc = network.arcs[index]
            if arc.target == ground and not to_ground:
                return None
            start = self.departure(arc, at_tail, size)
            if start is None:
                return None
            arrival = start + arc.transmission_time(size) + arc.latency
            if arrival > limit + self._eps:
                return None
            return start, arrival

        reached = earliest_arrivals(
            network,
            {source: ready},
            crossing,
            relays=lambda node: node != ground,
            tolerance=self._eps,
        )
        return {node: self._route(reached, node, size) for node in reached}

    def _route(
        self, reached: Mapping[NodeId, Reached], node: NodeId, size: float
    ) -> Route:
        """The route that reaches ``node``, with the hops from the source to it."""
        hops: list[Hop] = []
        current = reached[node]
        while current.previous is not None and current.arc is not None:
            arc = self._network.arcs[current.arc]
            start = current.departure
            assert start is not None
            hops.append(
                Hop(
                    arc.source,
                    arc.target,
                    start,
                    start + arc.transmission_time(size),
                    current.arrival,
                )
            )
            current = reached[current.previous]
        return Route(tuple(reversed(hops)), reached[node].arrival)
