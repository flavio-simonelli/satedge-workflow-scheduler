"""Safe-interval routing (SIPP): the data may wait wherever its PLMM has room.

The earliest arrival on a node is not always the best one with the PLMM: arriving earlier makes
the stay longer, and the PLMM may be full during part of it, where a later arrival would fit.
Safe Interval Path Planning splits the time of each node into its *free intervals*: the maximal
intervals in which its PLMM has room for the data item (:func:`free_intervals`). A stay has room
exactly when it lies in one of them.

The search is a Dijkstra over the states ``(node, free interval)``, with the arrival as the
label. Inside one free interval the earliest arrival dominates: the data can wait there as long
as it likes. Between two intervals it does not, and each keeps its own label. From a state
``(u, I)`` the data can reach each free interval ``J`` of a neighbour ``v``:

1. it waits on ``u`` until its first bit can reach ``v`` inside ``J``, and for the arc and the
   plan as usual: the earliest transmission from ``max(arrival, J.start - λ)``;
2. the transmission must end inside ``I``, which the stay on ``u`` must not leave;
3. the first bit must arrive in ``J``, and the whole data by its end.

The arcs are FIFO, so a state that fails one of these for ``J`` fails it for every later interval
of ``v`` too: the expansion stops there.

A route never goes back to a node it has already visited. This is the only reason the search may
miss a route that exists, since the earliest label of a state may come through a node that a
later, discarded label would have avoided.

The search has the same shape as :func:`~satedge_workflow_scheduler.scheduling.earliest.earliest_arrivals`
(a Dijkstra with the arrival as the label), but its states are pairs and its expansion depends on
the route so far, so the two are not shared.
"""

from __future__ import annotations

import heapq
import itertools
import math

from satedge_workflow_scheduler.common.intervals import TIME_TOLERANCE
from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.network import Arc
from satedge_workflow_scheduler.domain.nodes import NodeId
from satedge_workflow_scheduler.scheduling.plmm_occupancy import FreeInterval
from satedge_workflow_scheduler.scheduling.windows import arc_windows, earliest_finish
from satedge_workflow_scheduler.solvers.sequential.reservations import Reservations
from satedge_workflow_scheduler.solvers.sequential.routing.ports import (
    Hop,
    Route,
    RouteQuery,
    Stay,
)

type _State = tuple[NodeId, int]
"""A node and the index of one of its free intervals."""

_ALWAYS_FREE = (FreeInterval(0.0, math.inf),)
"""The free time of the ground station, which has no PLMM."""


class SafeIntervalRouter:
    """Routes over the contact plan, one per node and free interval of its PLMM.

    Args:
        eps: Tolerance for comparisons between instants.
    """

    def __init__(self, eps: float = TIME_TOLERANCE) -> None:
        self._eps = eps

    def routes_from(
        self, instance: ProblemInstance, reservations: Reservations, query: RouteQuery
    ) -> dict[NodeId, tuple[Route, ...]]:
        """The earliest route to every free interval reachable by ``query.limit``."""
        eps = self._eps
        network = instance.network
        ground = network.ground_station.id
        cache: dict[NodeId, tuple[FreeInterval, ...]] = {}

        def intervals(node: NodeId) -> tuple[FreeInterval, ...]:
            if node == ground:
                return _ALWAYS_FREE
            if node not in cache:
                cache[node] = reservations.plmm_free_intervals(
                    network.satellite(node), query.size, eps
                )
            return cache[node]

        source = query.source
        first = next(
            (
                k
                for k, interval in enumerate(intervals(source))
                if interval.contains(query.held_since, eps)
            ),
            None,
        )
        if first is None:
            # the PLMM of the source has no room even now: the data can go nowhere
            stuck = Route(
                source, (), (), query.ready, query.held_since, query.held_since
            )
            return {source: (stuck,)}

        start: _State = (source, first)
        best: dict[_State, Route] = {
            start: Route(
                source,
                (),
                (),
                query.ready,
                query.held_since,
                intervals(source)[first].end,
            )
        }
        settled: set[_State] = set()
        order = itertools.count()
        heap = [(query.ready, next(order), start)]

        while heap:
            _, _, state = heapq.heappop(heap)
            if state in settled:
                continue
            settled.add(state)
            node, _ = state
            if node == ground:
                continue
            here = best[state]
            departure = max(here.arrival, instance.release_of(node))
            if math.isinf(departure):
                continue  # the plan never reaches this node: nothing new leaves it
            visited = {hop.target for hop in here.hops} | {source}
            for index in network.outgoing[node]:
                arc = network.arcs[index]
                if arc.target in visited or (
                    arc.target == ground and not query.to_ground
                ):
                    continue
                for j, target in enumerate(intervals(arc.target)):
                    if target.end <= departure + eps or (arc.target, j) in settled:
                        continue
                    ready = max(departure, target.start - arc.latency)
                    hop = earliest_hop(arc, reservations, ready, query.size, eps)
                    if (
                        hop is None
                        or hop.end > here.hold_until + eps
                        or hop.arrival > query.limit + eps
                    ):
                        break  # a later interval would only need a later hop
                    first_bit = hop.start + arc.latency
                    if not target.contains(first_bit, eps):
                        continue  # the hop jumped past this interval
                    if hop.arrival > target.end + eps:
                        continue  # the data would not be all there before it closes
                    current = best.get((arc.target, j))
                    if current is not None and current.arrival <= hop.arrival + eps:
                        continue
                    best[(arc.target, j)] = Route(
                        target=arc.target,
                        hops=(*here.hops, hop),
                        stays=(*here.stays, Stay(node, here.held_since, hop.end)),
                        arrival=hop.arrival,
                        held_since=first_bit,
                        hold_until=target.end,
                    )
                    heapq.heappush(heap, (hop.arrival, next(order), (arc.target, j)))

        routes: dict[NodeId, list[Route]] = {}
        for state in sorted(settled, key=lambda s: (best[s].arrival, s[1])):
            routes.setdefault(state[0], []).append(best[state])
        return {node: tuple(found) for node, found in routes.items()}


def earliest_hop(
    arc: Arc,
    reservations: Reservations,
    ready: float,
    size: float,
    eps: float = TIME_TOLERANCE,
) -> Hop | None:
    """The earliest transmission of ``size`` Gb on ``arc`` from ``ready``, if any.

    Transmission and propagation must lie in one visibility window; the arc is busy only while
    it transmits. A later ``ready`` never gives an earlier hop.
    """
    traversal = arc.traversal_time(size)
    arrival = earliest_finish(
        arc_windows(arc.availability),
        ready,
        traversal,
        busy=reservations.arc_busy(arc.source, arc.target),
        occupied=arc.transmission_time(size),
        eps=eps,
    )
    if arrival is None:
        return None
    start = arrival - traversal
    return Hop(
        arc.source, arc.target, start, start + arc.transmission_time(size), arrival
    )
