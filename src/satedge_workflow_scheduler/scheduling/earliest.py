"""Earliest arrivals on a network: a Dijkstra search whose arcs may be crossed at some instants only.

The heuristic routes the data with it, on the arcs that the ledger leaves free, and the MILP takes
the static delays from it, on the arcs as they are. The search does not know either case: the
caller says, for each arc and each instant the data is ready at its tail, when the data can cross
it. That is the only difference between the uses.

The search is correct because the network is FIFO: leaving later never makes the data arrive
earlier, so the first time a node is reached is the earliest.
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from satedge_workflow_scheduler.domain.network import Network
from satedge_workflow_scheduler.domain.nodes import NodeId

Crossing = Callable[[int, float], tuple[float, float] | None]
"""``crossing(arc, ready)``: ``(departure, arrival)`` of the data that is ready at the tail of the
arc at ``ready``, or ``None`` if the arc cannot be crossed from then on."""


@dataclass(frozen=True, slots=True)
class Reached:
    """How a node is reached: the earliest arrival, and the last hop that brings the data there.

    Attributes:
        arrival: The earliest instant the data reaches the node.
        previous: The node the last hop comes from, or ``None`` at a source.
        arc: The arc of the last hop, or ``None`` at a source.
        departure: The departure of the last hop, or ``None`` at a source.
    """

    arrival: float
    previous: NodeId | None
    arc: int | None
    departure: float | None


def earliest_arrivals(
    network: Network,
    sources: Mapping[NodeId, float],
    crossing: Crossing,
    *,
    relays: Callable[[NodeId], bool] = lambda node: True,
    tolerance: float = 0.0,
) -> dict[NodeId, Reached]:
    """The earliest arrival at every node that the data reaches, from the ``sources``.

    Args:
        network: The network whose arcs are searched.
        sources: The nodes the data starts from, with the instant it is ready at each.
        crossing: When an arc can be crossed, see :data:`Crossing`.
        relays: Whether the data may leave a node, once it has arrived. A node that does not relay
            is reached but not expanded: the ground station, for instance.
        tolerance: A new arrival replaces the current one only if it is earlier by more than this.

    Returns:
        The reached nodes, with how each one is reached. The sources are included.
    """
    best: dict[NodeId, Reached] = {
        node: Reached(ready, None, None, None) for node, ready in sources.items()
    }
    fixed: set[NodeId] = set()
    counter = itertools.count()
    heap: list[tuple[float, int, NodeId]] = [
        (ready, next(counter), node) for node, ready in sources.items()
    ]
    heapq.heapify(heap)
    while heap:
        _, _, node = heapq.heappop(heap)
        if node in fixed:
            continue
        fixed.add(node)
        if not relays(node):
            continue
        ready = best[node].arrival
        for arc_index in network.outgoing[node]:
            arc = network.arcs[arc_index]
            if arc.target in fixed:
                continue
            crossed = crossing(arc_index, ready)
            if crossed is None:
                continue
            departure, arrival = crossed
            current = best.get(arc.target)
            if current is not None and current.arrival <= arrival + tolerance:
                continue
            best[arc.target] = Reached(arrival, node, arc_index, departure)
            heapq.heappush(heap, (arrival, next(counter), arc.target))
    return {node: best[node] for node in fixed}


LatestCrossing = Callable[[int, float], float | None]
"""``crossing(arc, due)``: the latest departure from the tail of the arc, for the data that must
be at its head by ``due``, or ``None`` if the arc cannot be crossed in time."""


def latest_departures(
    network: Network,
    dues: Mapping[NodeId, float],
    crossing: LatestCrossing,
    *,
    tolerance: float = 0.0,
) -> dict[NodeId, float]:
    """The latest instant the data can be on every node that it can still reach in time.

    The mirror of :func:`earliest_arrivals`: the search goes backward along the arcs, from the
    ``dues`` (the instants by which the data must be on each node) towards the sources of the
    arcs, and it keeps the latest instant found for each node. It is correct for the same reason:
    a departure is always earlier than the arrival it serves, because every arc has a positive
    latency, so a node is settled at its latest instant the first time it is popped.

    Args:
        network: The network whose arcs are searched backward.
        dues: The nodes the data must reach, with the instant by which it must be there.
        crossing: The latest departure on an arc, see :data:`LatestCrossing`.
        tolerance: A new departure replaces the current one only if it is later by more than this.

    Returns:
        The nodes that the search settled, each with its latest instant. The dues are included.
    """
    best: dict[NodeId, float] = dict(dues)
    fixed: set[NodeId] = set()
    counter = itertools.count()
    heap: list[tuple[float, int, NodeId]] = [
        (-due, next(counter), node) for node, due in dues.items()
    ]
    heapq.heapify(heap)
    while heap:
        _, _, node = heapq.heappop(heap)
        if node in fixed:
            continue
        fixed.add(node)
        due = best[node]
        for arc_index in network.incoming[node]:
            arc = network.arcs[arc_index]
            if arc.source in fixed:
                continue
            departure = crossing(arc_index, due)
            if departure is None:
                continue
            current = best.get(arc.source)
            if current is not None and current >= departure - tolerance:
                continue
            best[arc.source] = departure
            heapq.heappush(heap, (-departure, next(counter), arc.source))
    return {node: best[node] for node in fixed}


def path_to(reached: Mapping[NodeId, Reached], node: NodeId) -> tuple[int, ...]:
    """The arcs of the route that reaches ``node``, from its source, in order."""
    arcs: list[int] = []
    current = reached[node]
    while current.previous is not None and current.arc is not None:
        arcs.append(current.arc)
        current = reached[current.previous]
    return tuple(reversed(arcs))
