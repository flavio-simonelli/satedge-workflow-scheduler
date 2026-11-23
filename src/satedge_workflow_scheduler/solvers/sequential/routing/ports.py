"""Routing of a data item around the reservations: the common types and the interface.

The data item is on a node, where it has occupied the PLMM since ``held_since``, and can leave
from ``ready``. A router finds routes to the nodes it can reach, store-and-forward: a node
sends the data on only once it has all of it. Each transmission waits for a visibility window
of its arc, for the arc to be free and for the plan to reach the sending node.

On every node it leaves, the data occupies the PLMM from the start of its stay to the end of the
transmission that sends it away. A route is valid only if the PLMM of every node it leaves has
room for the whole stay.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.nodes import NodeId
from satedge_workflow_scheduler.solvers.sequential.reservations import Reservations


@dataclass(frozen=True, slots=True)
class Hop:
    """One transmission of a route.

    Attributes:
        source: Tail node of the arc.
        target: Head node of the arc.
        start: Start of the transmission.
        end: End of the transmission (``start + transmission time``): the arc is busy until then.
        arrival: Arrival of the last bit (``end + latency``, the traversal ends there).
    """

    source: NodeId
    target: NodeId
    start: float
    end: float
    arrival: float


@dataclass(frozen=True, slots=True)
class Stay:
    """The data in the PLMM of a node it then leaves, during ``[start, end)``."""

    node: NodeId
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class Route:
    """How the data item gets from its node to ``target``.

    Attributes:
        target: The node the route ends on.
        hops: The transmissions, in order. Empty if the data is already on ``target``.
        stays: The PLMM occupancies the route closes, one per node it leaves.
        arrival: When the data is entirely on ``target``.
        held_since: When the data starts occupying the PLMM of ``target``.
        hold_until: The data must leave ``target``, or be consumed there, by this instant: the end
            of the free interval of the PLMM it is in. ``+inf`` when the router does not know.
    """

    target: NodeId
    hops: tuple[Hop, ...]
    stays: tuple[Stay, ...]
    arrival: float
    held_since: float
    hold_until: float


@dataclass(frozen=True, slots=True)
class RouteQuery:
    """What to route: a data item, where it is and since when.

    Attributes:
        source: The node the data is on.
        held_since: Since when the data has occupied the PLMM of ``source``.
        ready: The data cannot leave ``source`` earlier.
        size: Size of the data in Gb.
        limit: Routes that arrive later are discarded.
        to_ground: Whether the data may enter the ground station: only the final output can.
            The ground station never relays.
    """

    source: NodeId
    held_since: float
    ready: float
    size: float
    limit: float
    to_ground: bool = False


class Router(Protocol):
    """Finds the routes of a data item around the reservations.

    A router keeps no state between calls: the instance and the reservations come with each
    query, so the same router serves every request.
    """

    def routes_from(
        self, instance: ProblemInstance, reservations: Reservations, query: RouteQuery
    ) -> Mapping[NodeId, tuple[Route, ...]]:
        """The routes of ``query`` to the nodes it reaches by ``query.limit``.

        Returns:
            For each node reached, its routes in order of arrival. ``query.source`` is always
            there, with an empty route first.
        """
        ...
