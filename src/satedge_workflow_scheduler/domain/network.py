"""Network of the constellation: orbital planes, ground station and directed arcs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from satedge_workflow_scheduler.common.errors import InstanceError
from satedge_workflow_scheduler.common.intervals import TimeWindow, WindowSet
from satedge_workflow_scheduler.domain.nodes import GroundStation, NodeId, Satellite


@dataclass(frozen=True)
class Plane:
    """An orbital plane: a group of satellites with one communication satellite (COM).

    A single satellite forms a plane of its own, and it is its own COM.

    Attributes:
        id: Plane identifier, unique in the network.
        satellites: The satellites of the plane, in declaration order.
        com: Identifier of the COM, which must be one of ``satellites``.
    """

    id: int
    satellites: tuple[Satellite, ...]
    com: NodeId

    def __post_init__(self) -> None:
        if not self.satellites:
            raise InstanceError(f"plane {self.id}: needs at least one satellite")
        ids = [s.id for s in self.satellites]
        if len(set(ids)) != len(ids):
            raise InstanceError(f"plane {self.id}: duplicate satellite ids in {ids}")
        if self.com not in ids:
            raise InstanceError(
                f"plane {self.id}: COM {self.com!r} is not one of its satellites"
            )


@dataclass(frozen=True)
class AlwaysAvailable:
    """An arc always available, e.g. an intra-plane optical link."""


@dataclass(frozen=True)
class WindowedAvailability:
    """An arc available only in its visibility windows (contact plan).

    Attributes:
        windows: Visibility windows, sorted and disjoint.
    """

    windows: tuple[TimeWindow, ...]

    def __post_init__(self) -> None:
        try:
            WindowSet(self.windows)
        except ValueError as error:
            raise InstanceError(f"visibility windows: {error}") from error


type Availability = AlwaysAvailable | WindowedAvailability


@dataclass(frozen=True)
class Arc:
    """A directed arc of the network (one direction of a full-duplex link).

    Attributes:
        source: Tail node.
        target: Head node.
        capacity: Net capacity in Gbps (> 0).
        latency: Propagation latency in seconds, ARQ/FEC included (> 0).
        availability: When the arc can carry data.
    """

    source: NodeId
    target: NodeId
    capacity: float
    latency: float
    availability: Availability

    def __post_init__(self) -> None:
        if self.source == self.target:
            raise InstanceError(f"arc {self.source}->{self.target} is a self loop")
        if self.capacity <= 0:
            raise InstanceError(f"arc {self.name}: capacity must be > 0")
        if self.latency <= 0:
            raise InstanceError(f"arc {self.name}: latency must be > 0")

    @property
    def name(self) -> str:
        """Readable arc name ``source->target``."""
        return f"{self.source}->{self.target}"

    @property
    def is_windowed(self) -> bool:
        """Whether the arc is available only in its visibility windows."""
        return isinstance(self.availability, WindowedAvailability)

    def transmission_time(self, size: float) -> float:
        """Time to transmit ``size`` Gb on this arc, in seconds."""
        return size / self.capacity

    def traversal_time(self, size: float) -> float:
        """Transmission plus propagation time, in seconds."""
        return self.transmission_time(size) + self.latency


@dataclass(frozen=True)
class Network:
    """The time-varying network: orbital planes, ground station and arcs.

    The satellites are the ones of the planes. The ground station has no outgoing arcs:
    it never relays data.

    Attributes:
        planes: The orbital planes, with unique ids and disjoint satellites.
        ground_station: The ground station node.
        arcs: The directed arcs, with unique (source, target) pairs. Two opposite arcs are
            the two directions of one full-duplex link and have the same availability.
    """

    planes: tuple[Plane, ...]
    ground_station: GroundStation
    arcs: tuple[Arc, ...]

    def __post_init__(self) -> None:
        if not self.planes:
            raise InstanceError("the network needs at least one plane")
        plane_ids = [p.id for p in self.planes]
        if len(set(plane_ids)) != len(plane_ids):
            raise InstanceError(f"duplicate plane ids in {plane_ids}")
        if self.ground_station.id in self.satellite_ids:
            raise InstanceError(
                f"ground station id {self.ground_station.id!r} is also a satellite"
            )
        if len(set(self.satellite_ids)) != len(self.satellite_ids):
            raise InstanceError("a satellite belongs to more than one plane")

        nodes = set(self.node_ids)
        seen: set[tuple[NodeId, NodeId]] = set()
        for arc in self.arcs:
            for endpoint in (arc.source, arc.target):
                if endpoint not in nodes:
                    raise InstanceError(f"arc {arc.name}: unknown node {endpoint!r}")
            if arc.source == self.ground_station.id:
                raise InstanceError(
                    f"arc {arc.name}: the ground station cannot transmit"
                )
            if (arc.source, arc.target) in seen:
                raise InstanceError(f"duplicate arc {arc.name}")
            seen.add((arc.source, arc.target))

        # a full-duplex link is two opposite arcs, and visibility is symmetric
        by_pair = {(arc.source, arc.target): arc for arc in self.arcs}
        for (source, target), arc in by_pair.items():
            reverse = by_pair.get((target, source))
            if reverse is not None and reverse.availability != arc.availability:
                raise InstanceError(
                    f"arcs {arc.name} and {reverse.name} are the two directions of one "
                    "link and must have the same availability"
                )

    @cached_property
    def satellites(self) -> tuple[Satellite, ...]:
        """All satellites of the network, plane by plane."""
        return tuple(s for plane in self.planes for s in plane.satellites)

    @property
    def satellite_ids(self) -> tuple[NodeId, ...]:
        """Satellite identifiers, in declaration order."""
        return tuple(s.id for s in self.satellites)

    @property
    def node_ids(self) -> tuple[NodeId, ...]:
        """All node identifiers: satellites and the ground station."""
        return (*self.satellite_ids, self.ground_station.id)

    @cached_property
    def _satellite_by_id(self) -> dict[NodeId, Satellite]:
        return {s.id: s for s in self.satellites}

    def satellite(self, node: NodeId) -> Satellite:
        """Return the satellite with identifier ``node``."""
        try:
            return self._satellite_by_id[node]
        except KeyError:
            raise InstanceError(f"unknown satellite {node!r}") from None

    def is_satellite(self, node: NodeId) -> bool:
        """Whether ``node`` is a satellite of the network."""
        return node in self._satellite_by_id

    @cached_property
    def outgoing(self) -> dict[NodeId, tuple[int, ...]]:
        """Indices in :attr:`arcs` of the arcs leaving each node."""
        result: dict[NodeId, list[int]] = {v: [] for v in self.node_ids}
        for k, arc in enumerate(self.arcs):
            result[arc.source].append(k)
        return {v: tuple(ks) for v, ks in result.items()}

    @cached_property
    def incoming(self) -> dict[NodeId, tuple[int, ...]]:
        """Indices in :attr:`arcs` of the arcs entering each node."""
        result: dict[NodeId, list[int]] = {v: [] for v in self.node_ids}
        for k, arc in enumerate(self.arcs):
            result[arc.target].append(k)
        return {v: tuple(ks) for v, ks in result.items()}

    @cached_property
    def arc_index(self) -> dict[tuple[NodeId, NodeId], int]:
        """Map from ``(source, target)`` to the index in :attr:`arcs`."""
        return {(arc.source, arc.target): k for k, arc in enumerate(self.arcs)}
