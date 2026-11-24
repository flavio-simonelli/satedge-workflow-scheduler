"""What the figures draw, as plain values in seconds: no matplotlib here."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.network import Arc, WindowedAvailability
from satedge_workflow_scheduler.domain.nodes import NodeId
from satedge_workflow_scheduler.domain.requests import Priority
from satedge_workflow_scheduler.domain.solution import Solution, TaskExecution, Transfer
from satedge_workflow_scheduler.scheduling.plmm_occupancy import (
    load_profile,
    plmm_intervals,
)


@dataclass(frozen=True)
class Span:
    """A time interval, in seconds."""

    start: float
    end: float


@dataclass(frozen=True)
class TaskBar:
    """A task run on a satellite."""

    request_id: str
    task_index: int
    start: float
    end: float
    urgent: bool


@dataclass(frozen=True)
class SatelliteHpc:
    """The computing windows of a satellite, and the tasks it runs."""

    satellite: NodeId
    windows: tuple[Span, ...]
    tasks: tuple[TaskBar, ...]


@dataclass(frozen=True)
class TransferBar:
    """A transmission of one data item on an arc; the data arrives after the latency."""

    request_id: str
    data_index: int
    start: float
    end: float
    arrival: float


@dataclass(frozen=True)
class ArcTraffic:
    """The transmissions on an arc, and the windows in which it is available.

    ``windows`` is ``None`` for an arc that is always available.
    """

    source: NodeId
    target: NodeId
    windows: tuple[Span, ...] | None
    transfers: tuple[TransferBar, ...]


@dataclass(frozen=True)
class MemorySegment:
    """A stretch of time over which the PLMM load of a satellite is constant."""

    start: float
    end: float
    load: float


@dataclass(frozen=True)
class MemoryCurve:
    """The PLMM load of a satellite against its capacity."""

    satellite: NodeId
    capacity: float
    segments: tuple[MemorySegment, ...]
    peak: float


@dataclass(frozen=True)
class Timeline:
    """Everything the figures of one solution show.

    Attributes:
        horizon: The scheduling horizon, in seconds.
        requests: Identifiers of the requests, in instance order (the colour order).
        urgent: Identifiers of the urgent requests.
        hpc: One entry per satellite.
        arcs: One entry per arc that carries data, in a fixed order.
        memory: One entry per satellite.
    """

    horizon: float
    requests: tuple[str, ...]
    urgent: frozenset[str]
    hpc: tuple[SatelliteHpc, ...]
    arcs: tuple[ArcTraffic, ...]
    memory: tuple[MemoryCurve, ...]


def timeline(instance: ProblemInstance, solution: Solution) -> Timeline:
    """The data of the figures of ``solution`` for ``instance``."""
    urgent = frozenset(r.id for r in instance.requests if r.priority is Priority.URGENT)
    schedule = solution.schedule
    return Timeline(
        horizon=instance.horizon,
        requests=tuple(r.id for r in instance.requests),
        urgent=urgent,
        hpc=_hpc(instance, schedule.executions, urgent),
        arcs=_arcs(instance, schedule.transfers),
        memory=_memory(instance, schedule.executions, schedule.transfers),
    )


def _hpc(
    instance: ProblemInstance,
    executions: Sequence[TaskExecution],
    urgent: frozenset[str],
) -> tuple[SatelliteHpc, ...]:
    rows = []
    for satellite in instance.network.satellites:
        windows = tuple(Span(w.start, w.end) for w in satellite.hpc_windows.windows)
        tasks = tuple(
            TaskBar(e.request_id, e.task_index, e.start, e.end, e.request_id in urgent)
            for e in executions
            if e.satellite == satellite.id
        )
        if tasks:  # only the satellites that compute something are drawn
            rows.append(SatelliteHpc(satellite.id, windows, tasks))
    return tuple(rows)


def _arcs(
    instance: ProblemInstance, transfers: Sequence[Transfer]
) -> tuple[ArcTraffic, ...]:
    by_pair: dict[tuple[NodeId, NodeId], list[TransferBar]] = {}
    arcs: dict[tuple[NodeId, NodeId], Arc] = {
        (a.source, a.target): a for a in instance.network.arcs
    }
    for t in transfers:
        arc = arcs[(t.source, t.target)]
        by_pair.setdefault((t.source, t.target), []).append(
            TransferBar(t.request_id, t.data_index, t.start, t.end, t.end + arc.latency)
        )
    rows = []
    for (source, target), bars in sorted(by_pair.items()):
        availability = arcs[(source, target)].availability
        windows = None
        if isinstance(availability, WindowedAvailability):
            windows = tuple(Span(w.start, w.end) for w in availability.windows.windows)
        rows.append(
            ArcTraffic(
                source, target, windows, tuple(sorted(bars, key=lambda b: b.start))
            )
        )
    return tuple(rows)


def _memory(
    instance: ProblemInstance,
    executions: Sequence[TaskExecution],
    transfers: Sequence[Transfer],
) -> tuple[MemoryCurve, ...]:
    occupancies = plmm_intervals(instance, executions, transfers)
    rows = []
    for satellite in instance.network.satellites:
        profile = load_profile(occupancies.get(satellite.id, []))
        segments = tuple(MemorySegment(s.start, s.end, s.load) for s in profile)
        peak = max((s.load for s in segments), default=0.0)
        if peak > 0:  # only the satellites whose memory holds data are drawn
            rows.append(
                MemoryCurve(satellite.id, satellite.plmm_capacity, segments, peak)
            )
    return tuple(rows)


@dataclass(frozen=True)
class RouteTask:
    """A task of one request, and the satellite that runs it."""

    satellite: NodeId
    task_index: int
    start: float
    end: float


@dataclass(frozen=True)
class Hop:
    """One transmission of a request, numbered in the order it happens."""

    number: int
    source: NodeId
    target: NodeId
    data_index: int
    start: float
    end: float
    arrival: float


@dataclass(frozen=True)
class RoutePlan:
    """Where one request runs, and the path of each of its data items.

    Attributes:
        request_id: The request.
        tasks: Its tasks in pipeline order, each with the satellite that runs it.
        hops: Its transmissions in time order.
    """

    request_id: str
    tasks: tuple[RouteTask, ...]
    hops: tuple[Hop, ...]


def route_of(data: Timeline, request_id: str) -> RoutePlan:
    """The tasks and the transmissions of ``request_id`` in ``data``.

    Raises:
        ValueError: If the request is not in the timeline.
    """
    if request_id not in data.requests:
        raise ValueError(f"unknown request {request_id!r}")
    tasks = sorted(
        (
            RouteTask(satellite.satellite, task.task_index, task.start, task.end)
            for satellite in data.hpc
            for task in satellite.tasks
            if task.request_id == request_id
        ),
        key=lambda t: t.task_index,
    )
    bars = sorted(
        (
            (arc.source, arc.target, bar)
            for arc in data.arcs
            for bar in arc.transfers
            if bar.request_id == request_id
        ),
        key=lambda item: item[2].start,
    )
    hops = tuple(
        Hop(number, source, target, bar.data_index, bar.start, bar.end, bar.arrival)
        for number, (source, target, bar) in enumerate(bars, start=1)
    )
    return RoutePlan(request_id, tuple(tasks), hops)
