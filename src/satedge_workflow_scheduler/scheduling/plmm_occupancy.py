"""PLMM occupancy of the data of a schedule, and the load profile of the PLMM of each satellite."""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from satedge_workflow_scheduler.common.intervals import TIME_TOLERANCE
from satedge_workflow_scheduler.domain.errors import InstanceError
from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.network import Network
from satedge_workflow_scheduler.domain.nodes import NodeId
from satedge_workflow_scheduler.domain.requests import Request
from satedge_workflow_scheduler.domain.solution import TaskExecution, Transfer


@dataclass(frozen=True, slots=True)
class PlmmOccupancyInterval:
    """Interval during which one data item occupies the PLMM of a satellite.

    Attributes:
        request_id: Identifier of the request the data belongs to.
        data_index: Index of the data item: 0 for the raw data, ``k`` for the output of task
            ``k - 1``.
        size: Size in Gb of the data item.
        start: Start of the occupancy in seconds.
        end: End of the occupancy in seconds. The interval is half-open, ``[start, end)``.
    """

    request_id: str
    data_index: int
    size: float
    start: float
    end: float


def plmm_intervals(
    instance: ProblemInstance,
    executions: Iterable[TaskExecution],
    transfers: Iterable[Transfer],
) -> dict[NodeId, list[PlmmOccupancyInterval]]:
    """Return the PLMM occupancies of the accepted requests, keyed by satellite.

    For each data item of a request (raw data, then the output of each task) the function
    follows its chain of transfers and records where the data sits and for how long:
    from its creation (or from the arrival of its first bit) until the end of the transfer
    that sends it away, or until the end of the task that uses it. The ground station never
    holds occupancies. Empty intervals are dropped.

    Raises:
        InstanceError: If a data item does not reach the satellite of the task that uses it.
            The final output is not checked here: its chain is not required to end at the
            ground station.
    """
    executions_by_request: dict[str, dict[int, TaskExecution]] = defaultdict(dict)
    for execution in executions:
        executions_by_request[execution.request_id][execution.task_index] = execution

    transfers_by_item: dict[tuple[str, int], list[Transfer]] = defaultdict(list)
    for transfer in transfers:
        transfers_by_item[(transfer.request_id, transfer.data_index)].append(transfer)

    network = instance.network
    result: dict[NodeId, list[PlmmOccupancyInterval]] = defaultdict(list)

    for request in instance.requests:
        tasks_done = executions_by_request.get(request.id)
        if not tasks_done:
            continue

        for data_index in range(len(request.tasks) + 1):
            size = request.data_sizes[data_index]
            node, start = _creation(request, data_index, tasks_done, network)
            chain = sorted(
                transfers_by_item.get((request.id, data_index), []),
                key=lambda t: t.start,
            )
            for hop in chain:
                _add(result, node, request.id, data_index, size, start, hop.end)
                arc = network.arcs[network.arc_index[(hop.source, hop.target)]]
                node = hop.target
                start = hop.start + arc.latency

            # data item k is the input of task k; the last one is the final output
            if data_index < len(request.tasks):
                if node != tasks_done[data_index].satellite:
                    raise InstanceError(
                        f"request {request.id}: data {data_index} does not reach "
                        f"the satellite of task {data_index} ({node!r} instead)"
                    )
                _add(
                    result,
                    node,
                    request.id,
                    data_index,
                    size,
                    start,
                    tasks_done[data_index].end,
                )

    return dict(result)


def _creation(
    request: Request,
    data_index: int,
    tasks_done: dict[int, TaskExecution],
    network: Network,
) -> tuple[NodeId, float]:
    """Node and instant at which data ``data_index`` is created in the PLMM."""
    if data_index == 0:
        return request.origin, request.release
    producer = tasks_done[data_index - 1]
    satellite = network.satellite(producer.satellite)
    size = request.data_sizes[data_index]
    return producer.satellite, producer.end - satellite.hpc_io_time(size)


def _add(
    result: dict[NodeId, list[PlmmOccupancyInterval]],
    node: NodeId,
    request_id: str,
    data_index: int,
    size: float,
    start: float,
    end: float,
) -> None:
    if size == 0 or end <= start + TIME_TOLERANCE:
        return
    result[node].append(PlmmOccupancyInterval(request_id, data_index, size, start, end))


@dataclass(frozen=True, slots=True)
class LoadSegment:
    """A time segment over which the PLMM load of a satellite is constant.

    Attributes:
        start: Start of the segment in seconds.
        end: End of the segment in seconds. Half-open, ``[start, end)``.
        load: Total size in Gb of the data present during the segment.
        items: The data present during the segment.
    """

    start: float
    end: float
    load: float
    items: tuple[PlmmOccupancyInterval, ...]


def load_profile(
    items: Iterable[PlmmOccupancyInterval], eps: float = TIME_TOLERANCE
) -> tuple[LoadSegment, ...]:
    """Return the PLMM load as consecutive segments, in time order.

    The profile changes only at the instants where an item starts or ends. Instants closer
    than ``eps`` are the same instant. Segments with no data are omitted.
    """
    items = tuple(items)
    times = sorted(t for item in items for t in (item.start, item.end))
    instants: list[float] = []
    for t in times:
        if not instants or t - instants[-1] > eps:
            instants.append(t)

    segments: list[LoadSegment] = []
    for start, end in itertools.pairwise(instants):
        present = tuple(
            item
            for item in items
            if item.start <= start + eps and item.end >= end - eps
        )
        if present:
            load = sum(item.size for item in present)
            segments.append(LoadSegment(start, end, load, present))
    return tuple(segments)


def overloads(
    profile: Iterable[LoadSegment], capacity: float, eps: float = TIME_TOLERANCE
) -> tuple[LoadSegment, ...]:
    """Return the segments of ``profile`` whose load exceeds ``capacity``, up to ``eps``."""
    return tuple(segment for segment in profile if segment.load > capacity + eps)


def within_capacity(
    items: Iterable[PlmmOccupancyInterval],
    capacity: float,
    eps: float = TIME_TOLERANCE,
) -> bool:
    """Whether the PLMM load of ``items`` stays within ``capacity``, up to ``eps``."""
    return not overloads(load_profile(items, eps), capacity, eps)
