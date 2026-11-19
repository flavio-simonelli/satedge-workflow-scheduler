"""PLMM occupancy of the data of a schedule, and the peak load of a set of occupancies."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from satedge_workflow_scheduler.common.errors import SolverError
from satedge_workflow_scheduler.common.intervals import TIME_TOLERANCE
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
        position: Index of the data item: 0 for the raw data, ``k`` for the output of task
            ``k - 1``.
        size: Size in Gb of the data item.
        start: Start of the occupancy in seconds.
        end: End of the occupancy in seconds. The interval is half-open, ``[start, end)``.
    """

    request_id: str
    position: int
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
        SolverError: If a data item that must reach a task or the ground does not have a
            complete chain of transfers.
    """
    executions_by_request: dict[str, dict[int, TaskExecution]] = defaultdict(dict)
    for execution in executions:
        executions_by_request[execution.request_id][execution.position] = execution

    transfers_by_item: dict[tuple[str, int], list[Transfer]] = defaultdict(list)
    for transfer in transfers:
        transfers_by_item[(transfer.request_id, transfer.position)].append(transfer)

    network = instance.network
    result: dict[NodeId, list[PlmmOccupancyInterval]] = defaultdict(list)

    for request in instance.requests:
        tasks_done = executions_by_request.get(request.id)
        if not tasks_done:
            continue

        for position in range(len(request.tasks) + 1):
            size = request.data_sizes[position]
            node, start = _creation(request, position, tasks_done, network)
            chain = sorted(
                transfers_by_item.get((request.id, position), []),
                key=lambda t: t.start,
            )
            for hop in chain:
                _add(result, node, request.id, position, size, start, hop.end)
                arc = network.arcs[network.arc_index[(hop.source, hop.target)]]
                node = hop.target
                start = hop.start + arc.latency

            if position < len(request.tasks):
                if node != tasks_done[position].satellite:
                    raise SolverError(
                        f"request {request.id}: data {position} does not reach "
                        f"the satellite of task {position} ({node!r} instead)"
                    )
                _add(
                    result,
                    node,
                    request.id,
                    position,
                    size,
                    start,
                    tasks_done[position].end,
                )

    return dict(result)


def _creation(
    request: Request,
    position: int,
    tasks_done: dict[int, TaskExecution],
    network: Network,
) -> tuple[NodeId, float]:
    """Node and instant at which data ``position`` is created in the PLMM."""
    if position == 0:
        return request.origin, request.release
    producer = tasks_done[position - 1]
    satellite = network.satellite(producer.satellite)
    size = request.data_sizes[position]
    return producer.satellite, producer.end - satellite.write_time(size)


def _add(
    result: dict[NodeId, list[PlmmOccupancyInterval]],
    node: NodeId,
    request_id: str,
    position: int,
    size: float,
    start: float,
    end: float,
) -> None:
    if size == 0 or end <= start + TIME_TOLERANCE:
        return
    result[node].append(PlmmOccupancyInterval(request_id, position, size, start, end))


def peak_load(
    items: Iterable[PlmmOccupancyInterval], eps: float = TIME_TOLERANCE
) -> float:
    """Return the maximum total size of the data present at the same time.

    Intervals are half-open. Events closer than ``eps`` are simultaneous: releases are
    applied before arrivals, so data leaving at ``t`` and data entering at ``t`` do not
    add up.
    """
    events: list[tuple[float, float]] = []
    for item in items:
        events.append((item.start, item.size))
        events.append((item.end, -item.size))
    if not events:
        return 0.0

    events.sort(key=lambda e: e[0])
    best = 0.0
    load = 0.0
    group: list[tuple[float, float]] = []
    for event in events:
        if group and event[0] - group[-1][0] > eps:
            load, best = _flush(group, load, best)
            group = []
        group.append(event)
    _, best = _flush(group, load, best)
    return best


def _flush(
    group: Sequence[tuple[float, float]], load: float, best: float
) -> tuple[float, float]:
    """Apply one group of simultaneous events: releases first, then arrivals."""
    for _, delta in group:
        if delta < 0:
            load += delta
    for _, delta in group:
        if delta > 0:
            load += delta
            best = max(best, load)
    return load, best


def fits_capacity(
    items: Iterable[PlmmOccupancyInterval], capacity: float, eps: float = TIME_TOLERANCE
) -> bool:
    """Whether the peak load of ``items`` stays within ``capacity``, up to ``eps``."""
    return peak_load(items, eps) <= capacity + eps
