"""Metrics of a solution, per request, per satellite and per arc, and its validity."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from satedge_workflow_scheduler.common.intervals import TimeWindow
from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.nodes import NodeId
from satedge_workflow_scheduler.domain.requests import Priority
from satedge_workflow_scheduler.domain.solution import (
    RequestOutcome,
    Solution,
    Transfer,
)
from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode, delivery_limit
from satedge_workflow_scheduler.scheduling.feasibility import Violation, check
from satedge_workflow_scheduler.scheduling.plmm_occupancy import (
    PlmmOccupancyInterval,
    load_profile,
    plmm_intervals,
)


@dataclass(frozen=True)
class RequestMetrics:
    """Outcome and timing of one request.

    Attributes:
        request_id: Identifier of the request.
        priority: Priority class of the request.
        committed: Whether the request comes from the committed calendar of the instance.
        accepted: Whether the request is served.
        rejection_reason: Why the request was rejected, if it was.
        release: Instant from which the raw data is available, in seconds.
        limit: Time limit of the delivery under the deadline mode, in seconds.
        delivery: Delivery instant, if the request is accepted.
        response: Delivery minus release, if the request is accepted.
        tardiness: Delivery beyond the limit, in seconds, if the request is accepted.
    """

    request_id: str
    priority: Priority
    committed: bool
    accepted: bool
    rejection_reason: str | None
    release: float
    limit: float
    delivery: float | None
    response: float | None
    tardiness: float | None


@dataclass(frozen=True)
class PlmmMetrics:
    """Memory load of the PLMM of one satellite.

    Attributes:
        satellite: The satellite.
        capacity: Net PLMM capacity, in Gb.
        peak: Largest load over time, in Gb; zero if no data is ever stored.
        peak_time: Start of the first segment where the peak is reached, or ``None``.
    """

    satellite: NodeId
    capacity: float
    peak: float
    peak_time: float | None


@dataclass(frozen=True)
class HpcMetrics:
    """Use of the computing windows of one satellite.

    Attributes:
        satellite: The satellite.
        available: Seconds of computing windows before the horizon.
        busy: Seconds the satellite spends executing tasks.
    """

    satellite: NodeId
    available: float
    busy: float

    @property
    def utilisation(self) -> float:
        """Busy time over available time, or zero if the satellite has no window."""
        return self.busy / self.available if self.available > 0 else 0.0


@dataclass(frozen=True)
class ArcMetrics:
    """Use of one arc of the network.

    Attributes:
        source: Tail node of the arc.
        target: Head node of the arc.
        busy: Seconds of transmission on the arc.
        volume: Gb transmitted on the arc.
    """

    source: NodeId
    target: NodeId
    busy: float
    volume: float


@dataclass(frozen=True)
class SolutionMetrics:
    """All the indicators of one solution of an instance.

    Attributes:
        requests: One entry per request of the instance, in instance order.
        plmm: One entry per satellite.
        hpc: One entry per satellite.
        arcs: One entry per arc that carries data.
        violations: The rules the solution breaks, as the checker finds them; empty if valid.
    """

    requests: tuple[RequestMetrics, ...]
    plmm: tuple[PlmmMetrics, ...]
    hpc: tuple[HpcMetrics, ...]
    arcs: tuple[ArcMetrics, ...]
    violations: tuple[Violation, ...]

    @property
    def accepted_ids(self) -> tuple[str, ...]:
        """Identifiers of the accepted requests, in instance order."""
        return tuple(r.request_id for r in self.requests if r.accepted)

    def count(self, priority: Priority, *, accepted: bool) -> int:
        """Number of requests of ``priority`` that are accepted, or rejected if not."""
        return sum(
            1
            for r in self.requests
            if r.priority is priority and r.accepted is accepted
        )

    @property
    def total_tardiness(self) -> float:
        """Sum of the tardiness of the accepted requests, in seconds."""
        return sum(r.tardiness for r in self.requests if r.tardiness is not None)

    @property
    def mean_response(self) -> float | None:
        """Mean response time of the accepted requests, or ``None`` if none is accepted."""
        responses = [r.response for r in self.requests if r.response is not None]
        return sum(responses) / len(responses) if responses else None


def analyze(
    instance: ProblemInstance, solution: Solution, mode: DeadlineMode
) -> SolutionMetrics:
    """The indicators of ``solution`` for ``instance`` under the deadline ``mode``.

    The occupancy of the PLMM and the use of the arcs are computed on the whole calendar,
    committed requests included, as the checker computes them.
    """
    schedule = solution.schedule
    outcomes = {o.request_id: o for o in solution.outcomes}
    horizon = instance.horizon

    requests = tuple(
        _request_metrics(instance, mode, outcomes[r.id], r.id)
        for r in instance.requests
    )

    occupancies = plmm_intervals(instance, schedule.executions, schedule.transfers)
    plmm = tuple(
        _plmm_metrics(
            satellite.id, satellite.plmm_capacity, occupancies.get(satellite.id, [])
        )
        for satellite in instance.network.satellites
    )

    busy_hpc: dict[NodeId, float] = {}
    for execution in schedule.executions:
        busy_hpc[execution.satellite] = (
            busy_hpc.get(execution.satellite, 0.0) + execution.end - execution.start
        )
    hpc = tuple(
        HpcMetrics(
            satellite.id,
            _window_time(satellite.hpc_windows.windows, horizon),
            busy_hpc.get(satellite.id, 0.0),
        )
        for satellite in instance.network.satellites
    )

    arcs = _arc_metrics(instance, schedule.transfers)
    return SolutionMetrics(
        requests=requests,
        plmm=plmm,
        hpc=hpc,
        arcs=arcs,
        violations=check(instance, solution, mode),
    )


def _request_metrics(
    instance: ProblemInstance,
    mode: DeadlineMode,
    outcome: RequestOutcome,
    request_id: str,
) -> RequestMetrics:
    request = instance.request(request_id)
    limit = delivery_limit(request, mode, instance.horizon)
    accepted = outcome.accepted
    delivery = outcome.delivery_time
    response = None
    tardiness = None
    if accepted and delivery is not None:
        response = delivery - request.release
        tardiness = max(0.0, delivery - limit)
    return RequestMetrics(
        request_id=request_id,
        priority=request.priority,
        committed=instance.is_committed(request_id),
        accepted=accepted,
        rejection_reason=outcome.rejection_reason,
        release=request.release,
        limit=limit,
        delivery=delivery,
        response=response,
        tardiness=tardiness,
    )


def _plmm_metrics(
    satellite: NodeId, capacity: float, items: Iterable[PlmmOccupancyInterval]
) -> PlmmMetrics:
    segments = load_profile(items)
    if not segments:
        return PlmmMetrics(satellite, capacity, 0.0, None)
    peak = max(segments, key=lambda s: s.load)
    return PlmmMetrics(satellite, capacity, peak.load, peak.start)


def _window_time(windows: Iterable[TimeWindow], horizon: float) -> float:
    """Seconds of ``windows`` that fall before ``horizon``."""
    total = 0.0
    for window in windows:
        total += max(0.0, min(window.end, horizon) - window.start)
    return total


def _arc_metrics(
    instance: ProblemInstance, transfers: Iterable[Transfer]
) -> tuple[ArcMetrics, ...]:
    busy: dict[tuple[NodeId, NodeId], float] = {}
    volume: dict[tuple[NodeId, NodeId], float] = {}
    for t in transfers:
        key = (t.source, t.target)
        size = instance.request(t.request_id).data_sizes[t.data_index]
        busy[key] = busy.get(key, 0.0) + t.end - t.start
        volume[key] = volume.get(key, 0.0) + size
    return tuple(
        ArcMetrics(source, target, busy[(source, target)], volume[(source, target)])
        for source, target in sorted(busy)
    )
