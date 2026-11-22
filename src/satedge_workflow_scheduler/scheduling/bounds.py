"""A priori time bounds of a request: EA, LA, ES and LC of the model.

The bounds come from a relaxation in which the request is alone in the network (section
"Limiti temporali a priori" of the model). They keep the constraints on the times of the
request itself: precedences, computing and visibility windows, store-and-forward, the time
limit and the plan release. They drop what ties different requests together, the task
overlap and the link contention, and the PLMM capacity. What is impossible for the request
alone is impossible in the full problem.

For the data item ``a`` (data index ``k``: 0 the raw data, ``k`` the output of task
``k - 1``) and the node ``v``:

- ``EA_{a,v}``, earliest availability: the earliest instant at which ``a`` can be entirely
  on ``v``;
- ``LA_{a,v}``, latest availability: the latest instant at which ``a`` can be on ``v`` and
  still reach its consumer in time.

For the task ``i`` (task index ``k``, input data ``k``, output data ``k + 1``) and the
satellite ``s``:

- ``ES_{i,s}``, earliest start: ``EF_{W(s)}(max(EA_{a,s}, b^plan_s), L̄_{i,s}) - L̄_{i,s}``;
- ``LC_{i,s}``, latest completion: ``LS_{W(s)}(LA_{a',s}, L̄_{i,s}) + L̄_{i,s}``.

An impossible bound is ``+inf`` for the earliest ones and ``-inf`` for the latest ones, as
``min {} = +inf`` and ``max {} = -inf`` in the model. The definitions are those of a
time-dependent shortest path, forward for EA and backward for LA, and they have a single
solution because every arc has a positive latency.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections.abc import Callable, Mapping

from satedge_workflow_scheduler.common.intervals import (
    TIME_TOLERANCE,
    TimeWindow,
    WindowSet,
)
from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.network import Network
from satedge_workflow_scheduler.domain.nodes import NodeId
from satedge_workflow_scheduler.domain.requests import Request
from satedge_workflow_scheduler.scheduling.windows import (
    arc_windows,
    earliest_finish,
    latest_start,
)

type NodeTimes = Mapping[NodeId, float]


class RequestBounds:
    """The a priori bounds of one request, alone in the network.

    Built by :func:`request_bounds`, which computes every bound at once because they depend
    on one another; this object only answers questions about them. Indices follow the
    pipeline: the task ``k`` uses the data ``k`` and produces the data ``k + 1``.
    """

    __slots__ = ("_compatible", "_ea", "_es", "_la", "_lc", "limit", "request_id")

    def __init__(
        self,
        request_id: str,
        limit: float,
        ea: tuple[NodeTimes, ...],
        la: tuple[NodeTimes, ...],
        es: tuple[NodeTimes, ...],
        lc: tuple[NodeTimes, ...],
        compatible: tuple[Mapping[NodeId, tuple[TimeWindow, ...]], ...],
    ) -> None:
        self.request_id = request_id
        """Identifier of the request."""
        self.limit = limit
        """Time limit ``τ̂_r`` of the delivery used for the bounds."""
        self._ea = ea
        self._la = la
        self._es = es
        self._lc = lc
        self._compatible = compatible

    def earliest_availability(self, data_index: int, node: NodeId) -> float:
        """``EA_{a,v}``: earliest instant at which data ``data_index`` is all on ``node``."""
        return self._ea[data_index][node]

    def latest_availability(self, data_index: int, node: NodeId) -> float:
        """``LA_{a,v}``: latest instant at which data ``data_index`` can be on ``node``."""
        return self._la[data_index][node]

    def earliest_start(self, task_index: int, satellite: NodeId) -> float:
        """``ES_{i,s}``: earliest start of task ``task_index`` on ``satellite``."""
        return self._es[task_index][satellite]

    def latest_completion(self, task_index: int, satellite: NodeId) -> float:
        """``LC_{i,s}``: latest completion of task ``task_index`` on ``satellite``."""
        return self._lc[task_index][satellite]

    def compatible_windows(
        self, task_index: int, satellite: NodeId
    ) -> tuple[TimeWindow, ...]:
        """Computing windows of ``satellite`` compatible with task ``task_index``.

        A window ``[t_w, T_w]`` is compatible if ``[max(ES, t_w), min(LC, T_w)]`` lasts at
        least ``L̄_{i,s}``: the task can run in it and still let the request be delivered.
        """
        return self._compatible[task_index][satellite]

    def can_run(self, task_index: int, satellite: NodeId) -> bool:
        """Whether task ``task_index`` has a compatible window on ``satellite``."""
        return bool(self._compatible[task_index][satellite])

    def executable_satellites(self, task_index: int) -> tuple[NodeId, ...]:
        """The satellites on which task ``task_index`` can run, in network order."""
        return tuple(s for s, ws in self._compatible[task_index].items() if ws)


def request_bounds(
    instance: ProblemInstance,
    request: Request,
    limit: float,
    eps: float = TIME_TOLERANCE,
) -> RequestBounds:
    """Compute the bounds of ``request``, alone in the network, for the time ``limit``.

    The limit is ``τ̂_r``: the deadline with hard deadlines, the horizon with soft ones
    (:func:`~satedge_workflow_scheduler.scheduling.deadline.delivery_limit`).
    """
    durations = tuple(
        {s.id: request.hpc_duration(k, s) for s in instance.network.satellites}
        for k in range(len(request.tasks))
    )
    ea = _earliest_availability(instance, request, durations, eps)
    la = _latest_availability(instance, request, durations, limit, eps)
    es, lc, compatible = _task_bounds(instance, durations, ea, la, eps)
    return RequestBounds(request.id, limit, ea, la, es, lc, compatible)


def _earliest_availability(
    instance: ProblemInstance,
    request: Request,
    durations: tuple[NodeTimes, ...],
    eps: float,
) -> tuple[NodeTimes, ...]:
    """``EA`` of every data item, forward along the pipeline.

    The raw data is on its origin from its release. Each later data item starts where its
    producer can end, after its input has arrived and the plan has reached the satellite,
    and then spreads over the network.
    """
    network = instance.network
    release = instance.release_of
    n = len(request.tasks)
    ea: list[NodeTimes] = []
    for k in range(n + 1):
        start = {v: math.inf for v in network.node_ids}
        if k == 0:
            start[request.origin] = request.release
        else:
            for s in network.satellites:
                ready = max(ea[k - 1][s.id], release(s.id))
                start[s.id] = _ef(s.hpc_windows, ready, durations[k - 1][s.id], eps)
        size = request.data_sizes[k]
        ea.append(_forward(network, start, size, release, to_ground=k == n, eps=eps))
    return tuple(ea)


def _latest_availability(
    instance: ProblemInstance,
    request: Request,
    durations: tuple[NodeTimes, ...],
    limit: float,
    eps: float,
) -> tuple[NodeTimes, ...]:
    """``LA`` of every data item, backward along the pipeline.

    The final output must be on the ground station by ``limit``. Each earlier data item
    must be where its consumer can still start in time, and before that on any node from
    which it can get there.
    """
    network = instance.network
    n = len(request.tasks)
    la: list[NodeTimes] = [{} for _ in range(n + 1)]
    for k in reversed(range(n + 1)):
        due = {v: -math.inf for v in network.node_ids}
        if k == n:
            due[network.ground_station.id] = limit
        else:
            for s in network.satellites:
                due[s.id] = _ls(s.hpc_windows, la[k + 1][s.id], durations[k][s.id], eps)
        la[k] = _backward(network, due, request.data_sizes[k], eps)
    return tuple(la)


def _task_bounds(
    instance: ProblemInstance,
    durations: tuple[NodeTimes, ...],
    ea: tuple[NodeTimes, ...],
    la: tuple[NodeTimes, ...],
    eps: float,
) -> tuple[
    tuple[NodeTimes, ...],
    tuple[NodeTimes, ...],
    tuple[Mapping[NodeId, tuple[TimeWindow, ...]], ...],
]:
    """``ES``, ``LC`` and the compatible windows of every task on every satellite."""
    es_all: list[NodeTimes] = []
    lc_all: list[NodeTimes] = []
    compatible_all: list[Mapping[NodeId, tuple[TimeWindow, ...]]] = []
    for k, task_durations in enumerate(durations):
        es: dict[NodeId, float] = {}
        lc: dict[NodeId, float] = {}
        compatible: dict[NodeId, tuple[TimeWindow, ...]] = {}
        for s in instance.network.satellites:
            d = task_durations[s.id]
            ready = max(ea[k][s.id], instance.release_of(s.id))
            es[s.id] = _ef(s.hpc_windows, ready, d, eps) - d
            lc[s.id] = _ls(s.hpc_windows, la[k + 1][s.id], d, eps) + d
            compatible[s.id] = tuple(
                w
                for w in s.hpc_windows
                if min(lc[s.id], w.end) - max(es[s.id], w.start) >= d - eps
            )
        es_all.append(es)
        lc_all.append(lc)
        compatible_all.append(compatible)
    return tuple(es_all), tuple(lc_all), tuple(compatible_all)


def _forward(
    network: Network,
    start: dict[NodeId, float],
    size: float,
    release: Callable[[NodeId], float],
    *,
    to_ground: bool,
    eps: float,
) -> dict[NodeId, float]:
    """Earliest availability on every node, from the instants in ``start``.

    A Dijkstra search on arrival instants: correct because leaving later never makes the
    data arrive earlier. The data leaves a node only after the plan has reached it. Only
    the final output (``to_ground``) may enter the ground station, which never relays.
    """
    ground = network.ground_station.id
    best = dict(start)
    counter = itertools.count()
    heap = [(t, next(counter), v) for v, t in best.items() if math.isfinite(t)]
    heapq.heapify(heap)
    while heap:
        t, _, node = heapq.heappop(heap)
        if t > best[node] or node == ground:
            continue
        for index in network.outgoing[node]:
            arc = network.arcs[index]
            if arc.target == ground and not to_ground:
                continue
            ready = max(t, release(node))
            arrival = _ef(
                arc_windows(arc.availability), ready, arc.traversal_time(size), eps
            )
            if arrival < best[arc.target]:
                best[arc.target] = arrival
                heapq.heappush(heap, (arrival, next(counter), arc.target))
    return best


def _backward(
    network: Network, due: dict[NodeId, float], size: float, eps: float
) -> dict[NodeId, float]:
    """Latest availability on every node, towards the instants in ``due``.

    The mirror of :func:`_forward`: the latest departure from the tail of each arc that
    still arrives at its head in time.
    """
    best = dict(due)
    counter = itertools.count()
    heap = [(-t, next(counter), v) for v, t in best.items() if math.isfinite(t)]
    heapq.heapify(heap)
    while heap:
        neg, _, node = heapq.heappop(heap)
        t = -neg
        if t < best[node]:
            continue
        for index in network.incoming[node]:
            arc = network.arcs[index]
            departure = _ls(
                arc_windows(arc.availability), t, arc.traversal_time(size), eps
            )
            if departure > best[arc.source]:
                best[arc.source] = departure
                heapq.heappush(heap, (-departure, next(counter), arc.source))
    return best


def _ef(windows: WindowSet | None, ready: float, length: float, eps: float) -> float:
    """``EF``, with ``+inf`` when the activity fits nowhere or cannot start at all."""
    if not math.isfinite(ready):
        return math.inf
    finish = earliest_finish(windows, ready, length, eps=eps)
    return math.inf if finish is None else finish


def _ls(windows: WindowSet | None, due: float, length: float, eps: float) -> float:
    """``LS``, with ``-inf`` when the activity fits nowhere or cannot end at all."""
    if not math.isfinite(due):
        return -math.inf
    start = latest_start(windows, due, length, eps=eps)
    return -math.inf if start is None else start
