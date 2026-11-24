"""The intervals of the reductions: where a task, a transmission or a stay in a PLMM can be.

Section "Intervalli" of the document. From the a priori bounds of a request alone in the network
(:meth:`ModelData.bounds`), every task, transmission and stay in a PLMM of the request gets the
interval that contains it in every feasible solution in which the request is accepted
(proposition "limiti" of the document):

- the task ``i`` in the window ``w`` of ``s`` runs in
  ``I^task_{i,s,w} = [max(ES_{i,s}, t_w), min(LC_{i,s}, T_w)]``, on the windows *compatible*
  with ``i``, those in which the interval lasts at least ``L̄_{i,s}``. The bounds give them
  (:meth:`RequestBounds.compatible_indices`);
- the data ``a`` on the arc ``e = (u, v)`` is transmitted, from the start to the arrival of the
  last bit, in ``I^link_{a,e} = [max(EA_{a,u}, b^plan_u), LA_{a,v}]``. A visibility window is
  compatible with ``a`` if it shares at least ``D_{a,e} + λ_e`` with the interval, and an arc
  always available is the single window ``[0, +∞)``;
- the data ``a`` occupies the PLMM of ``v`` within
  ``I^PLMM_{a,v} = [EA_{a,v} − enter_{a,v}, LA_{a,v} + exit_{a,v}]``. ``enter`` is the longest
  of the write ``w_{a,v}``, if ``a`` can be produced on ``v``, and of the transmissions into
  ``v``; ``exit`` is the longest of the consumer ``L̄_{j,v}``, if it can run on ``v``, and of
  the transmissions out of ``v``.

A committed request has the values of its calendar instead (section 7): its task runs in its
committed window only, its data crosses its committed arcs only, and its intervals are the
committed instants.

Only what can happen has an interval: a window that is not compatible, an arc that cannot be
crossed, a PLMM that the data never enters have none. A data item can be in the PLMM of ``v`` only
if its producer can run on ``v``, if it can arrive on ``v``, or if it is the raw data on its
origin: these are the only ways in which the presence ``q_{a,v}`` can be 1. That implies the
condition ``EA_{a,v} ≤ LA_{a,v}`` of the document, and it keeps every stay that the variables of
the model allow, whatever the rounding of the bounds.

The comparisons use the tolerance of the checker, in the direction that keeps more, and so do the
bounds they start from (:meth:`ModelData.bounds`): an activity that overflows its window, or
starts before its data is there, by less than the checker accepts is never removed for that.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from satedge_workflow_scheduler.common.intervals import TimeWindow
from satedge_workflow_scheduler.domain.nodes import NodeId
from satedge_workflow_scheduler.scheduling.bounds import RequestBounds
from satedge_workflow_scheduler.scheduling.feasibility import FEASIBILITY_TOLERANCE
from satedge_workflow_scheduler.solvers.milp.model_data import (
    ArcId,
    DataKey,
    ModelData,
    TaskKey,
)

_TOLERANCE = FEASIBILITY_TOLERANCE
"""How much shorter than needed an interval may be and still be kept: the checker's tolerance."""


@dataclass(frozen=True, slots=True)
class Span:
    """A closed interval ``[lo, hi]`` of instants. It is empty when ``lo > hi``, and its ends may
    be infinite."""

    lo: float
    hi: float

    @classmethod
    def of(cls, window: TimeWindow) -> Span:
        """The span of a window."""
        return cls(window.start, window.end)

    @property
    def length(self) -> float:
        """``hi − lo``: negative when the span is empty."""
        return self.hi - self.lo

    def meet(self, other: Span) -> Span:
        """The common part of the two spans, possibly empty."""
        return Span(max(self.lo, other.lo), min(self.hi, other.hi))


ANYTIME = Span(-math.inf, math.inf)
"""The span that contains every instant: what is known of a stay whose bounds are not finite."""

_ALWAYS = Span(0.0, math.inf)
"""The single window of an arc always available."""


@dataclass(frozen=True)
class RequestRanges:
    """The intervals of one request, on what the request can use only.

    Attributes:
        tasks: ``I^task_{i,s,w}`` by task, satellite and compatible window. Every task and every
            satellite is a key; a satellite with no compatible window maps to nothing.
        links: ``I^link_{a,e}`` of each arc that the data can cross.
        link_windows: The compatible windows of each arc with visibility windows that the data
            can cross, in chronological order.
        stays: ``I^PLMM_{a,v}`` of each satellite whose PLMM the data can enter.
    """

    tasks: Mapping[TaskKey, Mapping[NodeId, Mapping[int, Span]]]
    links: Mapping[tuple[DataKey, ArcId], Span]
    link_windows: Mapping[tuple[DataKey, ArcId], tuple[int, ...]]
    stays: Mapping[tuple[DataKey, NodeId], Span]

    @property
    def realizable(self) -> bool:
        """Whether every task can run on some satellite: if not, the request is never served."""
        return all(any(per_satellite.values()) for per_satellite in self.tasks.values())

    def task(self, i: TaskKey, satellite: NodeId) -> Mapping[int, Span]:
        """The compatible windows of the task on ``satellite``, with ``I^task`` in each."""
        return self.tasks[i][satellite]


def request_ranges(data: ModelData, request_id: str) -> RequestRanges:
    """The intervals of the request: from its calendar if it is committed, from its a priori
    bounds otherwise."""
    if data.is_committed(request_id):
        return _committed_ranges(data, request_id)
    return _new_ranges(data, request_id)


# --- new requests: the a priori bounds ---------------------------------------------------


def _new_ranges(data: ModelData, request_id: str) -> RequestRanges:
    bounds = data.bounds(request_id)

    tasks: dict[TaskKey, dict[NodeId, dict[int, Span]]] = {}
    for i in data.tasks_of(request_id):
        tasks[i] = {}
        for s in data.satellites:
            allowed = Span(
                bounds.earliest_start(i.task_index, s),
                bounds.latest_completion(i.task_index, s),
            )
            tasks[i][s] = {
                w: allowed.meet(Span.of(window))
                for w, window in bounds.compatible_indices(i.task_index, s).items()
            }

    links: dict[tuple[DataKey, ArcId], Span] = {}
    link_windows: dict[tuple[DataKey, ArcId], tuple[int, ...]] = {}
    for a in data.data_of(request_id):
        for e in data.arc_ids:
            u, v = data.endpoints(e)
            lo = max(
                bounds.earliest_availability(a.data_index, u), data.departure_release(u)
            )
            hi = bounds.latest_availability(a.data_index, v)
            if not (math.isfinite(lo) and math.isfinite(hi)):
                continue
            span = Span(lo, hi)
            needed = data.transmission_time(a, e) + data.latency(e) - _TOLERANCE
            windows = data.link_windows(e)
            if windows is None:
                if span.meet(_ALWAYS).length >= needed:
                    links[(a, e)] = span
                continue
            compatible = tuple(
                h
                for h, window in enumerate(windows)
                if span.meet(Span.of(window)).length >= needed
            )
            if compatible:
                links[(a, e)] = span
                link_windows[(a, e)] = compatible

    stays: dict[tuple[DataKey, NodeId], Span] = {}
    for a in data.data_of(request_id):
        for v in data.satellites:
            stay = _new_stay(data, bounds, tasks, links, a, v)
            if stay is not None:
                stays[(a, v)] = stay
    return RequestRanges(tasks, links, link_windows, stays)


def _new_stay(
    data: ModelData,
    bounds: RequestBounds,
    tasks: Mapping[TaskKey, Mapping[NodeId, Mapping[int, Span]]],
    links: Mapping[tuple[DataKey, ArcId], Span],
    a: DataKey,
    v: NodeId,
) -> Span | None:
    """``I^PLMM_{a,v}`` of a new request, or ``None`` if ``a`` can never be on ``v``."""
    producer, consumer = data.producer(a), data.consumer(a)
    produced_here = producer is not None and bool(tasks[producer][v])
    is_raw_here = producer is None and v == data.origin(a.request_id)
    arriving = [e for e in data.incoming(v) if (a, e) in links]
    if not (produced_here or is_raw_here or arriving):
        return None

    leaving = [e for e in data.outgoing(v) if (a, e) in links]
    enter = max(
        [data.transmission_time(a, e) for e in arriving]
        + ([data.hpc_io_time(data.size(a), v)] if produced_here else []),
        default=0.0,
    )
    exit_ = max(
        [data.transmission_time(a, e) for e in leaving]
        + (
            [data.duration(consumer, v)]
            if consumer is not None and tasks[consumer][v]
            else []
        ),
        default=0.0,
    )
    earliest = bounds.earliest_availability(a.data_index, v)
    latest = bounds.latest_availability(a.data_index, v)
    if not (math.isfinite(earliest) and math.isfinite(latest)):
        return ANYTIME
    return Span(earliest - enter, max(latest, earliest) + exit_)


# --- committed requests: the calendar ----------------------------------------------------


def _committed_ranges(data: ModelData, request_id: str) -> RequestRanges:
    tasks: dict[TaskKey, dict[NodeId, dict[int, Span]]] = {}
    for i in data.tasks_of(request_id):
        fixed = data.committed_execution(i)
        tasks[i] = {s: {} for s in data.satellites}
        end = fixed.start + data.duration(i, fixed.satellite)
        tasks[i][fixed.satellite] = {fixed.window: Span(fixed.start, end)}

    links: dict[tuple[DataKey, ArcId], Span] = {}
    link_windows: dict[tuple[DataKey, ArcId], tuple[int, ...]] = {}
    for a in data.data_of(request_id):
        for e, hop in data.committed_transfers(a).items():
            arrival = hop.start + data.transmission_time(a, e) + data.latency(e)
            links[(a, e)] = Span(hop.start, arrival)
            if hop.window is not None:
                link_windows[(a, e)] = (hop.window,)

    stays: dict[tuple[DataKey, NodeId], Span] = {}
    for a in data.data_of(request_id):
        for v in data.satellites:
            stay = _committed_stay(data, links, a, v)
            if stay is not None:
                stays[(a, v)] = stay
    return RequestRanges(tasks, links, link_windows, stays)


def _committed_stay(
    data: ModelData,
    links: Mapping[tuple[DataKey, ArcId], Span],
    a: DataKey,
    v: NodeId,
) -> Span | None:
    """The stay of a committed data item in the PLMM of ``v``: from (20a)–(20c) to (21a)–(21b)
    evaluated on the calendar, or ``None`` if the calendar never brings it there."""
    producer, consumer = data.producer(a), data.consumer(a)
    starts: list[float] = []
    if producer is None:
        if v == data.origin(a.request_id):
            starts.append(data.release(a.request_id))
    else:
        made = data.committed_execution(producer)
        if made.satellite == v:
            io = data.hpc_io_time(data.size(a), v)
            starts.append(made.start + data.duration(producer, v) - io)
    starts.extend(
        links[(a, e)].lo + data.latency(e) for e in data.incoming(v) if (a, e) in links
    )
    if not starts:
        return None

    ends: list[float] = [
        links[(a, e)].lo + data.transmission_time(a, e)
        for e in data.outgoing(v)
        if (a, e) in links
    ]
    if consumer is not None:
        used = data.committed_execution(consumer)
        if used.satellite == v:
            ends.append(used.start + data.duration(consumer, v))
    lo = min(starts)
    return Span(lo, max([lo, *ends]))
