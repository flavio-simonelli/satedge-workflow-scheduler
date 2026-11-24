"""The exact reductions of the model: which variables and constraints it contains (rules 1–6).

Section 6 of the document. The reductions remove the variables that are zero in every feasible
solution and the constraints that every feasible solution satisfies, so they shrink the model
without changing its optimum. They read the intervals of :mod:`.time_ranges`:

1. **satellites and computing windows**: ``u_{i,s,w}`` exists only on the windows compatible
   with ``i``, so ``x_{i,s}`` only where ``i`` can run;
2. **pairs of tasks**: (8) is written only on the windows compatible with both tasks in which
   their intervals overlap for a positive length; a pair with no such window leaves ``Q``, with
   ``same``, ``order`` and (9)–(10);
3. **arcs and visibility windows**: ``χ_{a,e}`` exists only on the arcs with a window compatible
   with ``a``, and ``ν_{a,e,h}`` only on the compatible windows. The start ``ξ_{a,e}`` and every
   constraint on it go with ``χ_{a,e}``: on an arc that is never crossed they bind nothing. In
   particular a data item that is not the final output never enters the ground station;
4. **pairs of data on the links**: ``σ_{a,b,e}`` and (18)–(19) exist only if both data can cross
   ``e`` and the intervals in which they can occupy it overlap for a positive length, inside a
   window compatible with both if the arc has windows;
5. **PLMM only where needed**: the PLMM of ``v`` holds only the data that can be on ``v``, and its
   capacity is written only if the peak of the sizes over the intervals ``I^PLMM`` exceeds
   ``K_v``. When ``I^PLMM_{a,v}`` ends before ``I^PLMM_{b,v}`` starts, ``π_{a,b,v}`` is fixed to 1
   and ``π_{b,a,v}`` to 0, and (23) is not written for the pair: removing the pair would stop
   the capacity from passing from ``a`` to ``b``, and cut feasible solutions;
6. **x continuous**: ``x_{i,s}`` is the sum of the ``u_{i,s,w}``, so it is an expression and
   never a variable (:mod:`.variables`). It holds with or without the other reductions.

A request that can never be served (section "Richieste impossibili") is *excluded*: under hard
deadlines it is rejected, ``ρ_r = 1`` is a constant and its cost a constant of the objective;
under soft deadlines the instance has no solution, and the error names the task that cannot run.

:class:`Reductions` answers the questions of the blocks on what exists. :meth:`Reductions.none`
keeps everything, the model of section 4 as written; :meth:`Reductions.exact` applies the rules.
The blocks do not know which of the two they read.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import combinations

from satedge_workflow_scheduler.domain.nodes import NodeId
from satedge_workflow_scheduler.solvers.errors import InfeasibleModelError
from satedge_workflow_scheduler.solvers.milp.errors import ModelBuildError
from satedge_workflow_scheduler.solvers.milp.model_data import (
    ArcId,
    DataKey,
    ModelData,
    TaskKey,
)
from satedge_workflow_scheduler.solvers.milp.time_ranges import (
    RequestRanges,
    Span,
    request_ranges,
)

TaskPair = tuple[TaskKey, TaskKey]
"""A pair ``(i, j)`` of tasks of different requests, as in ``Q``."""

LinkTriple = tuple[DataKey, DataKey, ArcId]
"""A pair ``(a, b)`` of data of different requests and an arc: the key of ``σ_{a,b,e}``."""

HpcWindow = tuple[NodeId, int]
"""A computing window: the satellite and the position of the window."""


class Reductions:
    """What the model contains: the variables that exist and the constraints that are written.

    Built by :meth:`none` or :meth:`exact`. Every query takes keys of the model data; asking about
    a key the data do not have is an error, not an empty answer.
    """

    def __init__(
        self,
        *,
        excluded: frozenset[str],
        task_windows: Mapping[tuple[TaskKey, NodeId], tuple[int, ...]],
        shared_windows: Mapping[TaskPair, tuple[HpcWindow, ...]],
        arcs: Mapping[DataKey, tuple[ArcId, ...]],
        link_windows: Mapping[tuple[DataKey, ArcId], tuple[int, ...]],
        link_triples: tuple[LinkTriple, ...],
        plmm_satellites: tuple[NodeId, ...],
        plmm_data: Mapping[NodeId, tuple[DataKey, ...]],
        plmm_order: Mapping[tuple[DataKey, DataKey, NodeId], bool],
    ) -> None:
        self._excluded = excluded
        self._task_windows = task_windows
        self._shared_windows = shared_windows
        self._arcs = arcs
        self._arc_sets = {a: frozenset(arcs) for a, arcs in arcs.items()}
        self._link_windows = link_windows
        self._link_triples = link_triples
        self._plmm_satellites = plmm_satellites
        self._plmm_data = plmm_data
        self._plmm_sets = {v: frozenset(data) for v, data in plmm_data.items()}
        self._plmm_order = plmm_order

    # --- construction ---------------------------------------------------------------------

    @classmethod
    def none(cls, data: ModelData) -> Reductions:
        """No reduction: every variable and constraint of the model as written."""
        sats = data.satellites
        every_window = {s: tuple(range(len(data.hpc_windows(s)))) for s in sats}
        return cls(
            excluded=frozenset(),
            task_windows={(i, s): every_window[s] for i in data.tasks for s in sats},
            shared_windows={
                pair: tuple((s, w) for s in sats for w in every_window[s])
                for pair in data.task_pairs
            },
            arcs={a: data.arc_ids for a in data.data},
            link_windows={
                (a, e): tuple(range(len(windows)))
                for a in data.data
                for e in data.arc_ids
                if (windows := data.link_windows(e)) is not None
            },
            link_triples=tuple(
                (a, b, e) for a, b in data.link_pairs for e in data.arc_ids
            ),
            plmm_satellites=sats,
            plmm_data={v: data.data for v in sats},
            plmm_order={},
        )

    @classmethod
    def exact(cls, data: ModelData) -> Reductions:
        """The reductions 1–5 of the document, on the intervals of the requests.

        Raises:
            InfeasibleModelError: If some request can never be served and the deadlines are
                soft, so it cannot be rejected either. Its reasons name, for each such request,
                the task that can run nowhere.
        """
        ranges: dict[str, RequestRanges] = {}
        excluded: set[str] = set()
        impossible: dict[str, str] = {}
        for r in data.request_ids:
            found = request_ranges(data, r)
            if not found.realizable:
                if data.may_reject():
                    excluded.add(r)
                else:
                    impossible[r] = (
                        data.unrealizable_reason(r)
                        or "un task non può essere eseguito su nessun satellite"
                    )
            ranges[r] = found
        if impossible:
            raise InfeasibleModelError(
                f"requests that can never be delivered: {', '.join(sorted(impossible))}",
                impossible,
            )
        return _ExactBuilder(data, ranges, frozenset(excluded)).build()

    # --- requests -------------------------------------------------------------------------

    @property
    def excluded(self) -> frozenset[str]:
        """The requests out of the model: never served, so rejected a priori."""
        return self._excluded

    def is_excluded(self, request_id: str) -> bool:
        """Whether the request is out of the model."""
        return request_id in self._excluded

    # --- rules 1 and 2: tasks -------------------------------------------------------------

    def task_windows(self, i: TaskKey, satellite: NodeId) -> tuple[int, ...]:
        """The windows of ``satellite`` that have ``u_{i,s,w}``, in chronological order."""
        return _lookup(self._task_windows, (i, satellite), "task windows")

    @property
    def task_pairs(self) -> tuple[TaskPair, ...]:
        """The pairs of ``Q`` that may overlap, in the order of :attr:`ModelData.task_pairs`."""
        return tuple(self._shared_windows)

    def shared_windows(self, i: TaskKey, j: TaskKey) -> tuple[HpcWindow, ...]:
        """The windows on which (8) is written for the pair ``(i, j)``."""
        return _lookup(self._shared_windows, (i, j), "shared windows")

    # --- rules 3 and 4: arcs --------------------------------------------------------------

    def arcs(self, a: DataKey) -> tuple[ArcId, ...]:
        """The arcs that ``a`` can cross, with ``χ_{a,e}`` and ``ξ_{a,e}``, in arc order."""
        return _lookup(self._arcs, a, "arcs")

    def crosses(self, a: DataKey, e: ArcId) -> bool:
        """Whether ``a`` can cross ``e``: ``χ_{a,e}`` exists."""
        return e in _lookup(self._arc_sets, a, "arcs")

    def link_windows(self, a: DataKey, e: ArcId) -> tuple[int, ...]:
        """The visibility windows of ``e`` with ``ν_{a,e,h}``, for an arc with windows that ``a``
        can cross."""
        return _lookup(self._link_windows, (a, e), "link windows")

    @property
    def link_triples(self) -> tuple[LinkTriple, ...]:
        """The triples with ``σ_{a,b,e}`` and (18)–(19), in the order of the pairs, then of the
        arcs."""
        return self._link_triples

    # --- rule 5: PLMM ---------------------------------------------------------------------

    @property
    def plmm_satellites(self) -> tuple[NodeId, ...]:
        """The satellites whose PLMM can be filled beyond its capacity, in network order."""
        return self._plmm_satellites

    def plmm_data(self, satellite: NodeId) -> tuple[DataKey, ...]:
        """The data that can be in the PLMM of ``satellite``, in the order of the data."""
        return _lookup(self._plmm_data, satellite, "PLMM data")

    def in_plmm(self, a: DataKey, satellite: NodeId) -> bool:
        """Whether ``a`` can be in the PLMM of ``satellite``."""
        return a in _lookup(self._plmm_sets, satellite, "PLMM data")

    def plmm_order(self, a: DataKey, b: DataKey, satellite: NodeId) -> bool | None:
        """``π_{a,b,v}`` when it is fixed: ``True`` if ``a`` always leaves before ``b`` enters,
        ``False`` if ``b`` always leaves before ``a`` enters, ``None`` if the model decides."""
        return self._plmm_order.get((a, b, satellite))


def reductions_for(data: ModelData, enabled: bool) -> Reductions:
    """The exact reductions if ``enabled``, none otherwise."""
    return Reductions.exact(data) if enabled else Reductions.none(data)


class _ExactBuilder:
    """Computes the sets of :meth:`Reductions.exact` from the intervals of every request."""

    def __init__(
        self,
        data: ModelData,
        ranges: Mapping[str, RequestRanges],
        excluded: frozenset[str],
    ) -> None:
        self._data = data
        self._ranges = ranges
        self._excluded = excluded

    def build(self) -> Reductions:
        plmm_data = self._plmm_data()
        plmm_satellites = self._plmm_satellites(plmm_data)
        return Reductions(
            excluded=self._excluded,
            task_windows=self._task_windows(),
            shared_windows=self._shared_windows(),
            arcs=self._arcs(),
            link_windows=self._link_windows(),
            link_triples=self._link_triples(),
            plmm_satellites=plmm_satellites,
            plmm_data=plmm_data,
            plmm_order=self._plmm_order(plmm_satellites, plmm_data),
        )

    # --- what each request can use, nothing for an excluded one ---------------------------

    def _windows(self, i: TaskKey, s: NodeId) -> Mapping[int, Span]:
        if i.request_id in self._excluded:
            return {}
        return self._ranges[i.request_id].task(i, s)

    def _link(self, a: DataKey, e: ArcId) -> Span | None:
        if a.request_id in self._excluded:
            return None
        return self._ranges[a.request_id].links.get((a, e))

    def _stay(self, a: DataKey, v: NodeId) -> Span | None:
        if a.request_id in self._excluded:
            return None
        return self._ranges[a.request_id].stays.get((a, v))

    # --- rule 1 ---------------------------------------------------------------------------

    def _task_windows(self) -> dict[tuple[TaskKey, NodeId], tuple[int, ...]]:
        data = self._data
        return {
            (i, s): tuple(sorted(self._windows(i, s)))
            for i in data.tasks
            for s in data.satellites
        }

    # --- rule 2 ---------------------------------------------------------------------------

    def _shared_windows(self) -> dict[TaskPair, tuple[HpcWindow, ...]]:
        result: dict[TaskPair, tuple[HpcWindow, ...]] = {}
        for i, j in self._data.task_pairs:
            shared = tuple(
                (s, w)
                for s in self._data.satellites
                for w, span in sorted(self._windows(i, s).items())
                if (other := self._windows(j, s).get(w)) is not None
                and span.meet(other).length > 0
            )
            if shared:
                result[(i, j)] = shared
        return result

    # --- rule 3 ---------------------------------------------------------------------------

    def _arcs(self) -> dict[DataKey, tuple[ArcId, ...]]:
        data = self._data
        return {
            a: tuple(e for e in data.arc_ids if self._link(a, e) is not None)
            for a in data.data
        }

    def _link_windows(self) -> dict[tuple[DataKey, ArcId], tuple[int, ...]]:
        result: dict[tuple[DataKey, ArcId], tuple[int, ...]] = {}
        for r, found in self._ranges.items():
            if r not in self._excluded:
                result.update(found.link_windows)
        return result

    # --- rule 4 ---------------------------------------------------------------------------

    def _link_triples(self) -> tuple[LinkTriple, ...]:
        data = self._data
        triples: list[LinkTriple] = []
        for a, b in data.link_pairs:
            for e in data.arc_ids:
                if self._may_share(a, b, e):
                    triples.append((a, b, e))
        return tuple(triples)

    def _may_share(self, a: DataKey, b: DataKey, e: ArcId) -> bool:
        """Whether ``a`` and ``b`` can occupy ``e`` at the same time.

        The arc is busy with ``a`` for ``[ξ_{a,e}, ξ_{a,e} + D_{a,e}]``, which lies in
        ``I^link_{a,e}`` shortened by ``λ_e`` at its end.
        """
        link_a, link_b = self._link(a, e), self._link(b, e)
        if link_a is None or link_b is None:
            return False
        latency = self._data.latency(e)
        busy = Span(link_a.lo, link_a.hi - latency).meet(
            Span(link_b.lo, link_b.hi - latency)
        )
        windows = self._data.link_windows(e)
        if windows is None:
            return busy.length > 0
        both = set(self._ranges[a.request_id].link_windows[(a, e)]) & set(
            self._ranges[b.request_id].link_windows[(b, e)]
        )
        return any(
            busy.meet(Span.of(windows.windows[h])).length > 0 for h in sorted(both)
        )

    # --- rule 5 ---------------------------------------------------------------------------

    def _plmm_data(self) -> dict[NodeId, tuple[DataKey, ...]]:
        data = self._data
        return {
            v: tuple(a for a in data.data if self._stay(a, v) is not None)
            for v in data.satellites
        }

    def _plmm_satellites(
        self, plmm_data: Mapping[NodeId, tuple[DataKey, ...]]
    ) -> tuple[NodeId, ...]:
        data = self._data
        return tuple(
            v
            for v in data.satellites
            if _peak((data.size(a), _present(self._stay(a, v))) for a in plmm_data[v])
            > data.plmm_capacity(v)
        )

    def _plmm_order(
        self,
        plmm_satellites: Iterable[NodeId],
        plmm_data: Mapping[NodeId, tuple[DataKey, ...]],
    ) -> dict[tuple[DataKey, DataKey, NodeId], bool]:
        order: dict[tuple[DataKey, DataKey, NodeId], bool] = {}
        for v in plmm_satellites:
            for a, b in combinations(plmm_data[v], 2):
                stay_a = _present(self._stay(a, v))
                stay_b = _present(self._stay(b, v))
                if stay_a.hi <= stay_b.lo:
                    order[(a, b, v)], order[(b, a, v)] = True, False
                elif stay_b.hi <= stay_a.lo:
                    order[(a, b, v)], order[(b, a, v)] = False, True
        return order


def _present(stay: Span | None) -> Span:
    if stay is None:
        raise ModelBuildError("a data item of the PLMM has no stay")
    return stay


def _peak(stays: Iterable[tuple[float, Span]]) -> float:
    """The largest total size of the stays that overlap at one instant.

    The stays are closed: two stays that touch count as overlapping, which can only keep a
    capacity that is not needed, never drop one that is.
    """
    events: list[tuple[float, int, float]] = []
    for size, stay in stays:
        if size > 0:
            events.append((stay.lo, 0, size))
            events.append((stay.hi, 1, -size))
    events.sort(key=lambda event: (event[0], event[1]))
    load = peak = 0.0
    for _, _, change in events:
        load += change
        peak = max(peak, load)
    return peak


def _lookup[K, V](table: Mapping[K, V], key: K, what: str) -> V:
    try:
        return table[key]
    except KeyError:
        raise ModelBuildError(f"no {what} for {key}: not a key of the model") from None
