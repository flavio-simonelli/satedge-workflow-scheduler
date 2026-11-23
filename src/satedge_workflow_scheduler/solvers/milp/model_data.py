"""The data of the model: the sets and the parameters of the notation, read from an instance.

:class:`ModelData` translates the instance into the notation of the document: the tasks and
the data of all the requests, committed and new, the committed calendar, the arcs, the windows,
the durations, the sizes, the costs and the a priori time bounds of the requests. It holds data
only, and nothing about variables, constraints or solvers. The blocks read it, and never read the
instance directly.

Tasks and data are identified by keys, not by integers, so that a block can say ``(i, s)`` and
read as the document does:

- a task ``i`` is a :class:`TaskKey` ``(request_id, task_index)``;
- a data item ``a`` is a :class:`DataKey` ``(request_id, data_index)``: index ``0`` is the raw
  data, index ``k`` is the output of task ``k - 1``, and the last one is the final output.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from satedge_workflow_scheduler.common.intervals import WindowSet
from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.network import Arc
from satedge_workflow_scheduler.domain.nodes import NodeId
from satedge_workflow_scheduler.domain.requests import Request
from satedge_workflow_scheduler.scheduling.bounds import RequestBounds, request_bounds
from satedge_workflow_scheduler.scheduling.deadline import (
    DeadlineMode,
    delivery_limit,
    may_reject,
)
from satedge_workflow_scheduler.scheduling.earliest import earliest_arrivals
from satedge_workflow_scheduler.scheduling.feasibility import FEASIBILITY_TOLERANCE
from satedge_workflow_scheduler.scheduling.objective import (
    rejection_weights,
    response_epsilon,
    tardiness_weights,
)
from satedge_workflow_scheduler.scheduling.screening import unrealizable_reason
from satedge_workflow_scheduler.scheduling.windows import arc_windows
from satedge_workflow_scheduler.solvers.milp.errors import ModelBuildError

ArcId = int
"""Position of an arc in ``Network.arcs``: the arcs are referred to by it."""


@dataclass(frozen=True, order=True)
class TaskKey:
    """A real task ``i`` of a request: the ``task_index``-th task of ``request_id``."""

    request_id: str
    task_index: int

    def __str__(self) -> str:
        return f"{self.request_id}.{self.task_index}"


@dataclass(frozen=True, order=True)
class DataKey:
    """A data item ``a`` of a request: ``0`` is the raw data, ``k`` the output of task ``k-1``."""

    request_id: str
    data_index: int

    def __str__(self) -> str:
        return f"{self.request_id}#{self.data_index}"


@dataclass(frozen=True)
class CommittedExecution:
    """The committed execution of a task: ``ŝ_i``, ``ŵ_i`` and ``t̂_i`` of the document.

    Attributes:
        satellite: The satellite that runs the task.
        window: Position of the computing window that contains the execution.
        start: The start of the execution.
    """

    satellite: NodeId
    window: int
    start: float


@dataclass(frozen=True)
class CommittedTransfer:
    """The committed transmission of a data item on an arc: ``ξ̂_{a,e}`` and ``ĥ_{a,e}``.

    Attributes:
        start: The start of the transmission.
        window: Position of the visibility window that contains it, or ``None`` on an arc that
            is always available.
    """

    start: float
    window: int | None


class ModelData:
    """The sets and parameters of the notation, for all the requests of an instance.

    The committed requests are in the model like the new ones, with their variables: the online
    block fixes them to the committed calendar (option A of the document). Keeping them in the
    model lets the existing blocks keep the new requests away from them on the HPCs, on the arcs
    and in the PLMMs, and it leaves room for a future replanning, which would fix fewer of them.

    Args:
        instance: The instance to read.
        deadline: The deadline mode, which sets the time limits and the rejections.
    """

    def __init__(self, instance: ProblemInstance, deadline: DeadlineMode) -> None:
        self._instance = instance
        self._deadline = deadline
        self._requests = {r.id: r for r in instance.requests}
        self._committed = instance.committed.request_ids
        self._delays: dict[tuple[NodeId, float], Mapping[NodeId, float]] = {}
        self._bounds: dict[str, RequestBounds] = {}
        self._arc_of = {
            (arc.source, arc.target): e for e, arc in enumerate(instance.network.arcs)
        }
        self._executions = {
            TaskKey(x.request_id, x.task_index): x
            for x in instance.committed.executions
        }
        self._transfers: dict[DataKey, dict[ArcId, float]] = {}
        for t in instance.committed.transfers:
            arc = self.arc_between(t.source, t.target)
            self._transfers.setdefault(DataKey(t.request_id, t.data_index), {})[arc] = (
                t.start
            )
        self._rejection_costs = rejection_weights(instance.requests)
        self._tardiness_weights = tardiness_weights(instance.requests, instance.horizon)

    # --- sets ---------------------------------------------------------------------------

    @property
    def request_ids(self) -> tuple[str, ...]:
        """Identifiers of all the requests, committed and new, in instance order (the FIFO
        tie-break order)."""
        return tuple(self._requests)

    @property
    def new_request_ids(self) -> tuple[str, ...]:
        """``R^new``: the requests still to plan, in instance order."""
        return tuple(r for r in self._requests if r not in self._committed)

    def is_committed(self, request_id: str) -> bool:
        """Whether the request is in ``R^imp``, the committed calendar."""
        return request_id in self._committed

    @property
    def is_online(self) -> bool:
        """Whether the instance has an online state: committed requests or a plan release."""
        return bool(self._committed) or any(
            self.plan_release(s) > 0 for s in self.satellites
        )

    # --- committed calendar --------------------------------------------------------------

    def committed_execution(self, i: TaskKey) -> CommittedExecution:
        """``ŝ_i``, ``ŵ_i``, ``t̂_i``: the committed execution of the task ``i``.

        Raises:
            ModelBuildError: If the task is not committed, or if its execution is in no
                computing window of its satellite.
        """
        execution = self._executions.get(i)
        if execution is None:
            raise ModelBuildError(f"task {i} is not committed")
        windows = self.hpc_windows(execution.satellite)
        for w, window in enumerate(windows):
            if _contains(window.start, window.end, execution.start, execution.end):
                return CommittedExecution(execution.satellite, w, execution.start)
        raise ModelBuildError(
            f"committed task {i} on {execution.satellite} is in no computing window"
        )

    def committed_transfers(self, a: DataKey) -> Mapping[ArcId, CommittedTransfer]:
        """``χ̂_{a,·}``, ``ξ̂_{a,·}``, ``ĥ_{a,·}``: the arcs a committed data item crosses.

        The arcs not in the result are not crossed. The visibility window of a transmission is
        the one that contains it, transmission and latency included.

        Raises:
            ModelBuildError: If a transmission is in no visibility window of its arc.
        """
        result: dict[ArcId, CommittedTransfer] = {}
        for e, start in self._transfers.get(a, {}).items():
            windows = self.link_windows(e)
            if windows is None:
                result[e] = CommittedTransfer(start, None)
                continue
            end = start + self.transmission_time(a, e) + self.latency(e)
            for h, window in enumerate(windows):
                if _contains(window.start, window.end, start, end):
                    result[e] = CommittedTransfer(start, h)
                    break
            else:
                raise ModelBuildError(
                    f"committed transmission of {a} on arc {e} is in no visibility window"
                )
        return result

    def arc_between(self, source: NodeId, target: NodeId) -> ArcId:
        """The arc from ``source`` to ``target``.

        Raises:
            ModelBuildError: If there is no such arc.
        """
        try:
            return self._arc_of[(source, target)]
        except KeyError:
            raise ModelBuildError(f"no arc from {source} to {target}") from None

    @property
    def tasks(self) -> tuple[TaskKey, ...]:
        """The real tasks ``I`` of all the requests, pipeline by pipeline."""
        return tuple(
            TaskKey(rid, k)
            for rid in self.request_ids
            for k in range(len(self._request(rid).tasks))
        )

    def tasks_of(self, request_id: str) -> tuple[TaskKey, ...]:
        """The real tasks of one request, in pipeline order."""
        return tuple(
            TaskKey(request_id, k) for k in range(len(self._request(request_id).tasks))
        )

    @property
    def task_pairs(self) -> tuple[tuple[TaskKey, TaskKey], ...]:
        """The pairs ``Q`` of tasks of different requests, ``(i, j)`` with ``i`` before ``j``.

        Tasks of the same request do not overlap: the precedences order them."""
        tasks = self.tasks
        return tuple(
            (i, j)
            for a, i in enumerate(tasks)
            for j in tasks[a + 1 :]
            if i.request_id != j.request_id
        )

    @property
    def link_pairs(self) -> tuple[tuple[DataKey, DataKey], ...]:
        """The pairs ``Q^link`` of data of different requests, ``(a, b)`` with ``a`` before ``b``
        in the order of :attr:`data`: each pair appears once."""
        data = self.data
        return tuple(
            (a, b)
            for k, a in enumerate(data)
            for b in data[k + 1 :]
            if a.request_id != b.request_id
        )

    @property
    def precedences(self) -> tuple[tuple[TaskKey, TaskKey], ...]:
        """The precedences ``A`` between consecutive tasks of each new request, ``(i, j)``
        with ``j`` the task after ``i`` in the pipeline."""
        return tuple(
            (TaskKey(rid, k), TaskKey(rid, k + 1))
            for rid in self.request_ids
            for k in range(len(self._request(rid).tasks) - 1)
        )

    @property
    def data(self) -> tuple[DataKey, ...]:
        """The data ``D̄`` of all the requests: the raw data, the intermediate data and the
        final output of each request."""
        return tuple(
            DataKey(rid, k)
            for rid in self.request_ids
            for k in range(len(self._request(rid).tasks) + 1)
        )

    def data_of(self, request_id: str) -> tuple[DataKey, ...]:
        """The data of one request: its raw data, its intermediate data and its final output."""
        return tuple(
            DataKey(request_id, k)
            for k in range(len(self._request(request_id).tasks) + 1)
        )

    @property
    def satellites(self) -> tuple[NodeId, ...]:
        """The satellites ``S``, in network order."""
        return self._instance.network.satellite_ids

    @property
    def nodes(self) -> tuple[NodeId, ...]:
        """The nodes ``V = S ∪ {g}``: the satellites and the ground station."""
        return self._instance.network.node_ids

    @property
    def ground_station(self) -> NodeId:
        """The ground station ``g``."""
        return self._instance.network.ground_station.id

    @property
    def arc_ids(self) -> tuple[ArcId, ...]:
        """The arcs ``E``, by position."""
        return tuple(range(len(self._instance.network.arcs)))

    # --- structure of the pipelines -----------------------------------------------------

    def _request(self, request_id: str) -> Request:
        """The domain request of ``request_id``: read here, and nowhere else in the model."""
        return self._requests[request_id]

    def producer(self, a: DataKey) -> TaskKey | None:
        """The task that produces the data ``a``; ``None`` for the raw data, which is on the
        origin satellite from the start."""
        if a.data_index == 0:
            return None
        return TaskKey(a.request_id, a.data_index - 1)

    def consumer(self, a: DataKey) -> TaskKey | None:
        """The task that consumes the data ``a``; ``None`` for the final output, which goes to
        the ground station."""
        if a.data_index == len(self._request(a.request_id).tasks):
            return None
        return TaskKey(a.request_id, a.data_index)

    def origin(self, request_id: str) -> NodeId:
        """The satellite that holds the raw data of the request."""
        return self._request(request_id).origin

    def final_data(self, request_id: str) -> DataKey:
        """The final output ``last_r`` of the request, which goes to the ground station."""
        return DataKey(request_id, len(self._request(request_id).tasks))

    def is_final(self, a: DataKey) -> bool:
        """Whether ``a`` is the final output of its request."""
        return a.data_index == len(self._request(a.request_id).tasks)

    # --- parameters of the tasks --------------------------------------------------------

    def duration(self, i: TaskKey, satellite: NodeId) -> float:
        """``L̄_{i,s}``: time that the task occupies the HPC of ``satellite``."""
        return self._request(i.request_id).hpc_duration(
            i.task_index, self._instance.network.satellite(satellite)
        )

    def hpc_windows(self, satellite: NodeId) -> WindowSet:
        """``W^HPC_s``: the computing windows of ``satellite``."""
        return self._instance.network.satellite(satellite).hpc_windows

    def plmm_capacity(self, satellite: NodeId) -> float:
        """``K_s``: capacity of the PLMM of ``satellite``, in Gb."""
        return self._instance.network.satellite(satellite).plmm_capacity

    def hpc_io_time(self, size: float, satellite: NodeId) -> float:
        """``w``: time to read or write ``size`` Gb between the PLMM and the HPC."""
        return self._instance.network.satellite(satellite).hpc_io_time(size)

    # --- parameters of the data and of the arcs -----------------------------------------

    def size(self, a: DataKey) -> float:
        """``d_a``: size of the data item, in Gb."""
        return self._request(a.request_id).data_sizes[a.data_index]

    def endpoints(self, e: ArcId) -> tuple[NodeId, NodeId]:
        """The tail and the head of the arc ``e``."""
        arc = self._arc(e)
        return arc.source, arc.target

    def _arc(self, e: ArcId) -> Arc:
        """The domain arc at position ``e``: read here, and nowhere else in the model."""
        return self._instance.network.arcs[e]

    def transmission_time(self, a: DataKey, e: ArcId) -> float:
        """``D_{a,e}``: time to transmit the data item on the arc."""
        return self._arc(e).transmission_time(self.size(a))

    def fixed_delay(self, source: NodeId, target: NodeId, size: float) -> float:
        """``γ^{hk}(d)``: the least delay of ``size`` Gb from ``source`` to ``target``.

        It is the shortest path on the arcs, where an arc costs its transmission time plus its
        latency (level 3 of the document: the visibility windows are not considered, and the
        routers are always on). Without a path, it is ``H + 1``, a delay no time limit allows, so
        that the pair can never be used.

        Raises:
            ModelBuildError: If a node is not in the network.
        """
        if source == target:
            return 0.0
        return self._shortest_delays(source, size).get(target, self.horizon + 1.0)

    def _shortest_delays(self, source: NodeId, size: float) -> Mapping[NodeId, float]:
        key = (source, size)
        if key not in self._delays:
            network = self._instance.network

            def static(arc: int, ready: float) -> tuple[float, float]:
                # without windows an arc is always open: the data leaves when it is ready
                link = network.arcs[arc]
                return ready, ready + link.transmission_time(size) + link.latency

            reached = earliest_arrivals(network, {source: 0.0}, static)
            self._delays[key] = {node: r.arrival for node, r in reached.items()}
        return self._delays[key]

    def latency(self, e: ArcId) -> float:
        """``λ_e``: propagation latency of the arc, in seconds."""
        return self._arc(e).latency

    def link_windows(self, e: ArcId) -> WindowSet | None:
        """``W^link_e``: visibility windows of the arc, ``None`` if it is always available."""
        return arc_windows(self._arc(e).availability)

    def outgoing(self, node: NodeId) -> tuple[ArcId, ...]:
        """``δ⁺(v)``: arcs leaving ``node``."""
        return self._instance.network.outgoing[node]

    def incoming(self, node: NodeId) -> tuple[ArcId, ...]:
        """``δ⁻(v)``: arcs entering ``node``."""
        return self._instance.network.incoming[node]

    # --- time, deadlines and objective ---------------------------------------------------

    @property
    def horizon(self) -> float:
        """``H``: the scheduling horizon, in seconds."""
        return self._instance.horizon

    def limit(self, request_id: str) -> float:
        """``τ̂_r``: time limit of the delivery of the request, under the deadline policy."""
        return delivery_limit(self._request(request_id), self._deadline, self.horizon)

    def may_reject(self) -> bool:
        """Whether the requests may be rejected under the deadline policy."""
        return may_reject(self._deadline)

    def due(self, request_id: str) -> float:
        """``dl_r``: the deadline of the request, as given by the instance."""
        return self._request(request_id).deadline

    def release(self, request_id: str) -> float:
        """``b_r``: instant from which the raw data of the request is available."""
        return self._request(request_id).release

    def plan_release(self, satellite: NodeId) -> float:
        """``b^plan_s``: instant from which the new plan reaches ``satellite``."""
        return self._instance.release_of(satellite)

    def departure_release(self, node: NodeId) -> float:
        """``b^plan_u`` of the tail ``u`` of an arc: when the new plan lets data leave ``node``.

        The ground station has no plan release: it is 0 there."""
        if node not in self.satellites:
            return 0.0
        return self.plan_release(node)

    # --- a priori bounds ------------------------------------------------------------------

    def bounds(self, request_id: str) -> RequestBounds:
        """``EA``, ``LA``, ``ES`` and ``LC`` of a new request, alone in the network.

        The bounds of the section "Limiti temporali a priori" of the document, for the time limit
        ``τ̂_r`` of the deadline mode. Each request is computed once, when first asked.

        The instants are compared with the tolerance of the checker, not with the tighter one
        of the heuristic: an activity that overflows its window by less than the checker
        accepts still fits, so the reductions that read the bounds never remove a calendar the
        checker accepts because of a single comparison.

        Raises:
            ModelBuildError: If the request is committed: its bounds are the values of its
                calendar, not those of a relaxation.
        """
        if self.is_committed(request_id):
            raise ModelBuildError(
                f"request {request_id} is committed: its calendar gives its bounds"
            )
        if request_id not in self._bounds:
            self._bounds[request_id] = request_bounds(
                self._instance,
                self._request(request_id),
                self.limit(request_id),
                eps=_TOLERANCE,
            )
        return self._bounds[request_id]

    def unrealizable_reason(self, request_id: str) -> str | None:
        """Why the new request can never be delivered by its time limit, or ``None`` if it may
        be (section "Richieste impossibili" of the document), with the tolerance of
        :meth:`bounds`."""
        return unrealizable_reason(
            self._request(request_id),
            self._instance,
            self.limit(request_id),
            eps=_TOLERANCE,
        )

    @property
    def plmm_big_m(self) -> float:
        """``M^P = max_r τ̂_r + max_i L̄_i + max_{a,e} D_{a,e} + max_e λ_e``: the big-M of the PLMM."""
        return (
            max(self.limit(r) for r in self.request_ids)
            + max(self.duration(i, s) for i in self.tasks for s in self.satellites)
            + max(self.transmission_time(a, e) for a in self.data for e in self.arc_ids)
            + max(self.latency(e) for e in self.arc_ids)
        )

    @property
    def epsilon(self) -> float:
        """``ε``: weight of the response time in the objective (1′) and (1″)."""
        return response_epsilon(self._instance.requests)

    @property
    def rejection_costs(self) -> Mapping[str, float]:
        """``κ_r``: cost of rejecting each request, with the hard deadlines."""
        return self._rejection_costs

    @property
    def tardiness_weights(self) -> Mapping[str, float]:
        """``ω_r``: weight of the tardiness of each request, with the soft deadlines."""
        return self._tardiness_weights

    @property
    def mode(self) -> DeadlineMode:
        """The deadline mode the data were read with."""
        return self._deadline


_TOLERANCE = FEASIBILITY_TOLERANCE
"""Tolerance on the instants of the committed calendar, the same as the checker's."""


def _contains(lo: float, hi: float, start: float, end: float) -> bool:
    return lo - _TOLERANCE <= start and end <= hi + _TOLERANCE
