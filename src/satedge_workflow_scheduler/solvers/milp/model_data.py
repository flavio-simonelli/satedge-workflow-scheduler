"""The data of the model: the sets and the parameters of the notation, read from an instance.

:class:`ModelData` translates the instance into the notation of the document: the tasks and
the data of the new requests, the arcs, the windows, the durations, the sizes and the costs. It
holds data only, and nothing about variables, constraints or solvers. The blocks read it, and
never read the instance directly.

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
from satedge_workflow_scheduler.scheduling.deadline import (
    DeadlineMode,
    delivery_limit,
    may_reject,
)
from satedge_workflow_scheduler.scheduling.objective import (
    rejection_weights,
    tardiness_weights,
)
from satedge_workflow_scheduler.scheduling.windows import arc_windows

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


class ModelData:
    """The sets and parameters of the notation, for the new requests of an instance.

    Committed requests are not in the model data: their calendar is fixed by the online state, and
    their tasks and data do not get variables.

    Args:
        instance: The instance to read.
        deadline: The deadline mode, which sets the time limits and the rejections.
    """

    def __init__(self, instance: ProblemInstance, deadline: DeadlineMode) -> None:
        self._instance = instance
        self._deadline = deadline
        self._requests = {r.id: r for r in instance.new_requests}
        self._rejection_costs = rejection_weights(instance.requests)
        self._tardiness_weights = tardiness_weights(instance.requests, instance.horizon)

    # --- sets ---------------------------------------------------------------------------

    @property
    def request_ids(self) -> tuple[str, ...]:
        """Identifiers of the new requests, in instance order (the FIFO tie-break order)."""
        return tuple(self._requests)

    @property
    def tasks(self) -> tuple[TaskKey, ...]:
        """The real tasks ``I`` of the new requests, pipeline by pipeline."""
        return tuple(
            TaskKey(rid, k)
            for rid in self.request_ids
            for k in range(len(self._request(rid).tasks))
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
        """The data ``D̄`` of the new requests: the raw data, the intermediate data and the
        final output of each request."""
        return tuple(
            DataKey(rid, k)
            for rid in self.request_ids
            for k in range(len(self._request(rid).tasks) + 1)
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
        """Whether the new requests may be rejected under the deadline policy."""
        return may_reject(self._deadline)

    def release(self, request_id: str) -> float:
        """``b_r``: instant from which the raw data of the request is available."""
        return self._request(request_id).release

    def plan_release(self, satellite: NodeId) -> float:
        """``b^plan_s``: instant from which the new plan reaches ``satellite``."""
        return self._instance.release_of(satellite)

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
