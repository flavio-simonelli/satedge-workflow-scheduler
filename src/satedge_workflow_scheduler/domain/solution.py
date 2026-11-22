"""Solution of a scheduling instance: task executions, data transfers and request outcomes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from satedge_workflow_scheduler.domain.errors import InstanceError
from satedge_workflow_scheduler.domain.nodes import NodeId


class SolveStatus(StrEnum):
    """Status of the solution returned by a solver."""

    OPTIMAL = "optimal"
    FEASIBLE = "feasible"


@dataclass(frozen=True)
class TaskExecution:
    """Execution of one task of a request on a satellite.

    Attributes:
        request_id: Identifier of the request.
        task_index: Index of the task in the pipeline of the request.
        satellite: Satellite that runs the task.
        start: Start instant in seconds (>= 0).
        end: End instant in seconds (> start).
    """

    request_id: str
    task_index: int
    satellite: NodeId
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise InstanceError(
                f"execution of {self.request_id}, task {self.task_index}: "
                f"invalid interval [{self.start}, {self.end}]"
            )


@dataclass(frozen=True)
class Transfer:
    """Transmission of one data item of a request on one arc.

    Attributes:
        request_id: Identifier of the request.
        data_index: Index of the data item: 0 for the raw data, ``k`` for the output of task
            ``k - 1``.
        source: Tail node of the arc.
        target: Head node of the arc.
        start: Start instant in seconds (>= 0).
        end: End instant in seconds (> start).
    """

    request_id: str
    data_index: int
    source: NodeId
    target: NodeId
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise InstanceError(
                f"transfer of {self.request_id}, data {self.data_index} on "
                f"{self.source}->{self.target}: invalid interval [{self.start}, {self.end}]"
            )


@dataclass(frozen=True)
class Schedule:
    """A calendar: the task executions and the data transmissions of some requests.

    The same type is the committed calendar of an instance and the calendar of a solution:
    the calendar of the next epoch is the one of the solution, restricted to the requests
    not yet delivered (:meth:`restricted_to`).

    Attributes:
        executions: Executions of the tasks, at most one per task of a request.
        transfers: Transmissions of the data, at most one per data item of a request and
            arc.
    """

    executions: tuple[TaskExecution, ...] = ()
    transfers: tuple[Transfer, ...] = ()

    def __post_init__(self) -> None:
        tasks = [(e.request_id, e.task_index) for e in self.executions]
        if len(set(tasks)) != len(tasks):
            raise InstanceError("a task of a request is executed more than once")
        hops = [
            (t.request_id, t.data_index, t.source, t.target) for t in self.transfers
        ]
        if len(set(hops)) != len(hops):
            raise InstanceError(
                "a data item of a request uses the same arc more than once"
            )

    @property
    def request_ids(self) -> frozenset[str]:
        """Identifiers of the requests that appear in the calendar."""
        ids = {e.request_id for e in self.executions}
        return frozenset(ids | {t.request_id for t in self.transfers})

    def restricted_to(self, request_ids: Iterable[str]) -> Schedule:
        """The calendar of the requests in ``request_ids`` only, in the same order."""
        keep = frozenset(request_ids)
        return Schedule(
            tuple(e for e in self.executions if e.request_id in keep),
            tuple(t for t in self.transfers if t.request_id in keep),
        )


@dataclass(frozen=True)
class RequestOutcome:
    """Result for one request.

    Attributes:
        request_id: Identifier of the request.
        accepted: Whether the request is served.
        delivery_time: Instant in seconds at which the final output reaches the ground
            station. Set if and only if the request is accepted.
        rejection_reason: Why the request was rejected, when the solver gives one. Set only
            if the request is rejected.
    """

    request_id: str
    accepted: bool
    delivery_time: float | None = None
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        if self.accepted and self.delivery_time is None:
            raise InstanceError(
                f"request {self.request_id} is accepted but has no delivery time"
            )
        if not self.accepted and self.delivery_time is not None:
            raise InstanceError(
                f"request {self.request_id} is rejected but has a delivery time"
            )
        if self.accepted and self.rejection_reason is not None:
            raise InstanceError(
                f"request {self.request_id} is accepted but has a rejection reason"
            )


@dataclass(frozen=True)
class ObjectiveValue:
    """Value of the objective of a schedule, with its two terms kept apart.

    Attributes:
        primary: The lexicographically first term: the weighted rejections (hard) or the
            weighted tardiness (soft).
        response: The total response time of the accepted requests, delivery minus
            release, without the tie-break weight.
    """

    primary: float
    response: float

    def total(self, eps: float) -> float:
        """Single value minimised by a solver: ``primary + eps * response``."""
        return self.primary + eps * self.response


@dataclass(frozen=True)
class Solution:
    """The result of a solver for an instance.

    Attributes:
        schedule: Calendar of the accepted requests, committed ones included. The requests
            that appear in it are exactly the accepted ones.
        outcomes: One outcome per request of the instance.
        status: Status reported by the solver.
        objective: Value of the objective, with its terms, if the solver reports one.
    """

    schedule: Schedule
    outcomes: tuple[RequestOutcome, ...]
    status: SolveStatus
    objective: ObjectiveValue | None = None

    def __post_init__(self) -> None:
        ids = [o.request_id for o in self.outcomes]
        if len(set(ids)) != len(ids):
            raise InstanceError("a request has more than one outcome")
        accepted = frozenset(self.accepted_ids)
        if self.schedule.request_ids != accepted:
            raise InstanceError(
                "the calendar must hold exactly the accepted requests: "
                f"calendar {sorted(self.schedule.request_ids)}, "
                f"accepted {sorted(accepted)}"
            )

    def outcome(self, request_id: str) -> RequestOutcome:
        """Return the outcome of the request with identifier ``request_id``."""
        for outcome in self.outcomes:
            if outcome.request_id == request_id:
                return outcome
        raise InstanceError(f"no outcome for request {request_id!r}")

    @property
    def accepted_ids(self) -> tuple[str, ...]:
        """Identifiers of the accepted requests, in outcome order."""
        return tuple(o.request_id for o in self.outcomes if o.accepted)
