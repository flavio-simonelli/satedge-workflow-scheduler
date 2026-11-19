"""Solution of a scheduling instance: task executions, data transfers and request outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from satedge_workflow_scheduler.common.errors import InstanceError
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
        position: Index of the task in the pipeline of the request.
        satellite: Satellite that runs the task.
        start: Start instant in seconds (>= 0).
        end: End instant in seconds (> start).
    """

    request_id: str
    position: int
    satellite: NodeId
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise InstanceError(
                f"execution of {self.request_id}, task {self.position}: "
                f"invalid interval [{self.start}, {self.end}]"
            )


@dataclass(frozen=True)
class Transfer:
    """Transmission of one data item of a request on one arc.

    Attributes:
        request_id: Identifier of the request.
        position: Index of the data item: 0 for the raw data, ``k`` for the output of task
            ``k - 1``.
        source: Tail node of the arc.
        target: Head node of the arc.
        start: Start instant in seconds (>= 0).
        end: End instant in seconds (> start).
    """

    request_id: str
    position: int
    source: NodeId
    target: NodeId
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise InstanceError(
                f"transfer of {self.request_id}, data {self.position} on "
                f"{self.source}->{self.target}: invalid interval [{self.start}, {self.end}]"
            )


@dataclass(frozen=True)
class RequestOutcome:
    """Result for one request.

    Attributes:
        request_id: Identifier of the request.
        accepted: Whether the request is served.
        delivery_time: Instant in seconds at which the final output reaches the ground
            station. Set if and only if the request is accepted.
    """

    request_id: str
    accepted: bool
    delivery_time: float | None = None

    def __post_init__(self) -> None:
        if self.accepted and self.delivery_time is None:
            raise InstanceError(
                f"request {self.request_id} is accepted but has no delivery time"
            )
        if not self.accepted and self.delivery_time is not None:
            raise InstanceError(
                f"request {self.request_id} is rejected but has a delivery time"
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
    """A complete schedule for an instance.

    Attributes:
        outcomes: One outcome per request of the instance.
        executions: Executions of the tasks of the accepted requests.
        transfers: Transmissions of the data of the accepted requests.
        status: Status reported by the solver.
        objective: Value of the objective, with its terms, if the solver reports one.
    """

    outcomes: tuple[RequestOutcome, ...]
    executions: tuple[TaskExecution, ...]
    transfers: tuple[Transfer, ...]
    status: SolveStatus
    objective: ObjectiveValue | None = None

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
