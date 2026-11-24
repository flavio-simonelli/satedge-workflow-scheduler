"""Document of a result file."""

from __future__ import annotations

from pydantic import Field

from satedge_workflow_scheduler.domain.requests import Priority
from satedge_workflow_scheduler.io.documents.base import SCHEMA_VERSION, Document
from satedge_workflow_scheduler.io.documents.schedule import ExecutionDoc, TransferDoc


class RequestResultDoc(Document):
    """Outcome and timing of one request."""

    id: str
    priority: Priority
    committed: bool
    accepted: bool
    rejection_reason: str | None = None
    release_s: float
    limit_s: float
    delivery_s: float | None = None
    response_s: float | None = None
    tardiness_s: float | None = None


class ObjectiveDoc(Document):
    """The two terms of the objective."""

    primary: float
    response: float


class StatisticsDoc(Document):
    """What the solve cost; a field is ``null`` if the method does not report it."""

    method: str
    wall_time_s: float | None = None
    solver_time_s: float | None = None
    bound: float | None = None
    gap: float | None = None
    nodes: float | None = None
    rounds: int = 1
    binary_variables: int | None = None
    constraints: int | None = None


class PlmmResultDoc(Document):
    """Peak memory load of the PLMM of a satellite."""

    satellite: str
    capacity_gb: float
    peak_gb: float
    peak_time_s: float | None = None


class HpcResultDoc(Document):
    """Use of the computing windows of a satellite."""

    satellite: str
    available_s: float
    busy_s: float
    utilisation: float


class ArcResultDoc(Document):
    """Use of an arc of the network."""

    source: str
    target: str
    busy_s: float
    volume_gb: float


class ViolationDoc(Document):
    """A rule that the solution breaks."""

    constraint: str
    request: str | None = None
    message: str


class ResultDoc(Document):
    """The result of one solve, as written to the JSON file.

    The schedule and the outcomes are enough to rebuild the solution; the metrics are derived
    from them and are written for the reader of the file.
    """

    schema_version: str = SCHEMA_VERSION
    instance: str
    deadline: str
    status: str
    objective: ObjectiveDoc | None = None
    statistics: StatisticsDoc
    requests: list[RequestResultDoc] = Field(default_factory=list)
    executions: list[ExecutionDoc] = Field(default_factory=list)
    transfers: list[TransferDoc] = Field(default_factory=list)
    plmm: list[PlmmResultDoc] = Field(default_factory=list)
    hpc: list[HpcResultDoc] = Field(default_factory=list)
    arcs: list[ArcResultDoc] = Field(default_factory=list)
    violations: list[ViolationDoc] = Field(default_factory=list)
