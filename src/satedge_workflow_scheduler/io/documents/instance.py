"""Document of an instance file: its requests and its topology."""

from __future__ import annotations

from pydantic import Field, field_validator

from satedge_workflow_scheduler.domain.requests import Priority
from satedge_workflow_scheduler.io.documents.base import (
    SCHEMA_VERSION,
    Document,
    check_schema_version,
)
from satedge_workflow_scheduler.io.documents.topology import TopologyDoc


class PriorityClassDoc(Document):
    """Default delivery delay of a priority class, in seconds after the release."""

    max_delivery_s: float


class TaskDoc(Document):
    """One task of the pipeline of a request."""

    execution_s: float
    output_size_gb: float


class RequestDoc(Document):
    """A request: a pipeline of tasks on the raw data of an origin satellite."""

    id: str
    priority: Priority
    origin: str
    release_s: float
    deadline_s: float | None = None
    raw_size_gb: float
    tasks: list[TaskDoc] = Field(min_length=1)


def _default_priority_classes() -> dict[Priority, PriorityClassDoc]:
    return {
        Priority.URGENT: PriorityClassDoc(max_delivery_s=7200.0),
        Priority.NON_URGENT: PriorityClassDoc(max_delivery_s=25200.0),
    }


class InstanceDoc(Document):
    """A whole instance file.

    Attributes:
        schema_version: Version of the format, ``major.minor.patch``.
        name: Name of the instance; empty means the name of the file.
        ground_station: Identifier of the ground station.
        hpc_rate_gbps: HPC rate of the satellites that do not give their own.
        horizon_s: Scheduling horizon; required without requests, otherwise the latest deadline.
        priority_classes: Default delivery delay per priority, used by requests without a
            ``deadline_s``. If given, it replaces the defaults entirely.
        topology: Planes and links.
        requests: The requests, in FIFO order.
        plan_release_s: Plan release per satellite, in seconds; ``null`` means never (infinite).
            Satellites not listed receive the plan at time 0.
    """

    schema_version: str = SCHEMA_VERSION
    name: str = ""
    ground_station: str = "GS"
    hpc_rate_gbps: float = 1.0
    horizon_s: float | None = None
    priority_classes: dict[Priority, PriorityClassDoc] = Field(
        default_factory=_default_priority_classes
    )
    topology: TopologyDoc
    requests: list[RequestDoc] = Field(default_factory=list)
    plan_release_s: dict[str, float | None] = Field(default_factory=dict)

    @field_validator("schema_version")
    @classmethod
    def _same_major(cls, value: str) -> str:
        return check_schema_version(value)
