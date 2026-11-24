"""Documents of the YAML files: the shape of the input, checked before the instance is built.

The documents mirror the YAML files field by field. Semantic checks (positive capacities,
disjoint windows, known nodes) are left to the domain, which builds the instance from them.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from satedge_workflow_scheduler.domain.requests import Priority

SCHEMA_VERSION = "1.0.0"
"""Version of the document format; the major number must match."""


def check_schema_version(value: str) -> str:
    """Return ``value`` if this reader understands its major version, otherwise raise."""
    if value.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
        raise ValueError(
            f"unsupported schema version {value!r}: this reader understands "
            f"{SCHEMA_VERSION!r} and its minor versions"
        )
    return value


class _Document(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WindowDoc(_Document):
    """A window of time, in seconds."""

    start_s: float
    end_s: float


class SatelliteDoc(_Document):
    """A satellite: its PLMM, and its computing windows."""

    id: str
    plmm_capacity_gb: float
    hpc_rate_gbps: float | None = None
    hpc_windows: list[WindowDoc] = Field(default_factory=list)


class PlaneDoc(_Document):
    """An orbital plane, with the satellite that is its communication node."""

    id: int
    com: str
    satellites: list[SatelliteDoc] = Field(min_length=1)


class LinkDoc(_Document):
    """A link between two nodes; ``windows`` makes it available only in those windows."""

    a: str
    b: str
    capacity_gbps: float = 1.0
    latency_s: float = 0.015
    windows: list[WindowDoc] | None = None


class TopologyDoc(_Document):
    """The explicit topology: planes and links."""

    kind: Literal["explicit"] = "explicit"
    planes: list[PlaneDoc] = Field(min_length=1)
    links: list[LinkDoc] = Field(default_factory=list)


class PriorityClassDoc(_Document):
    """Default delivery delay of a priority class, in seconds after the release."""

    max_delivery_s: float


class TaskDoc(_Document):
    """One task of the pipeline of a request."""

    execution_s: float
    output_size_gb: float


class RequestDoc(_Document):
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


class InstanceDoc(_Document):
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


class ExecutionDoc(_Document):
    """A task run on a satellite."""

    request: str
    task: int
    satellite: str
    start_s: float
    end_s: float


class TransferDoc(_Document):
    """A transmission of one data item on an arc."""

    request: str
    data: int
    source: str
    target: str
    start_s: float
    end_s: float


class RequestResultDoc(_Document):
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


class ObjectiveDoc(_Document):
    """The two terms of the objective."""

    primary: float
    response: float


class StatisticsDoc(_Document):
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


class PlmmResultDoc(_Document):
    """Peak memory load of the PLMM of a satellite."""

    satellite: str
    capacity_gb: float
    peak_gb: float
    peak_time_s: float | None = None


class HpcResultDoc(_Document):
    """Use of the computing windows of a satellite."""

    satellite: str
    available_s: float
    busy_s: float
    utilisation: float


class ArcResultDoc(_Document):
    """Use of an arc of the network."""

    source: str
    target: str
    busy_s: float
    volume_gb: float


class ViolationDoc(_Document):
    """A rule that the solution breaks."""

    constraint: str
    request: str | None = None
    message: str


class ResultDoc(_Document):
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


class StrengtheningDoc(_Document):
    """Optional strengthening cuts of the MILP."""

    arrival_without_big_m: bool = False
    contact_capacity: bool = False


class MilpDoc(_Document):
    """Settings of the MILP: the level of the model and its simplifications."""

    level: int
    reductions: bool = True
    strengthening: StrengtheningDoc = Field(default_factory=StrengtheningDoc)
    lazy: list[str] = Field(default_factory=list)
    plmm_formulation: Literal["flow", "events"] = "flow"
    polish: bool = True
    warm_start: bool = False


class SequentialSettingsDoc(_Document):
    """Settings of the sequential heuristic."""

    max_backtracks: int = 100
    verify: bool = True


class LimitsDoc(_Document):
    """Limits of the MILP solve. A file of the sequential method must not set them."""

    time_limit_s: float | None = None
    mip_gap: float = 1e-6
    threads: int | None = None
    memory_limit_gb: float | None = None


class SolverDoc(_Document):
    """The solver file: which method to run, and its settings.

    Attributes:
        schema_version: Version of the format, ``major.minor.patch``.
        method: ``milp`` needs the ``milp`` section; ``sequential`` uses ``sequential``.
        deadline: ``hard`` rejects the requests that miss their deadline; ``soft`` serves them
            all and penalises the lateness.
        milp: Settings of the MILP.
        sequential: Settings of the sequential heuristic.
        limits: Limits of the MILP solve.
        verbose: Whether the solver writes its log.
    """

    schema_version: str = SCHEMA_VERSION
    method: Literal["milp", "sequential"]
    deadline: Literal["hard", "soft"] = "hard"
    milp: MilpDoc | None = None
    sequential: SequentialSettingsDoc = Field(default_factory=SequentialSettingsDoc)
    limits: LimitsDoc = Field(default_factory=LimitsDoc)
    verbose: bool = False

    @field_validator("schema_version")
    @classmethod
    def _same_major(cls, value: str) -> str:
        return check_schema_version(value)

    @model_validator(mode="after")
    def _method_has_its_section(self) -> SolverDoc:
        if self.method == "milp" and self.milp is None:
            raise ValueError("method 'milp' needs a 'milp' section")
        if self.method == "sequential" and "limits" in self.model_fields_set:
            raise ValueError(
                "limits apply to method 'milp' only: the sequential heuristic has no solve "
                "to limit, so remove the 'limits' section"
            )
        return self


class ContactLinkDoc(_Document):
    """Visibility windows of the link between two nodes."""

    a: str
    b: str
    windows: list[WindowDoc]


class ContactPlanDoc(_Document):
    """Visibility windows of the links of a topology, from a separate file."""

    schema_version: str = SCHEMA_VERSION
    links: list[ContactLinkDoc] = Field(default_factory=list)

    @field_validator("schema_version")
    @classmethod
    def _same_major(cls, value: str) -> str:
        return check_schema_version(value)


class CommittedDoc(_Document):
    """The committed calendar: a saved result, or the executions and transfers written here.

    Attributes:
        result: Path of a result file, relative to this file; its calendar is committed.
        requests: With ``result``, only these requests are committed; all of them if omitted.
        executions: Inline executions, instead of ``result``.
        transfers: Inline transfers, instead of ``result``.
    """

    result: str | None = None
    requests: list[str] | None = None
    executions: list[ExecutionDoc] = Field(default_factory=list)
    transfers: list[TransferDoc] = Field(default_factory=list)

    @model_validator(mode="after")
    def _one_source(self) -> CommittedDoc:
        inline = bool(self.executions or self.transfers)
        if self.result is not None and inline:
            raise ValueError("give either 'result' or inline executions and transfers")
        if self.requests is not None and self.result is None:
            raise ValueError("'requests' filters a 'result': give 'result' too")
        return self


class OnlineDoc(_Document):
    """The state of a planning epoch: when the plan reaches each satellite, and what is fixed.

    Attributes:
        schema_version: Version of the format, ``major.minor.patch``.
        plan_release_s: Plan release of the satellites it lists; ``null`` means never.
        committed: The calendar of the requests fixed by the previous plan.
    """

    schema_version: str = SCHEMA_VERSION
    plan_release_s: dict[str, float | None] = Field(default_factory=dict)
    committed: CommittedDoc = Field(default_factory=CommittedDoc)

    @field_validator("schema_version")
    @classmethod
    def _same_major(cls, value: str) -> str:
        return check_schema_version(value)
