"""Documents of the YAML and JSON files: the shape of each file, checked on reading."""

from satedge_workflow_scheduler.io.documents.base import (
    SCHEMA_VERSION,
    check_schema_version,
)
from satedge_workflow_scheduler.io.documents.contact import (
    ContactLinkDoc,
    ContactPlanDoc,
)
from satedge_workflow_scheduler.io.documents.instance import (
    InstanceDoc,
    PriorityClassDoc,
    RequestDoc,
    TaskDoc,
)
from satedge_workflow_scheduler.io.documents.online import CommittedDoc, OnlineDoc
from satedge_workflow_scheduler.io.documents.result import (
    ArcResultDoc,
    HpcResultDoc,
    ObjectiveDoc,
    PlmmResultDoc,
    RequestResultDoc,
    ResultDoc,
    StatisticsDoc,
    ViolationDoc,
)
from satedge_workflow_scheduler.io.documents.schedule import ExecutionDoc, TransferDoc
from satedge_workflow_scheduler.io.documents.solver import (
    LimitsDoc,
    MilpDoc,
    SequentialSettingsDoc,
    SolverDoc,
    StrengtheningDoc,
)
from satedge_workflow_scheduler.io.documents.topology import (
    ExplicitTopologyDoc,
    LinkDoc,
    LinkParamsDoc,
    PeriodicHpcDoc,
    PlaneDoc,
    RingTopologyDoc,
    SatelliteDoc,
    TopologyDoc,
    WindowDoc,
)

__all__ = [
    "SCHEMA_VERSION",
    "ArcResultDoc",
    "CommittedDoc",
    "ContactLinkDoc",
    "ContactPlanDoc",
    "ExecutionDoc",
    "ExplicitTopologyDoc",
    "HpcResultDoc",
    "InstanceDoc",
    "LimitsDoc",
    "LinkDoc",
    "LinkParamsDoc",
    "MilpDoc",
    "ObjectiveDoc",
    "OnlineDoc",
    "PeriodicHpcDoc",
    "PlaneDoc",
    "PlmmResultDoc",
    "PriorityClassDoc",
    "RequestDoc",
    "RequestResultDoc",
    "ResultDoc",
    "RingTopologyDoc",
    "SatelliteDoc",
    "SequentialSettingsDoc",
    "SolverDoc",
    "StatisticsDoc",
    "StrengtheningDoc",
    "TaskDoc",
    "TopologyDoc",
    "TransferDoc",
    "ViolationDoc",
    "WindowDoc",
    "check_schema_version",
]
