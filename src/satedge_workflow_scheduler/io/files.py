"""Loading of the instance, epoch-state and solver files: each is read, then translated."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

from satedge_workflow_scheduler.common.errors import SchedulerError
from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.nodes import NodeId
from satedge_workflow_scheduler.domain.solution import Schedule
from satedge_workflow_scheduler.io.documents import (
    ContactPlanDoc,
    InstanceDoc,
    OnlineDoc,
    SolverDoc,
)
from satedge_workflow_scheduler.io.errors import DocumentError
from satedge_workflow_scheduler.io.reading import read_document
from satedge_workflow_scheduler.io.results import read_result
from satedge_workflow_scheduler.io.translate import (
    apply_contact_plan,
    explicit_document,
    instance_from_document,
    schedule_from_documents,
    solver_config_from_document,
)
from satedge_workflow_scheduler.solvers.milp.config import MilpConfig
from satedge_workflow_scheduler.solvers.sequential.config import SequentialConfig


def load_instance(
    path: str | Path,
    *,
    contact_plan: str | Path | None = None,
    online: str | Path | None = None,
) -> ProblemInstance:
    """The instance described by the YAML file at ``path``.

    ``contact_plan`` gives the visibility windows of the windowed links, and ``online`` the state
    of a planning epoch: the plan release of the satellites it lists, and the committed calendar.
    The name of the instance is the name of the file, unless the file gives one.

    Raises:
        DocumentError: If a file cannot be read, does not match the format, or describes an
            instance that the domain rejects. The message names the file and the field.
    """
    file = Path(path)
    doc = read_document(file, InstanceDoc)
    if not doc.name:
        doc = doc.model_copy(update={"name": file.stem})
    plan = (
        None
        if contact_plan is None
        else read_document(Path(contact_plan), ContactPlanDoc)
    )
    state = None if online is None else load_online_state(online)
    try:
        doc = explicit_document(doc)
        if plan is not None:
            doc = apply_contact_plan(doc, plan)
        instance = instance_from_document(doc)
        if state is not None:
            release, committed = state
            instance = replace(
                instance,
                plan_release={**instance.plan_release, **release},
                committed=committed,
            )
    except (SchedulerError, ValueError) as error:
        raise DocumentError(f"{file}: {error}") from error
    return instance


def load_online_state(path: str | Path) -> tuple[dict[NodeId, float], Schedule]:
    """The plan release of the satellites and the committed calendar of the file at ``path``.

    A committed result is read from the file it names, relative to this file; the calendar is
    restricted to the requests the file lists, if it lists any.

    Raises:
        DocumentError: If a file cannot be read or does not match its format.
    """
    file = Path(path)
    doc = read_document(file, OnlineDoc)
    release = {
        node: math.inf if value is None else value
        for node, value in doc.plan_release_s.items()
    }
    committed = doc.committed
    if committed.result is None:
        return release, schedule_from_documents(
            committed.executions, committed.transfers
        )
    schedule = read_result(file.parent / committed.result).schedule
    if committed.requests is not None:
        schedule = schedule.restricted_to(committed.requests)
    return release, schedule


def load_solver_config(path: str | Path) -> MilpConfig | SequentialConfig:
    """The solver configuration described by the YAML file at ``path``.

    Raises:
        DocumentError: If the file cannot be read, does not match the format, or describes an
            invalid configuration.
    """
    file = Path(path)
    doc = read_document(file, SolverDoc)
    try:
        return solver_config_from_document(doc)
    except (SchedulerError, ValueError) as error:
        raise DocumentError(f"{file}: {error}") from error
