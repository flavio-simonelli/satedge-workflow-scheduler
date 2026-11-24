"""Writing of instances and of their epoch state as YAML files: the inverse of the readers.

The written instance is always explicit: its windows are written in full, so the file needs no
contact plan to be read back.
"""

from __future__ import annotations

import math
from pathlib import Path

import yaml

from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.network import (
    AlwaysAvailable,
    Arc,
    Availability,
)
from satedge_workflow_scheduler.domain.nodes import Satellite
from satedge_workflow_scheduler.domain.requests import Request
from satedge_workflow_scheduler.io.documents import (
    SCHEMA_VERSION,
    CommittedDoc,
    ExecutionDoc,
    ExplicitTopologyDoc,
    InstanceDoc,
    LinkDoc,
    OnlineDoc,
    PlaneDoc,
    RequestDoc,
    SatelliteDoc,
    TaskDoc,
    TransferDoc,
    WindowDoc,
)
from satedge_workflow_scheduler.io.errors import DocumentError


def instance_document(instance: ProblemInstance) -> InstanceDoc:
    """The explicit document of ``instance``.

    Raises:
        DocumentError: If two opposite arcs of a link differ, which the format cannot express.
    """
    network = instance.network
    planes = [
        PlaneDoc(
            id=plane.id,
            com=plane.com,
            satellites=[_satellite(s) for s in plane.satellites],
        )
        for plane in network.planes
    ]
    return InstanceDoc(
        schema_version=SCHEMA_VERSION,
        name=instance.name,
        ground_station=network.ground_station.id,
        horizon_s=instance.horizon,
        topology=ExplicitTopologyDoc(
            kind="explicit", planes=planes, links=_links(network.arcs)
        ),
        requests=[_request(r) for r in instance.requests],
        plan_release_s=_plan_release(instance),
    )


def write_instance(path: str | Path, instance: ProblemInstance) -> None:
    """Write ``instance`` as a YAML file at ``path``, creating its folder."""
    document = instance_document(instance)
    _write_yaml(path, document.model_dump(mode="json"))


def online_document(instance: ProblemInstance) -> OnlineDoc:
    """The epoch state of ``instance``: its plan release and its committed calendar, inline."""
    committed = instance.committed
    return OnlineDoc(
        schema_version=SCHEMA_VERSION,
        plan_release_s=_plan_release(instance),
        committed=CommittedDoc(
            executions=[
                ExecutionDoc(
                    request=e.request_id,
                    task=e.task_index,
                    satellite=e.satellite,
                    start_s=e.start,
                    end_s=e.end,
                )
                for e in committed.executions
            ],
            transfers=[
                TransferDoc(
                    request=t.request_id,
                    data=t.data_index,
                    source=t.source,
                    target=t.target,
                    start_s=t.start,
                    end_s=t.end,
                )
                for t in committed.transfers
            ],
        ),
    )


def write_online_state(path: str | Path, instance: ProblemInstance) -> None:
    """Write the epoch state of ``instance`` as a YAML file at ``path``."""
    _write_yaml(path, online_document(instance).model_dump(mode="json"))


def _satellite(sat: Satellite) -> SatelliteDoc:
    return SatelliteDoc(
        id=sat.id,
        plmm_capacity_gb=sat.plmm_capacity,
        hpc_rate_gbps=sat.hpc_rate,
        hpc_windows=[
            WindowDoc(start_s=w.start, end_s=w.end) for w in sat.hpc_windows.windows
        ],
    )


def _links(arcs: tuple[Arc, ...]) -> list[LinkDoc]:
    """One link per pair of nodes: opposite arcs are one link, and an arc to ground is one too."""
    groups: dict[frozenset[str], list[Arc]] = {}
    for arc in arcs:
        groups.setdefault(frozenset((arc.source, arc.target)), []).append(arc)
    links = []
    for group in groups.values():
        first = group[0]
        for other in group[1:]:
            if (
                other.capacity != first.capacity
                or other.latency != first.latency
                or _windows_of(other.availability) != _windows_of(first.availability)
            ):
                raise DocumentError(
                    f"arcs {first.source}-{first.target} and {other.source}-{other.target} "
                    "differ: the file format writes one link for both directions"
                )
        links.append(
            LinkDoc(
                a=first.source,
                b=first.target,
                capacity_gbps=first.capacity,
                latency_s=first.latency,
                windows=_windows_of(first.availability),
            )
        )
    return links


def _windows_of(availability: Availability) -> list[WindowDoc] | None:
    if isinstance(availability, AlwaysAvailable):
        return None
    return [
        WindowDoc(start_s=w.start, end_s=w.end) for w in availability.windows.windows
    ]


def _request(request: Request) -> RequestDoc:
    return RequestDoc(
        id=request.id,
        priority=request.priority,
        origin=request.origin,
        release_s=request.release,
        deadline_s=request.deadline,
        raw_size_gb=request.raw_size,
        tasks=[
            TaskDoc(execution_s=t.execution_time, output_size_gb=t.output_size)
            for t in request.tasks
        ],
    )


def _plan_release(instance: ProblemInstance) -> dict[str, float | None]:
    return {
        node: None if math.isinf(value) else value
        for node, value in instance.plan_release.items()
    }


def _write_yaml(path: str | Path, data: object) -> None:
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    file.write_text(text, encoding="utf-8")
