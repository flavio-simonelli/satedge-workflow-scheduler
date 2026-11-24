"""Translation of the instance documents into the domain objects, and back."""

from __future__ import annotations

import math

from satedge_workflow_scheduler.common.intervals import TimeWindow, WindowSet
from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.network import (
    AlwaysAvailable,
    Arc,
    Availability,
    Network,
    Plane,
    WindowedAvailability,
)
from satedge_workflow_scheduler.domain.nodes import GroundStation, Satellite
from satedge_workflow_scheduler.domain.requests import Request, Task
from satedge_workflow_scheduler.io.documents import (
    InstanceDoc,
    LinkDoc,
    PlaneDoc,
    RequestDoc,
    SatelliteDoc,
    WindowDoc,
)
from satedge_workflow_scheduler.io.errors import DocumentError


def instance_from_document(doc: InstanceDoc) -> ProblemInstance:
    """The instance described by ``doc``.

    Raises:
        DocumentError: If the document refers to a missing priority class, or lacks the
            horizon, or if the domain rejects the data.
    """
    network = Network(
        planes=tuple(_plane(p, doc.hpc_rate_gbps) for p in doc.topology.planes),
        ground_station=GroundStation(doc.ground_station),
        arcs=_arcs(doc),
    )
    requests = tuple(_request(doc, r) for r in doc.requests)
    return ProblemInstance(
        network=network,
        requests=requests,
        horizon=_horizon(doc, requests),
        plan_release=_plan_release(doc, network),
        name=doc.name,
    )


def _plane(plane: PlaneDoc, default_rate: float) -> Plane:
    return Plane(
        id=plane.id,
        satellites=tuple(_satellite(s, default_rate) for s in plane.satellites),
        com=plane.com,
    )


def _satellite(sat: SatelliteDoc, default_rate: float) -> Satellite:
    rate = sat.hpc_rate_gbps if sat.hpc_rate_gbps is not None else default_rate
    return Satellite(
        id=sat.id,
        plmm_capacity=sat.plmm_capacity_gb,
        hpc_rate=rate,
        hpc_windows=_windows(sat.hpc_windows),
    )


def _windows(windows: list[WindowDoc]) -> WindowSet:
    return WindowSet(tuple(TimeWindow(w.start_s, w.end_s) for w in windows))


def _arcs(doc: InstanceDoc) -> tuple[Arc, ...]:
    ground = doc.ground_station
    arcs: list[Arc] = []
    for link in doc.topology.links:
        availability = _availability(link)
        for source, target in _directions(link, ground):
            arcs.append(
                Arc(
                    source=source,
                    target=target,
                    capacity=link.capacity_gbps,
                    latency=link.latency_s,
                    availability=availability,
                )
            )
    return tuple(arcs)


def _directions(link: LinkDoc, ground: str) -> list[tuple[str, str]]:
    """The arcs of a link: two opposite arcs between satellites, one arc towards the ground."""
    if link.a == ground and link.b == ground:
        raise DocumentError(f"link from {ground} to itself")
    if ground == link.a:
        return [(link.b, link.a)]
    if ground == link.b:
        return [(link.a, link.b)]
    return [(link.a, link.b), (link.b, link.a)]


def _availability(link: LinkDoc) -> Availability:
    if link.windows is None:
        return AlwaysAvailable()
    return WindowedAvailability(_windows(link.windows))


def _request(doc: InstanceDoc, req: RequestDoc) -> Request:
    if req.deadline_s is not None:
        deadline = req.deadline_s
    else:
        delay = doc.priority_classes.get(req.priority)
        if delay is None:
            raise DocumentError(
                f"request {req.id}: no default delivery delay for {req.priority.value!r}, "
                "and no deadline_s"
            )
        deadline = req.release_s + delay.max_delivery_s
    return Request(
        id=req.id,
        priority=req.priority,
        origin=req.origin,
        release=req.release_s,
        deadline=deadline,
        raw_size=req.raw_size_gb,
        tasks=tuple(
            Task(execution_time=t.execution_s, output_size=t.output_size_gb)
            for t in req.tasks
        ),
    )


def _horizon(doc: InstanceDoc, requests: tuple[Request, ...]) -> float:
    if doc.horizon_s is not None:
        return doc.horizon_s
    if not requests:
        raise DocumentError("horizon_s is required when the instance has no requests")
    return max(r.deadline for r in requests)


def _plan_release(doc: InstanceDoc, network: Network) -> dict[str, float]:
    release: dict[str, float] = {s.id: 0.0 for s in network.satellites}
    for node, value in doc.plan_release_s.items():
        release[node] = math.inf if value is None else value
    return release
