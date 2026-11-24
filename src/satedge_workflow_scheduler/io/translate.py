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
from satedge_workflow_scheduler.domain.solution import Schedule, TaskExecution, Transfer
from satedge_workflow_scheduler.io.documents import (
    ContactPlanDoc,
    ExecutionDoc,
    InstanceDoc,
    LinkDoc,
    PlaneDoc,
    RequestDoc,
    SatelliteDoc,
    SolverDoc,
    TransferDoc,
    WindowDoc,
)
from satedge_workflow_scheduler.io.errors import DocumentError
from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.solvers.milp.backends.ports import SolveParams
from satedge_workflow_scheduler.solvers.milp.config import (
    LazyGroup,
    Level,
    MilpConfig,
    PlmmFormulation,
    Strengthening,
)
from satedge_workflow_scheduler.solvers.sequential.config import SequentialConfig


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


def solver_config_from_document(doc: SolverDoc) -> MilpConfig | SequentialConfig:
    """The configuration that ``doc`` describes.

    Raises:
        DocumentError: If the MILP level is not one of the levels, or the document lacks the
            section of its method.
    """
    deadline = DeadlineMode(doc.deadline)
    if doc.method == "sequential":
        return SequentialConfig(
            deadline=deadline,
            max_backtracks=doc.sequential.max_backtracks,
            verify=doc.sequential.verify,
        )
    if doc.milp is None:
        raise DocumentError("method 'milp' needs a 'milp' section")
    milp = doc.milp
    try:
        level = Level(milp.level)
    except ValueError as error:
        raise DocumentError(
            f"milp.level must be one of 3, 4, 5, 6, got {milp.level}"
        ) from error
    return MilpConfig(
        level=level,
        deadline=deadline,
        reductions=milp.reductions,
        strengthening=Strengthening(
            arrival_without_big_m=milp.strengthening.arrival_without_big_m,
            contact_capacity=milp.strengthening.contact_capacity,
        ),
        lazy=frozenset(LazyGroup(group) for group in milp.lazy),
        plmm_formulation=PlmmFormulation(milp.plmm_formulation),
        polish=milp.polish,
        warm_start=milp.warm_start,
        solve=SolveParams(
            time_limit=doc.limits.time_limit_s,
            mip_gap=doc.limits.mip_gap,
            threads=doc.limits.threads,
            memory_limit_gb=doc.limits.memory_limit_gb,
            verbose=doc.verbose,
        ),
    )


def apply_contact_plan(doc: InstanceDoc, plan: ContactPlanDoc) -> InstanceDoc:
    """The instance document with the windows of ``plan`` on its links.

    A link that the plan names becomes windowed. The plan may only name links of the topology,
    each once, and only links that the instance does not window already.

    Raises:
        DocumentError: If the plan names a missing link, names a link twice, or a link that is
            windowed in the instance already.
    """
    links = list(doc.topology.links)
    position = {frozenset((link.a, link.b)): i for i, link in enumerate(links)}
    named: set[frozenset[str]] = set()
    for entry in plan.links:
        key = frozenset((entry.a, entry.b))
        label = f"{entry.a}-{entry.b}"
        if key not in position:
            raise DocumentError(f"contact plan: the topology has no link {label}")
        if key in named:
            raise DocumentError(f"contact plan: link {label} is listed twice")
        index = position[key]
        if links[index].windows is not None:
            raise DocumentError(
                f"link {label} is windowed in the instance: remove its windows or its "
                "contact plan entry"
            )
        links[index] = links[index].model_copy(update={"windows": entry.windows})
        named.add(key)
    topology = doc.topology.model_copy(update={"links": links})
    return doc.model_copy(update={"topology": topology})


def schedule_from_documents(
    executions: list[ExecutionDoc], transfers: list[TransferDoc]
) -> Schedule:
    """The calendar described by execution and transfer documents."""
    return Schedule(
        executions=tuple(
            TaskExecution(
                request_id=e.request,
                task_index=e.task,
                satellite=e.satellite,
                start=e.start_s,
                end=e.end_s,
            )
            for e in executions
        ),
        transfers=tuple(
            Transfer(
                request_id=t.request,
                data_index=t.data,
                source=t.source,
                target=t.target,
                start=t.start_s,
                end=t.end_s,
            )
            for t in transfers
        ),
    )
