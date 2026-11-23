"""Candidates of a task: where and when it can run, in earliest-finish (EFT) order.

A candidate is a satellite, a route of the input to it and one of its computing windows. The task
starts at the first free instant of the window after the input has arrived and the plan has
reached the satellite. Only the windows compatible with the task are tried
(:meth:`RequestBounds.compatible_windows`), and the task must end by its latest completion
``LC_{i,s}``: a later end would leave no time to deliver the request, even alone in the network.
It must also end by the ``hold_until`` of the route: the input stays on the satellite until the
task ends, inside the free interval of the PLMM it arrived in.

The order is the EFT rule: earlier end first, then the satellite reached with fewer hops, then the
satellite that comes first in the network, then the earlier window.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from satedge_workflow_scheduler.common.intervals import TIME_TOLERANCE, WindowSet
from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.nodes import NodeId, Satellite
from satedge_workflow_scheduler.domain.requests import Request
from satedge_workflow_scheduler.scheduling.bounds import RequestBounds
from satedge_workflow_scheduler.scheduling.windows import earliest_finish
from satedge_workflow_scheduler.solvers.sequential.reservations import Reservations
from satedge_workflow_scheduler.solvers.sequential.routing.ports import Route


@dataclass(frozen=True, slots=True, order=True)
class TaskCandidate:
    """A place and a time for a task, with the route of its input.

    The candidates are ordered by EFT: the fields that take part in the comparison are declared in
    the order of the criteria, and the others are left out of it (``compare=False``). So sorting
    the candidates sorts them by the criteria, in this order:

    1. ``end``: the task ends first;
    2. ``hops``: the input reaches the satellite with fewer transmissions;
    3. ``position``: the satellite comes first in the network;
    4. ``window_start``: the earlier computing window.

    Attributes:
        end: End of the task, once its output is written.
        hops: Number of transmissions of the route of the input.
        position: Index of the satellite in the network.
        window_start: Start of the computing window the task runs in.
        satellite: The satellite that runs the task.
        start: Start of the task.
        input_route: How the input reaches ``satellite``.
    """

    end: float
    hops: int
    position: int
    window_start: float
    satellite: Satellite = field(compare=False)
    start: float = field(compare=False)
    input_route: Route = field(compare=False)


def task_candidates(
    instance: ProblemInstance,
    reservations: Reservations,
    request: Request,
    bounds: RequestBounds,
    task_index: int,
    input_routes: Mapping[NodeId, Sequence[Route]],
    eps: float = TIME_TOLERANCE,
) -> list[TaskCandidate]:
    """The candidates of task ``task_index`` of ``request``, in EFT order.

    Args:
        instance: The instance, for the satellites and their plan release.
        reservations: What is already reserved: the HPC of a satellite runs one task at a time.
        request: The request of the task.
        bounds: The a priori bounds of the request.
        task_index: The task.
        input_routes: The routes of the input to each satellite it can reach. A satellite may
            have several, one per free interval of its PLMM: the task must end by the
            ``hold_until`` of the route it uses.
        eps: Tolerance for comparisons between instants.
    """
    candidates: list[TaskCandidate] = []
    for position, satellite in enumerate(instance.network.satellites):
        duration = request.hpc_duration(task_index, satellite)
        completion = bounds.latest_completion(task_index, satellite.id)
        busy = reservations.hpc_busy(satellite.id)
        windows = bounds.compatible_windows(task_index, satellite.id)
        for route in input_routes.get(satellite.id, ()):
            ready = max(route.arrival, instance.release_of(satellite.id))
            if math.isinf(ready):
                continue
            # the input stays on the satellite until the end of the task
            latest_end = min(completion, route.hold_until)
            for window in windows:
                end = earliest_finish(
                    WindowSet((window,)), ready, duration, busy=busy, eps=eps
                )
                if end is None or end > latest_end + eps:
                    continue
                candidates.append(
                    TaskCandidate(
                        end=end,
                        hops=len(route.hops),
                        position=position,
                        window_start=window.start,
                        satellite=satellite,
                        start=end - duration,
                        input_route=route,
                    )
                )
    return sorted(candidates)
