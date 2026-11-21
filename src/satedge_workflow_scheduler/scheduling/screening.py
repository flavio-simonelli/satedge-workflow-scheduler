"""Screening of the requests: what can never be scheduled, whatever the other requests do.

These are conditions on the data of the model, not errors: an instance that violates them is
valid, and the requests concerned are excluded a priori. With hard deadlines they are
rejected; with soft deadlines, where no request can be rejected, the instance is infeasible.
"""

from __future__ import annotations

from satedge_workflow_scheduler.common.intervals import TIME_TOLERANCE
from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.network import Network
from satedge_workflow_scheduler.domain.nodes import NodeId
from satedge_workflow_scheduler.domain.requests import Request
from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode, delivery_limit


def executable_satellites(
    request: Request, task_index: int, network: Network
) -> tuple[NodeId, ...]:
    """Satellites with a computing window long enough for task ``task_index`` of ``request``.

    The task occupies the HPC of ``s`` for :meth:`Request.hpc_duration`, which depends on
    the HPC rate of ``s``: it can run on ``s`` only if one window of ``s`` is at least that
    long. On the other satellites the task can never run.
    """
    result: list[NodeId] = []
    for satellite in network.satellites:
        duration = request.hpc_duration(task_index, satellite)
        if any(w.duration >= duration - TIME_TOLERANCE for w in satellite.hpc_windows):
            result.append(satellite.id)
    return tuple(result)


def unrealizable_reason(request: Request, network: Network, limit: float) -> str | None:
    """Why ``request`` can never be delivered by ``limit``, or ``None`` if it may be.

    Two conditions are checked, ignoring the network delays and the other requests:

    - every task has at least one satellite with a window long enough for it;
    - the pipeline can end by ``limit`` even running its tasks one after the other with no
      wait, each on the satellite where it is shortest: ``b_r + Σ_i min_s L̄_{i,s} <= limit``.
    """
    shortest_total = 0.0
    for task_index in range(len(request.tasks)):
        satellites = executable_satellites(request, task_index, network)
        if not satellites:
            return f"task {task_index} fits in no computing window"
        shortest_total += min(
            request.hpc_duration(task_index, network.satellite(s)) for s in satellites
        )
    earliest_end = request.release + shortest_total
    if earliest_end > limit + TIME_TOLERANCE:
        return (
            f"the pipeline cannot end before {earliest_end} s, "
            f"after the limit of {limit} s"
        )
    return None


def unrealizable_requests(
    instance: ProblemInstance, mode: DeadlineMode
) -> dict[str, str]:
    """The new requests of ``instance`` that can never be delivered, with the reason.

    The limit of each request is the one of ``mode`` (:func:`delivery_limit`). Committed
    requests are not screened: their calendar is already fixed.
    """
    result: dict[str, str] = {}
    for request in instance.new_requests:
        limit = delivery_limit(request, mode, instance.horizon)
        reason = unrealizable_reason(request, instance.network, limit)
        if reason is not None:
            result[request.id] = reason
    return result
