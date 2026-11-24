"""Screening of the requests: what can never be scheduled, whatever the other requests do.

A request is impossible when one of its tasks can run on no satellite even with the request
alone in the network: no satellite has ``ES_{i,s} + L̄_{i,s} <= LC_{i,s}`` (section
"Richieste impossibili" of the model, with the bounds of :mod:`.bounds`). This covers, for
example, a task longer than every computing window, a pipeline that cannot end by its limit
even without transfers, and a delivery that misses every ground contact.

These are conditions on the data of the model, not errors: an instance that violates them is
valid, and the requests concerned are excluded a priori. With hard deadlines they are
rejected; with soft deadlines, where no request can be rejected, the instance is infeasible.
"""

from __future__ import annotations

from satedge_workflow_scheduler.common.intervals import TIME_TOLERANCE
from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.network import Network
from satedge_workflow_scheduler.domain.requests import Request
from satedge_workflow_scheduler.scheduling.bounds import RequestBounds, request_bounds
from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode, delivery_limit


def unrealizable_reason(
    request: Request, bounds: RequestBounds, network: Network
) -> str | None:
    """Why ``request`` can never be delivered by ``bounds.limit``, or ``None`` if it may be.

    ``bounds`` are the a priori bounds of the request (:func:`request_bounds`), which the
    caller usually needs anyway. The reason, in Italian since it reaches the user, names the
    first task that can run on no satellite (numbered from 1), and says whether it is longer
    than every computing window or only cannot be placed in time.
    """
    for task_index in range(len(request.tasks)):
        if bounds.executable_satellites(task_index):
            continue
        fits_somewhere = any(
            w.duration >= request.hpc_duration(task_index, s) - TIME_TOLERANCE
            for s in network.satellites
            for w in s.hpc_windows
        )
        if not fits_somewhere:
            return f"il task {task_index + 1} non entra in nessuna finestra di calcolo"
        return (
            f"il task {task_index + 1} non può essere eseguito su nessun satellite "
            f"in tempo per consegnare entro {bounds.limit:g} s"
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
        bounds = request_bounds(instance, request, limit)
        reason = unrealizable_reason(request, bounds, instance.network)
        if reason is not None:
            result[request.id] = reason
    return result
