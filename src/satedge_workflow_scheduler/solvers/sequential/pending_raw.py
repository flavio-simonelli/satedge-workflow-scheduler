"""The raw data of a request not planned yet, reserved on its origin until the last useful instant.

With soft deadlines no request is rejected, so the raw data of every new request will occupy the
PLMM of its origin from its release, until it leaves. Until the request is planned its exit is
unknown. The raw data is then reserved up to ``LA``, the latest instant at which it can still be
on the origin and be used, in some calendar, before the horizon (:mod:`..scheduling.bounds`):

- it covers the transmission of the raw data and its execution on the origin, whichever the
  calendar chooses, so the request planned later always finds the room it needs there;
- it does not reserve the horizon, which would take capacity for nothing after the raw is no
  longer useful.

When the request is planned, its occupancy is computed exactly and replaces this one.
"""

from __future__ import annotations

import math

from satedge_workflow_scheduler.common.intervals import TIME_TOLERANCE
from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.requests import Request
from satedge_workflow_scheduler.scheduling.bounds import request_bounds
from satedge_workflow_scheduler.scheduling.plmm_occupancy import PlmmOccupancyInterval


def pending_raw_occupancy(
    instance: ProblemInstance, request: Request, eps: float = TIME_TOLERANCE
) -> PlmmOccupancyInterval | None:
    """The occupancy ``[release, LA)`` of the raw data on its origin, or ``None`` if it is empty.

    ``LA`` is the latest availability of the raw data on the origin, for the horizon: the soft
    deadline mode delivers every request by the horizon. It is ``None`` when the raw data cannot
    be used at all, which the screening reports as an infeasible request.
    """
    bounds = request_bounds(instance, request, instance.horizon, eps)
    until = min(bounds.latest_availability(0, request.origin), instance.horizon)
    if math.isinf(until) or until <= request.release + eps:
        return None
    return PlmmOccupancyInterval(
        request.id, 0, request.raw_size, request.release, until
    )
