"""Earliest instant at which the raw data of a pending request can leave its origin PLMM.

Used by the soft deadline mode: while a request is not planned yet, its raw data is booked in
the PLMM of its origin only up to this instant, which holds in any calendar.
"""

from __future__ import annotations

from satedge_workflow_scheduler.common.intervals import TimeWindow
from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.network import Arc, WindowedAvailability
from satedge_workflow_scheduler.domain.requests import Request


def earliest_raw_departure(
    instance: ProblemInstance, request: Request, eps: float
) -> float:
    """Return the earliest instant ``e_r`` at which the raw data of ``request`` can leave.

    Two ways of freeing the origin PLMM are considered, ignoring contention with other
    requests:

    - the first task runs on the origin in its first usable HPC window; the raw data is
      freed when the task ends;
    - the raw data is sent on an arc towards a satellite; it is freed when the transmission
      ends.

    The result is the earliest of these instants, capped at the horizon. If neither way is
    possible, the horizon is returned.
    """
    network = instance.network
    origin = network.satellite(request.origin)
    first_duration = request.hpc_duration(0, origin)
    candidates: list[float] = []

    for window in origin.hpc_windows:
        start = max(request.release, window.start)
        if start + first_duration <= window.end + eps:
            candidates.append(start + first_duration)
            break

    ground = network.ground_station.id
    for index in network.outgoing[request.origin]:
        arc = network.arcs[index]
        if arc.target == ground:
            continue
        tx = arc.transmission_time(request.raw_size)
        departure = _first_departure(arc, request.release, tx, eps)
        if departure is not None:
            candidates.append(departure + tx)

    if not candidates:
        return instance.horizon
    return min(min(candidates), instance.horizon)


def _first_departure(arc: Arc, ready: float, tx: float, eps: float) -> float | None:
    """First instant at which a transmission of ``tx`` can start on an empty ``arc``."""
    if not isinstance(arc.availability, WindowedAvailability):
        return ready
    for window in arc.availability.windows:
        start = _start_in(window, ready, tx, arc.latency, eps)
        if start is not None:
            return start
    return None


def _start_in(
    window: TimeWindow, ready: float, tx: float, latency: float, eps: float
) -> float | None:
    """Start inside ``window`` that lets transmission and latency finish in it, if any."""
    start = max(ready, window.start)
    if start + tx + latency <= window.end + eps:
        return start
    return None
