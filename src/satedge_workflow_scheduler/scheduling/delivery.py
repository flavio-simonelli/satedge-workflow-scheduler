"""Delivery of a request: the instant at which its final output reaches the ground station."""

from __future__ import annotations

from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.solution import Schedule
from satedge_workflow_scheduler.scheduling.errors import DeliveryError


def delivery_time(
    instance: ProblemInstance, schedule: Schedule, request_id: str
) -> float:
    """The instant at which the final output of ``request_id`` is entirely on the ground.

    The output is the last data item of the request. Each transmission of it that ends at the
    ground station delivers it at its start, plus its transmission time and the latency of the
    arc; the delivery is the last of them.

    Raises:
        DeliveryError: If no transmission brings the final output to the ground station.
    """
    network = instance.network
    request = instance.request(request_id)
    ground = network.ground_station.id
    final = len(request.tasks)
    arrivals = []
    for transfer in schedule.transfers:
        if (
            transfer.request_id != request_id
            or transfer.data_index != final
            or transfer.target != ground
        ):
            continue
        arc = network.arcs[network.arc_index[(transfer.source, transfer.target)]]
        arrivals.append(
            transfer.start + arc.transmission_time(request.output_size) + arc.latency
        )
    if not arrivals:
        raise DeliveryError(f"the calendar delivers no final output of {request_id}")
    return max(arrivals)
