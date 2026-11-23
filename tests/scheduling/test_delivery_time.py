from __future__ import annotations

import pytest
from sample import sample_instance

from satedge_workflow_scheduler.domain.solution import Schedule, Transfer
from satedge_workflow_scheduler.scheduling.delivery import delivery_time
from satedge_workflow_scheduler.scheduling.errors import DeliveryError


def _to_ground(request_id: str, start: float, data_index: int = 2) -> Transfer:
    return Transfer(request_id, data_index, "S2", "G", start, start + 1.0)


def test_delivery_is_the_start_of_the_transfer_to_ground_plus_transmission_and_latency() -> (
    None
):
    instance = sample_instance()
    arc = instance.network.arcs[instance.network.arc_index[("S2", "G")]]
    schedule = Schedule((), (_to_ground("r1", start=8.0),))
    expected = 8.0 + arc.transmission_time(1.0) + arc.latency
    assert delivery_time(instance, schedule, "r1") == pytest.approx(expected)


def test_transfers_of_other_requests_or_other_data_do_not_deliver() -> None:
    instance = sample_instance()
    schedule = Schedule(
        (),
        (_to_ground("r2", start=8.0), _to_ground("r1", start=8.0, data_index=1)),
    )
    with pytest.raises(DeliveryError):
        delivery_time(instance, schedule, "r1")


def test_a_calendar_without_transfer_to_ground_has_no_delivery() -> None:
    with pytest.raises(DeliveryError):
        delivery_time(sample_instance(), Schedule(), "r1")
