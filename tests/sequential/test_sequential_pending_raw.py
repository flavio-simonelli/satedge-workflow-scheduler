from __future__ import annotations

import pytest
from sample import sample_instance

from satedge_workflow_scheduler.scheduling.bounds import request_bounds
from satedge_workflow_scheduler.solvers.sequential.pending_raw import (
    pending_raw_occupancy,
)


def test_the_pending_raw_is_reserved_from_its_release_up_to_its_latest_availability() -> (
    None
):
    instance = sample_instance()
    request = instance.request("r1")
    bounds = request_bounds(instance, request, instance.horizon)
    item = pending_raw_occupancy(instance, request)
    assert item is not None
    assert item.start == request.release
    assert item.end == pytest.approx(bounds.latest_availability(0, request.origin))
    assert item.size == request.raw_size


def test_the_reservation_does_not_go_beyond_the_horizon() -> None:
    instance = sample_instance()
    item = pending_raw_occupancy(instance, instance.request("r2"))
    assert item is not None
    assert item.end <= instance.horizon
