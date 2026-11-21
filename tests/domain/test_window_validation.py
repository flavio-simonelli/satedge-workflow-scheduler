from __future__ import annotations

import pytest

from satedge_workflow_scheduler.common.errors import InstanceError
from satedge_workflow_scheduler.common.intervals import TimeWindow
from satedge_workflow_scheduler.domain.network import (
    Arc,
    Network,
    Plane,
    WindowedAvailability,
)
from satedge_workflow_scheduler.domain.nodes import GroundStation, Satellite


def test_satellite_rejects_overlapping_hpc_windows_as_instance_error() -> None:
    with pytest.raises(InstanceError, match="HPC of satellite S1"):
        Satellite(
            id="S1",
            plmm_capacity=10.0,
            hpc_rate=1.0,
            hpc_windows=(TimeWindow(0.0, 6.0), TimeWindow(5.0, 9.0)),
        )


def test_windowed_availability_rejects_unsorted_windows_as_instance_error() -> None:
    with pytest.raises(InstanceError, match="visibility windows"):
        WindowedAvailability((TimeWindow(10.0, 12.0), TimeWindow(0.0, 5.0)))


def _link(windows: tuple[TimeWindow, ...], reverse: tuple[TimeWindow, ...]) -> Network:
    a = Satellite("A", plmm_capacity=10.0, hpc_rate=1.0, hpc_windows=())
    b = Satellite("B", plmm_capacity=10.0, hpc_rate=1.0, hpc_windows=())
    return Network(
        planes=(Plane(1, (a,), "A"), Plane(2, (b,), "B")),
        ground_station=GroundStation("G"),
        arcs=(
            Arc("A", "B", 1.0, 1.0, WindowedAvailability(windows)),
            Arc("B", "A", 1.0, 1.0, WindowedAvailability(reverse)),
        ),
    )


def test_the_two_directions_of_a_link_must_have_the_same_windows() -> None:
    with pytest.raises(InstanceError, match="same availability"):
        _link((TimeWindow(0.0, 10.0),), (TimeWindow(0.0, 12.0),))


def test_a_link_with_the_same_windows_in_both_directions_is_accepted() -> None:
    windows = (TimeWindow(0.0, 10.0),)
    assert len(_link(windows, windows).arcs) == 2
