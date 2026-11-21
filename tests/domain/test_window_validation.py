from __future__ import annotations

import pytest

from satedge_workflow_scheduler.common.errors import InstanceError
from satedge_workflow_scheduler.common.intervals import TimeWindow
from satedge_workflow_scheduler.domain.network import WindowedAvailability
from satedge_workflow_scheduler.domain.nodes import Satellite


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
