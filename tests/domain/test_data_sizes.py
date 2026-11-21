from __future__ import annotations

import pytest

from satedge_workflow_scheduler.common.errors import InstanceError
from satedge_workflow_scheduler.domain.nodes import Satellite
from satedge_workflow_scheduler.domain.requests import Priority, Request, Task


def test_task_output_must_be_positive() -> None:
    with pytest.raises(InstanceError):
        Task(execution_time=10.0, output_size=0.0)


def test_raw_data_must_be_positive() -> None:
    with pytest.raises(InstanceError):
        Request(
            id="r",
            priority=Priority.URGENT,
            origin="S1",
            release=0.0,
            deadline=100.0,
            raw_size=0.0,
            tasks=(Task(execution_time=10.0, output_size=1.0),),
        )


def test_hpc_duration_adds_the_read_and_write_times_of_the_satellite() -> None:
    # raw 2 Gb, output 3 Gb: on a 1 Gbps HPC 2 + 10 + 3 s, on a 2 Gbps HPC 1 + 10 + 1.5 s
    request = Request(
        id="r",
        priority=Priority.URGENT,
        origin="S1",
        release=0.0,
        deadline=100.0,
        raw_size=2.0,
        tasks=(Task(execution_time=10.0, output_size=3.0),),
    )
    slow = Satellite(id="S1", plmm_capacity=10.0, hpc_rate=1.0, hpc_windows=())
    fast = Satellite(id="S2", plmm_capacity=10.0, hpc_rate=2.0, hpc_windows=())
    assert request.hpc_duration(0, slow) == pytest.approx(15.0)
    assert request.hpc_duration(0, fast) == pytest.approx(12.5)


def test_hpc_rate_must_be_positive() -> None:
    with pytest.raises(InstanceError):
        Satellite(id="S1", plmm_capacity=10.0, hpc_rate=0.0, hpc_windows=())
