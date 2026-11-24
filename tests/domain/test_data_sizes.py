from __future__ import annotations

import pytest

from satedge_workflow_scheduler.common.intervals import WindowSet
from satedge_workflow_scheduler.domain.errors import InstanceError
from satedge_workflow_scheduler.domain.nodes import Satellite
from satedge_workflow_scheduler.domain.requests import Priority, Request, Task


def test_task_output_must_be_positive() -> None:
    with pytest.raises(InstanceError):
        Task(execution_time=10.0, output_size=0.0)


def test_raw_data_must_be_positive() -> None:
    task = Task(execution_time=10.0, output_size=1.0)
    with pytest.raises(InstanceError):
        Request(
            id="r",
            priority=Priority.URGENT,
            origin="S1",
            release=0.0,
            deadline=100.0,
            raw_size=0.0,
            tasks=(task,),
        )


def _request() -> Request:
    return Request(
        id="r",
        priority=Priority.URGENT,
        origin="S1",
        release=0.0,
        deadline=100.0,
        raw_size=2.0,
        tasks=(Task(execution_time=10.0, output_size=3.0),),
    )


def test_hpc_duration_adds_the_read_and_write_times() -> None:
    # raw 2 Gb and output 3 Gb on a 1 Gbps HPC: read 2 s, run 10 s, write 3 s
    s1 = Satellite(id="S1", plmm_capacity=10.0, hpc_rate=1.0, hpc_windows=WindowSet())
    assert _request().hpc_duration(0, s1) == pytest.approx(15.0)


def test_hpc_duration_follows_the_rate_of_the_satellite() -> None:
    # The case study has 1 Gbps on every satellite, but the rate is a satellite
    # parameter: at 2 Gbps reading and writing take half the time.
    s2 = Satellite(id="S2", plmm_capacity=10.0, hpc_rate=2.0, hpc_windows=WindowSet())
    assert _request().hpc_duration(0, s2) == pytest.approx(1.0 + 10.0 + 1.5)


def test_hpc_rate_must_be_positive() -> None:
    with pytest.raises(InstanceError):
        Satellite(id="S1", plmm_capacity=10.0, hpc_rate=0.0, hpc_windows=WindowSet())
