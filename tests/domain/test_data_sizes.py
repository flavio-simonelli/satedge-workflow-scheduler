from __future__ import annotations

import pytest

from satedge_workflow_scheduler.common.errors import InstanceError
from satedge_workflow_scheduler.domain.requests import Priority, Request, Task


def test_task_output_must_be_positive() -> None:
    with pytest.raises(InstanceError):
        Task(duration=10.0, output_size=0.0)


def test_raw_data_must_be_positive() -> None:
    with pytest.raises(InstanceError):
        Request(
            id="r",
            priority=Priority.URGENT,
            origin="S1",
            release=0.0,
            deadline=100.0,
            raw_size=0.0,
            tasks=(Task(duration=10.0, output_size=1.0),),
        )
