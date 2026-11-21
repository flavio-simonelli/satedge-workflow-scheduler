from __future__ import annotations

import math

import pytest

from satedge_workflow_scheduler.common.errors import InstanceError
from satedge_workflow_scheduler.common.intervals import WindowSet
from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.network import Network, Plane
from satedge_workflow_scheduler.domain.nodes import GroundStation, NodeId, Satellite
from satedge_workflow_scheduler.domain.requests import Priority, Request, Task
from satedge_workflow_scheduler.domain.solution import Schedule, TaskExecution, Transfer

COMMITTED = Schedule(
    executions=(TaskExecution("c", 0, "S1", 0.0, 15.0),),
    transfers=(Transfer("c", 1, "S1", "G", 15.0, 18.0),),
)


def _request(rid: str) -> Request:
    return Request(
        id=rid,
        priority=Priority.URGENT,
        origin="S1",
        release=0.0,
        deadline=500.0,
        raw_size=2.0,
        tasks=(Task(execution_time=10.0, output_size=3.0),),
    )


OFFLINE = Schedule()


def _instance(
    committed: Schedule = OFFLINE,
    plan_release: dict[NodeId, float] | None = None,
) -> ProblemInstance:
    s1 = Satellite("S1", plmm_capacity=10.0, hpc_rate=1.0, hpc_windows=WindowSet())
    network = Network(
        planes=(Plane(id=1, satellites=(s1,), com="S1"),),
        ground_station=GroundStation("G"),
        arcs=(),
    )
    return ProblemInstance(
        network=network,
        requests=(_request("c"), _request("n")),
        horizon=500.0,
        plan_release=(
            {s: 0.0 for s in network.satellite_ids}
            if plan_release is None
            else plan_release
        ),
        committed=committed,
    )


def test_default_instance_is_offline() -> None:
    instance = _instance()
    assert instance.release_of("S1") == 0.0
    assert instance.committed_requests == ()
    assert instance.new_requests == instance.requests


def test_committed_and_new_requests_are_split() -> None:
    instance = _instance(COMMITTED)
    assert [r.id for r in instance.committed_requests] == ["c"]
    assert [r.id for r in instance.new_requests] == ["n"]


def test_plan_release_may_be_infinite() -> None:
    assert _instance(plan_release={"S1": math.inf}).release_of("S1") == math.inf


def test_plan_release_cannot_be_changed_through_the_original_dict() -> None:
    releases = {"S1": 40.0}
    instance = _instance(plan_release=releases)
    releases["S1"] = 0.0
    assert instance.release_of("S1") == 40.0


@pytest.mark.parametrize("release", [-1.0, math.nan])
def test_plan_release_must_be_non_negative(release: float) -> None:
    with pytest.raises(InstanceError, match="plan release"):
        _instance(plan_release={"S1": release})


def test_every_satellite_must_have_a_plan_release() -> None:
    with pytest.raises(InstanceError, match="missing"):
        _instance(plan_release={})


def test_plan_release_only_for_satellites() -> None:
    with pytest.raises(InstanceError, match="not a satellite"):
        _instance(plan_release={"G": 10.0})


def test_committed_calendar_must_refer_to_requests_of_the_instance() -> None:
    unknown = Schedule(executions=(TaskExecution("x", 0, "S1", 0.0, 15.0),))
    with pytest.raises(InstanceError, match="unknown request 'x'"):
        _instance(unknown)


def test_committed_request_needs_one_execution_per_task() -> None:
    # a transfer makes "c" committed, but its task has no execution
    partial = Schedule(transfers=(Transfer("c", 1, "S1", "G", 15.0, 18.0),))
    with pytest.raises(InstanceError, match="one execution per task"):
        _instance(partial)


def test_is_committed_tells_committed_from_new_requests() -> None:
    instance = _instance(COMMITTED)
    assert instance.is_committed("c")
    assert not instance.is_committed("n")
