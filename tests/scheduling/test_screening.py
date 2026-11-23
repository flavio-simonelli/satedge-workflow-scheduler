from __future__ import annotations

from satedge_workflow_scheduler.common.intervals import TimeWindow, WindowSet
from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.network import (
    AlwaysAvailable,
    Arc,
    Network,
    Plane,
)
from satedge_workflow_scheduler.domain.nodes import GroundStation, Satellite
from satedge_workflow_scheduler.domain.requests import Priority, Request, Task
from satedge_workflow_scheduler.domain.solution import Schedule, TaskExecution
from satedge_workflow_scheduler.scheduling.bounds import request_bounds
from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.scheduling.screening import (
    unrealizable_reason,
    unrealizable_requests,
)

# S1 <-> S2 and S2 -> G always available, 1 Gbps, latency 1 s. Every HPC at 1 Gbps.
# Raw 2 Gb on S1, one task of 10 s with a 3 Gb output: the task lasts 2 + 10 + 3 = 15 s,
# the raw takes 3 s on an arc and the output 4 s.
# Released at 0, the best delivery is 22: raw to S2 by 3, task on S2 in [3, 18], output
# to G by 22 (on S1 the task would end at 15, but the output needs two hops: 23).


def _reason(request: Request, instance: ProblemInstance, limit: float) -> str | None:
    bounds = request_bounds(instance, request, limit)
    return unrealizable_reason(request, bounds, instance.network)


def _instance(*requests: Request, window: float = 200.0) -> ProblemInstance:
    satellites = [
        Satellite(sid, 10.0, 1.0, WindowSet((TimeWindow(0.0, window),)))
        for sid in ("S1", "S2")
    ]
    network = Network(
        planes=(Plane(1, (satellites[0],), "S1"), Plane(2, (satellites[1],), "S2")),
        ground_station=GroundStation("G"),
        arcs=(
            Arc("S1", "S2", 1.0, 1.0, AlwaysAvailable()),
            Arc("S2", "S1", 1.0, 1.0, AlwaysAvailable()),
            Arc("S2", "G", 1.0, 1.0, AlwaysAvailable()),
        ),
    )
    return ProblemInstance(
        network=network,
        requests=requests,
        horizon=300.0,
        plan_release={s: 0.0 for s in network.satellite_ids},
    )


def _request(rid: str = "r", release: float = 0.0, deadline: float = 100.0) -> Request:
    return Request(
        id=rid,
        priority=Priority.URGENT,
        origin="S1",
        release=release,
        deadline=deadline,
        raw_size=2.0,
        tasks=(Task(execution_time=10.0, output_size=3.0),),
    )


def test_a_feasible_request_has_no_reason() -> None:
    assert _reason(_request(), _instance(_request()), limit=22.0) is None


def test_a_task_longer_than_every_window_fits_nowhere() -> None:
    instance = _instance(_request(), window=10.0)
    reason = _reason(_request(), instance, limit=100.0)
    assert reason == "task 0 fits in no computing window"


def test_the_delivery_to_the_ground_counts() -> None:
    # The pipeline alone ends at 15, before the limit of 21, but its output cannot
    # reach the ground before 22.
    reason = _reason(_request(), _instance(_request()), limit=21.0)
    assert reason == "task 0 can run on no satellite in time to deliver by 21.0 s"


def test_hard_mode_screens_against_the_deadline() -> None:
    late = _request("late", release=90.0, deadline=100.0)
    on_time = _request("ok", release=0.0, deadline=100.0)
    result = unrealizable_requests(_instance(late, on_time), DeadlineMode.HARD)
    assert result.keys() == {"late"}


def test_soft_mode_screens_against_the_horizon() -> None:
    # the deadline of 100 is missed, but the horizon of 300 is not
    late = _request("late", release=90.0, deadline=100.0)
    assert unrealizable_requests(_instance(late), DeadlineMode.SOFT) == {}


def test_committed_requests_are_not_screened() -> None:
    late = _request("late", release=90.0, deadline=100.0)
    instance = _instance(late)
    committed = ProblemInstance(
        network=instance.network,
        requests=(late,),
        horizon=300.0,
        plan_release=instance.plan_release,
        committed=Schedule(executions=(TaskExecution("late", 0, "S2", 93.0, 108.0),)),
    )
    assert unrealizable_requests(committed, DeadlineMode.HARD) == {}
