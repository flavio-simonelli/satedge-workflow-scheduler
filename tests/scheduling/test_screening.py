from __future__ import annotations

from satedge_workflow_scheduler.common.intervals import TimeWindow
from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.network import Network, Plane
from satedge_workflow_scheduler.domain.nodes import GroundStation, Satellite
from satedge_workflow_scheduler.domain.requests import Priority, Request, Task
from satedge_workflow_scheduler.domain.solution import Schedule, TaskExecution
from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.scheduling.screening import (
    executable_satellites,
    unrealizable_reason,
    unrealizable_requests,
)

# Every satellite has a 1 Gbps HPC. Raw 2 Gb, one task of 10 s with a 3 Gb output:
# the task occupies the HPC for 2 + 10 + 3 = 15 s.


def _network(s1_window: float = 20.0, s2_window: float = 20.0) -> Network:
    s1 = Satellite("S1", 10.0, hpc_rate=1.0, hpc_windows=(TimeWindow(0.0, s1_window),))
    s2 = Satellite("S2", 10.0, hpc_rate=1.0, hpc_windows=(TimeWindow(0.0, s2_window),))
    return Network(
        planes=(
            Plane(id=1, satellites=(s1,), com="S1"),
            Plane(id=2, satellites=(s2,), com="S2"),
        ),
        ground_station=GroundStation("G"),
        arcs=(),
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


def test_a_task_runs_only_where_a_window_is_long_enough() -> None:
    # the 15 s task fits in the 20 s window of S2, not in the 14 s window of S1
    network = _network(s1_window=14.0, s2_window=20.0)
    assert executable_satellites(_request(), 0, network) == ("S2",)


def test_a_window_exactly_as_long_as_the_task_is_enough() -> None:
    network = _network(s1_window=15.0, s2_window=15.0)
    assert executable_satellites(_request(), 0, network) == ("S1", "S2")


def test_a_task_longer_than_every_window_makes_the_request_unrealizable() -> None:
    network = _network(s1_window=10.0, s2_window=10.0)
    reason = unrealizable_reason(_request(), network, limit=100.0)
    assert reason == "task 0 fits in no computing window"


def test_the_pipeline_must_end_by_the_limit() -> None:
    # release 80 + 15 s of HPC = 95: it ends by 95, not by 94
    network = _network()
    assert unrealizable_reason(_request(release=80.0), network, limit=95.0) is None
    reason = unrealizable_reason(_request(release=80.0), network, limit=94.0)
    assert reason is not None and "95" in reason


def _instance(*requests: Request, committed: Schedule | None = None) -> ProblemInstance:
    return ProblemInstance(
        network=_network(),
        requests=requests,
        horizon=200.0,
        committed=committed or Schedule(),
    )


def test_hard_mode_screens_against_the_deadline() -> None:
    late = _request("late", release=90.0, deadline=100.0)
    on_time = _request("ok", release=0.0, deadline=100.0)
    result = unrealizable_requests(_instance(late, on_time), DeadlineMode.HARD)
    assert result.keys() == {"late"}


def test_soft_mode_screens_against_the_horizon() -> None:
    # the deadline of 100 is missed, but the horizon of 200 is not
    late = _request("late", release=90.0, deadline=100.0)
    assert unrealizable_requests(_instance(late), DeadlineMode.SOFT) == {}


def test_committed_requests_are_not_screened() -> None:
    late = _request("late", release=90.0, deadline=100.0)
    committed = Schedule(executions=(TaskExecution("late", 0, "S2", 0.0, 15.0),))
    instance = _instance(late, committed=committed)
    assert unrealizable_requests(instance, DeadlineMode.HARD) == {}
