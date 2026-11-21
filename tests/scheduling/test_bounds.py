from __future__ import annotations

import math

from satedge_workflow_scheduler.common.intervals import TimeWindow, WindowSet
from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.network import (
    AlwaysAvailable,
    Arc,
    Network,
    Plane,
    WindowedAvailability,
)
from satedge_workflow_scheduler.domain.nodes import GroundStation, NodeId, Satellite
from satedge_workflow_scheduler.domain.requests import Priority, Request, Task
from satedge_workflow_scheduler.scheduling.bounds import RequestBounds, request_bounds

# S1 <-> S2 always available, S2 -> G only in [100, 200]; 1 Gbps, latency 1 s.
# HPC at 1 Gbps: S1 computes in [0, 50], S2 in [20, 60].
# Request: raw 2 Gb on S1 from 0, one task of 10 s with a 1 Gb output, so the task lasts
# 2 + 10 + 1 = 13 s. The raw takes 2 + 1 = 3 s on an arc, the output 1 + 1 = 2 s.
#
# EA of the raw:     S1 0, S2 3 (sent at once), never on G.
# EA of the output:  S1 13 (task on S1 in [0, 13]), S2 15 (from S1, better than 33 if
#                    the task ran on S2 from 20), G 102 (waits the window at 100).
# With limit 150:
# LA of the output:  G 150, S2 148, S1 146.
# LA of the raw:     S1 44 (sent to S2 by 44, better than the task on S1 by 37),
#                    S2 47 (task on S2 from 47 ends at 60).
# ES of the task:    S1 0, S2 20.     LC of the task: S1 50, S2 60.


def _instance(
    plan_release: dict[NodeId, float] | None = None,
    s1_windows: tuple[TimeWindow, ...] = (TimeWindow(0.0, 50.0),),
) -> ProblemInstance:
    s1 = Satellite("S1", 10.0, 1.0, WindowSet(s1_windows))
    s2 = Satellite("S2", 10.0, 1.0, WindowSet((TimeWindow(20.0, 60.0),)))
    contact = WindowedAvailability(WindowSet((TimeWindow(100.0, 200.0),)))
    network = Network(
        planes=(Plane(1, (s1,), "S1"), Plane(2, (s2,), "S2")),
        ground_station=GroundStation("G"),
        arcs=(
            Arc("S1", "S2", 1.0, 1.0, AlwaysAvailable()),
            Arc("S2", "S1", 1.0, 1.0, AlwaysAvailable()),
            Arc("S2", "G", 1.0, 1.0, contact),
        ),
    )
    request = Request(
        id="r",
        priority=Priority.URGENT,
        origin="S1",
        release=0.0,
        deadline=500.0,
        raw_size=2.0,
        tasks=(Task(execution_time=10.0, output_size=1.0),),
    )
    return ProblemInstance(
        network=network,
        requests=(request,),
        horizon=500.0,
        plan_release={
            **{s: 0.0 for s in network.satellite_ids},
            **(plan_release or {}),
        },
    )


def _bounds(
    limit: float = 150.0,
    plan_release: dict[NodeId, float] | None = None,
    s1_windows: tuple[TimeWindow, ...] = (TimeWindow(0.0, 50.0),),
) -> RequestBounds:
    instance = _instance(plan_release, s1_windows)
    return request_bounds(instance, instance.requests[0], limit)


def test_earliest_availability_of_the_raw() -> None:
    b = _bounds()
    assert b.earliest_availability(0, "S1") == 0.0
    assert b.earliest_availability(0, "S2") == 3.0


def test_intermediate_data_never_enters_the_ground_station() -> None:
    assert _bounds().earliest_availability(0, "G") == math.inf
    assert _bounds().latest_availability(0, "G") == -math.inf


def test_earliest_availability_of_the_output_takes_the_best_way() -> None:
    b = _bounds()
    assert b.earliest_availability(1, "S1") == 13.0
    # produced on S1 and sent: 15, earlier than running the task on S2 (33)
    assert b.earliest_availability(1, "S2") == 15.0
    # the ground window opens at 100
    assert b.earliest_availability(1, "G") == 102.0


def test_latest_availability_goes_back_from_the_limit() -> None:
    b = _bounds()
    assert b.latest_availability(1, "G") == 150.0
    assert b.latest_availability(1, "S2") == 148.0
    assert b.latest_availability(1, "S1") == 146.0
    # the raw can leave S1 by 44 and still be used on S2 by 47
    assert b.latest_availability(0, "S1") == 44.0
    assert b.latest_availability(0, "S2") == 47.0


def test_earliest_start_and_latest_completion_of_the_task() -> None:
    b = _bounds()
    assert b.earliest_start(0, "S1") == 0.0
    assert b.earliest_start(0, "S2") == 20.0
    assert b.latest_completion(0, "S1") == 50.0
    assert b.latest_completion(0, "S2") == 60.0
    assert b.executable_satellites(0) == ("S1", "S2")
    assert b.compatible_windows(0, "S1") == (TimeWindow(0.0, 50.0),)


def test_a_limit_before_the_ground_contact_makes_every_bound_impossible() -> None:
    # the output cannot reach G by 101: the window opens at 100 and the output takes 2 s
    b = _bounds(limit=101.0)
    assert b.latest_availability(1, "S2") == -math.inf
    assert b.latest_completion(0, "S1") == -math.inf
    assert b.executable_satellites(0) == ()
    assert not b.can_run(0, "S1")


def test_plan_release_delays_the_task_and_the_departures() -> None:
    b = _bounds(plan_release={"S1": 30.0})
    assert b.earliest_start(0, "S1") == 30.0
    # the raw leaves S1 only at 30 and reaches S2 at 33
    assert b.earliest_availability(0, "S2") == 33.0
    assert b.earliest_start(0, "S2") == 33.0


def test_a_satellite_the_plan_never_reaches_cannot_be_used() -> None:
    b = _bounds(plan_release={"S2": math.inf})
    assert not b.can_run(0, "S2")
    # nor relay: the output cannot leave S2 towards G
    assert b.earliest_availability(1, "G") == math.inf


def test_a_window_too_late_for_the_delivery_is_not_compatible() -> None:
    # S1 also computes in [140, 160], but a task ending after 146 leaves no time to send
    # the output to S2 and G by 150: only the first window is compatible.
    b = _bounds(s1_windows=(TimeWindow(0.0, 50.0), TimeWindow(140.0, 160.0)))
    assert b.latest_completion(0, "S1") == 50.0
    assert b.compatible_windows(0, "S1") == (TimeWindow(0.0, 50.0),)
