from __future__ import annotations

import pytest

from satedge_workflow_scheduler.common.intervals import TimeWindow
from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.network import (
    AlwaysAvailable,
    Arc,
    Network,
    Plane,
    WindowedAvailability,
)
from satedge_workflow_scheduler.domain.nodes import GroundStation, Satellite
from satedge_workflow_scheduler.domain.requests import Priority, Request, Task
from satedge_workflow_scheduler.solvers.priority_edf_eft.pending_raw import (
    earliest_raw_departure,
)

EPS = 1e-9
HORIZON = 500.0


def _instance(
    s1_windows: tuple[TimeWindow, ...],
    arcs: tuple[Arc, ...] = (),
    release: float = 0.0,
) -> ProblemInstance:
    s1 = Satellite(
        id="S1", plmm_capacity=10.0, hpc_write_rate=1.0, hpc_windows=s1_windows
    )
    s2 = Satellite(
        id="S2",
        plmm_capacity=10.0,
        hpc_write_rate=1.0,
        hpc_windows=(TimeWindow(0.0, 1000.0),),
    )
    network = Network(
        planes=(
            Plane(id=1, satellites=(s1,), com="S1"),
            Plane(id=2, satellites=(s2,), com="S2"),
        ),
        ground_station=GroundStation(id="G"),
        arcs=arcs,
    )
    request = Request(
        id="r",
        priority=Priority.NON_URGENT,
        origin="S1",
        release=release,
        deadline=900.0,
        raw_size=2.0,
        tasks=(Task(duration=10.0, output_size=1.0),),
    )
    return ProblemInstance(network=network, requests=(request,), horizon=HORIZON)


def _arc(windows: WindowedAvailability | AlwaysAvailable) -> Arc:
    # capacity 1 Gbps, latency 1 s: 2 Gb of raw data take 2 s to send
    return Arc(
        source="S1", target="S2", capacity=1.0, latency=1.0, availability=windows
    )


def _raw(instance: ProblemInstance) -> float:
    return earliest_raw_departure(instance, instance.requests[0], EPS)


def test_hpc_window_before_the_contact_wins() -> None:
    # HPC: starts at 10, the task ends at 20. Contact: opens at 50, sent by 52.
    instance = _instance(
        (TimeWindow(10.0, 100.0),),
        arcs=(_arc(WindowedAvailability((TimeWindow(50.0, 100.0),))),),
    )
    assert _raw(instance) == pytest.approx(20.0)


def test_contact_before_the_hpc_window_wins() -> None:
    # Contact: opens at 5, sent by 7. HPC: starts at 30, ends at 40.
    instance = _instance(
        (TimeWindow(30.0, 100.0),),
        arcs=(_arc(WindowedAvailability((TimeWindow(5.0, 100.0),))),),
    )
    assert _raw(instance) == pytest.approx(7.0)


def test_a_first_task_longer_than_the_first_window_uses_the_next_one() -> None:
    # The first window [0, 5) cannot hold the 10 s task; the next one starts at 20.
    instance = _instance((TimeWindow(0.0, 5.0), TimeWindow(20.0, 100.0)))
    assert _raw(instance) == pytest.approx(30.0)


def test_without_any_resource_the_horizon_is_returned() -> None:
    instance = _instance(())
    assert _raw(instance) == pytest.approx(HORIZON)


def test_the_result_is_capped_at_the_horizon() -> None:
    # the task would end at 505, after the horizon of 500
    instance = _instance((TimeWindow(495.0, 600.0),))
    assert _raw(instance) == pytest.approx(HORIZON)


def test_always_available_arc_frees_the_raw_after_the_transmission() -> None:
    instance = _instance((), arcs=(_arc(AlwaysAvailable()),))
    assert _raw(instance) == pytest.approx(2.0)
