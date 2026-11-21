from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from satedge_workflow_scheduler.common.errors import SolverError
from satedge_workflow_scheduler.common.intervals import TimeWindow
from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.network import (
    AlwaysAvailable,
    Arc,
    Network,
    Plane,
)
from satedge_workflow_scheduler.domain.nodes import GroundStation, Satellite
from satedge_workflow_scheduler.domain.requests import Priority, Request, Task
from satedge_workflow_scheduler.domain.solution import TaskExecution, Transfer
from satedge_workflow_scheduler.scheduling.plmm_occupancy import (
    PlmmOccupancyInterval,
    load_profile,
    overloads,
    plmm_intervals,
    within_capacity,
)


def _peak(items: list[PlmmOccupancyInterval]) -> float:
    """Largest load of the profile, 0 when there is no data."""
    return max((segment.load for segment in load_profile(items)), default=0.0)


def _sat(sid: str) -> Satellite:
    return Satellite(
        id=sid,
        plmm_capacity=10.0,
        hpc_rate=1.0,
        hpc_windows=(TimeWindow(0.0, 1000.0),),
    )


def _arc(source: str, target: str) -> Arc:
    return Arc(
        source=source,
        target=target,
        capacity=1.0,
        latency=1.0,
        availability=AlwaysAvailable(),
    )


def _instance() -> ProblemInstance:
    sats = {sid: _sat(sid) for sid in ("S1", "S2", "S3")}
    network = Network(
        planes=tuple(
            Plane(id=i, satellites=(s,), com=s.id) for i, s in enumerate(sats.values())
        ),
        ground_station=GroundStation(id="G"),
        arcs=(
            _arc("S1", "S2"),
            _arc("S1", "S3"),
            _arc("S3", "S2"),
            _arc("S2", "G"),
        ),
    )
    request = Request(
        id="r",
        priority=Priority.URGENT,
        origin="S1",
        release=0.0,
        deadline=500.0,
        raw_size=2.0,
        tasks=(Task(execution_time=10.0, output_size=3.0),),
    )
    return ProblemInstance(network=network, requests=(request,), horizon=500.0)


def test_direct_route_occupies_origin_raw_and_task_satellite_output() -> None:
    instance = _instance()
    executions = [TaskExecution("r", 0, "S2", start=10.0, end=20.0)]
    transfers = [
        Transfer("r", 0, "S1", "S2", start=0.0, end=2.0),
        Transfer("r", 1, "S2", "G", start=20.0, end=23.0),
    ]

    result = plmm_intervals(instance, executions, transfers)

    assert result == {
        "S1": [PlmmOccupancyInterval("r", 0, 2.0, start=0.0, end=2.0)],
        # raw: arrives at S2 after latency 1, used by the task until 20
        # output: produced by the task, written from 20 - 3 = 17, sent until 23
        "S2": [
            PlmmOccupancyInterval("r", 0, 2.0, start=1.0, end=20.0),
            PlmmOccupancyInterval("r", 1, 3.0, start=17.0, end=23.0),
        ],
    }
    assert "G" not in result


def test_intermediate_satellite_holds_data_between_hops() -> None:
    instance = _instance()
    executions = [TaskExecution("r", 0, "S2", start=10.0, end=20.0)]
    transfers = [
        Transfer("r", 0, "S1", "S3", start=0.0, end=4.0),
        Transfer("r", 0, "S3", "S2", start=6.0, end=9.0),
        Transfer("r", 1, "S2", "G", start=20.0, end=23.0),
    ]

    result = plmm_intervals(instance, executions, transfers)

    # on S3 from the first bit of the first hop (0 + latency 1) to the end of the second hop
    assert PlmmOccupancyInterval("r", 0, 2.0, start=1.0, end=9.0) in result["S3"]
    assert result["S1"] == [PlmmOccupancyInterval("r", 0, 2.0, start=0.0, end=4.0)]


def test_rejected_request_has_no_occupancy() -> None:
    assert plmm_intervals(_instance(), executions=[], transfers=[]) == {}


def test_data_not_reaching_its_task_satellite_is_an_error() -> None:
    instance = _instance()
    executions = [TaskExecution("r", 0, "S2", start=10.0, end=20.0)]
    transfers = [Transfer("r", 1, "S2", "G", start=20.0, end=23.0)]

    with pytest.raises(SolverError):
        plmm_intervals(instance, executions, transfers)


def test_half_open_intervals_do_not_overlap_at_the_boundary() -> None:
    leaving = PlmmOccupancyInterval("a", 0, 2.0, start=0.0, end=5.0)
    entering = PlmmOccupancyInterval("b", 0, 3.0, start=5.0, end=10.0)
    assert _peak([leaving, entering]) == pytest.approx(3.0)


def test_overlapping_intervals_add_up() -> None:
    a = PlmmOccupancyInterval("a", 0, 2.0, start=0.0, end=5.0)
    b = PlmmOccupancyInterval("b", 0, 3.0, start=4.0, end=10.0)
    assert _peak([a, b]) == pytest.approx(5.0)


def test_peak_of_nothing_is_zero() -> None:
    assert _peak([]) == 0.0


def test_fits_compares_peak_with_capacity() -> None:
    items = [
        PlmmOccupancyInterval("a", 0, 2.0, start=0.0, end=5.0),
        PlmmOccupancyInterval("b", 0, 3.0, start=4.0, end=10.0),
    ]
    assert within_capacity(items, capacity=5.0)
    assert not within_capacity(items, capacity=4.0)


_interval = st.tuples(
    st.floats(min_value=0.0, max_value=100.0),
    st.floats(min_value=0.1, max_value=50.0),
    st.floats(min_value=0.1, max_value=10.0),
)


@given(st.lists(_interval, min_size=1, max_size=20))
def test_peak_is_between_largest_size_and_total_size(
    raw: list[tuple[float, float, float]],
) -> None:
    items = [
        PlmmOccupancyInterval("r", i, size, start=start, end=start + length)
        for i, (start, length, size) in enumerate(raw)
    ]
    value = _peak(items)
    sizes = [item.size for item in items]
    assert value >= max(sizes) - 1e-9
    assert value <= sum(sizes) + 1e-9


@given(st.lists(st.floats(min_value=0.1, max_value=10.0), min_size=1, max_size=20))
def test_disjoint_intervals_peak_is_the_largest_size(sizes: list[float]) -> None:
    items = [
        PlmmOccupancyInterval("r", i, size, start=2.0 * i, end=2.0 * i + 1.0)
        for i, size in enumerate(sizes)
    ]
    assert _peak(items) == pytest.approx(max(sizes))


def test_load_profile_splits_time_where_data_starts_or_ends() -> None:
    a = PlmmOccupancyInterval("a", 0, 2.0, start=0.0, end=5.0)
    b = PlmmOccupancyInterval("b", 0, 3.0, start=4.0, end=10.0)
    segments = load_profile([a, b])
    assert [(s.start, s.end, s.load) for s in segments] == [
        (0.0, 4.0, 2.0),
        (4.0, 5.0, 5.0),
        (5.0, 10.0, 3.0),
    ]
    assert segments[1].items == (a, b)


def test_overloads_return_the_segments_above_capacity_with_their_data() -> None:
    a = PlmmOccupancyInterval("a", 0, 2.0, start=0.0, end=5.0)
    b = PlmmOccupancyInterval("b", 0, 3.0, start=4.0, end=10.0)
    profile = load_profile([a, b])
    over = overloads(profile, capacity=4.0)
    assert [(s.start, s.end, s.load) for s in over] == [(4.0, 5.0, 5.0)]
    assert over[0].items == (a, b)
    assert overloads(profile, capacity=5.0) == ()
